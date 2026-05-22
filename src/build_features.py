from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr

from paths import (
    CLIPPED_DIR,
    PROCESSED_ERA5_DIR,
    PROCESSED_FEATURES_DIR,
    PROCESSED_GLOFAS_DIR,
    PROCESSED_SRTM_DIR,
    PROJECT_ROOT,
)


ERA5_MULTIYEAR = PROCESSED_ERA5_DIR / "era5_sindh_multiyear_combined.nc"
GLOFAS_MULTIYEAR = PROCESSED_GLOFAS_DIR / "glofas_sindh_multiyear_clean.nc"
ERA5_CLIPPED_2020 = CLIPPED_DIR / "era5_sindh_2020_clipped.nc"
GLOFAS_CLIPPED_2020 = CLIPPED_DIR / "glofas_sindh_2020_clipped.nc"
SRTM_CLIPPED = CLIPPED_DIR / "srtm_sindh_clipped.tif"
SRTM_MOSAIC = PROCESSED_SRTM_DIR / "srtm_sindh_mosaic.tif"

RAINFALL_ONLY_OUTPUT = PROCESSED_FEATURES_DIR / "flood_features_rainfall_only.csv"
HYDROLOGY_OUTPUT = PROCESSED_FEATURES_DIR / "flood_features_hydrology.csv"
DEFAULT_OUTPUT = PROCESSED_FEATURES_DIR / "flood_ml_features.csv"

SPATIAL_DIMS = ("lat", "lon")
DAILY_SUM_VARIABLES = {"tp", "sro", "ssro", "ro", "e", "evaporation"}
TRAIN_FRACTION_FOR_Q95 = 0.65


def _default_existing_path(primary: Path, fallback: Path) -> Path:
    return primary if primary.exists() else fallback


def _find_variable(ds: xr.Dataset, preferred: list[str], keywords: list[str]) -> str:
    for name in preferred:
        if name in ds.data_vars:
            return name

    for name, variable in ds.data_vars.items():
        text = " ".join(
            str(variable.attrs.get(key, "")).lower()
            for key in ["standard_name", "long_name", "description", "units", "GRIB_name"]
        )
        if any(keyword in text for keyword in keywords):
            return name

    raise ValueError(f"Could not identify variable. Available variables: {list(ds.data_vars)}")


def _spatial_dims(data: xr.DataArray) -> list[str]:
    return [dim for dim in SPATIAL_DIMS if dim in data.dims]


def _spatial_stats(data: xr.DataArray, prefix: str) -> pd.DataFrame:
    dims = _spatial_dims(data)
    if not dims:
        series = data.to_series()
        return series.rename(f"{prefix}_value").to_frame()

    stats = {
        f"{prefix}_mean": data.mean(dim=dims, skipna=True),
        f"{prefix}_min": data.min(dim=dims, skipna=True),
        f"{prefix}_max": data.max(dim=dims, skipna=True),
        f"{prefix}_std": data.std(dim=dims, skipna=True),
    }
    return xr.Dataset(stats).to_dataframe()[list(stats)]


def _normalise_daily_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def aggregate_era5_daily_stats(input_path: Path | None = None) -> pd.DataFrame:
    input_path = input_path or _default_existing_path(ERA5_MULTIYEAR, ERA5_CLIPPED_2020)
    print("\nAggregating ERA5 daily spatial statistics...")
    ds = xr.open_dataset(input_path, engine="netcdf4")
    frames: list[pd.DataFrame] = []
    try:
        if "time" not in ds.dims and "time" not in ds.coords:
            raise ValueError(f"ERA5 dataset has no time dimension: {input_path}")

        for var_name, data in ds.data_vars.items():
            if "time" not in data.dims or var_name == "spatial_ref":
                continue

            if var_name in DAILY_SUM_VARIABLES:
                daily = data.resample(time="1D").sum(skipna=True, min_count=1)
                aggregation = "daily_total"
            else:
                daily = data.resample(time="1D").mean(skipna=True)
                aggregation = "daily_mean"

            frames.append(_spatial_stats(daily, f"era5_{var_name}_{aggregation}"))
    finally:
        ds.close()

    if not frames:
        raise ValueError(f"No time-varying ERA5 variables found in {input_path}")

    df = _normalise_daily_dates(pd.concat(frames, axis=1).reset_index())
    print(f"  ERA5 daily feature rows: {len(df)}")
    return df


def aggregate_glofas_discharge_stats(input_path: Path | None = None) -> pd.DataFrame:
    input_path = input_path or _default_existing_path(GLOFAS_MULTIYEAR, GLOFAS_CLIPPED_2020)
    print("\nAggregating GloFAS discharge statistics...")
    ds = xr.open_dataset(input_path, engine="netcdf4")
    try:
        if "time" not in ds.dims and "time" not in ds.coords:
            raise ValueError(f"GloFAS dataset has no time dimension: {input_path}")

        discharge_var = _find_variable(ds, ["dis24", "dis", "discharge"], ["discharge", "river"])
        data = ds[discharge_var]
        if "time" not in data.dims:
            raise ValueError(f"GloFAS discharge variable has no time dimension: {discharge_var}")

        daily = data.resample(time="1D").mean(skipna=True)
        df = _normalise_daily_dates(_spatial_stats(daily, "glofas_discharge").reset_index())
    finally:
        ds.close()

    print(f"  GloFAS daily feature rows: {len(df)}")
    return df


def _first_existing_column(df: pd.DataFrame, candidates: list[str], contains: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    for column in df.columns:
        text = column.lower()
        if all(part in text for part in contains):
            return column
    return None


def _numeric_columns_matching(df: pd.DataFrame, include_any: tuple[str, ...]) -> list[str]:
    return [
        column
        for column in df.select_dtypes(include="number").columns
        if any(token in column.lower() for token in include_any)
        and not column.startswith(("label_", "target_"))
    ]


def _add_lags(result: pd.DataFrame, columns: list[str], lags: tuple[int, ...], family: str) -> pd.DataFrame:
    lagged_frames = []
    for lag in lags:
        lagged = result[columns].shift(lag)
        lagged.columns = [f"{column}_{family}_lag_{lag}d" for column in columns]
        lagged_frames.append(lagged)
    return pd.concat([result, *lagged_frames], axis=1) if lagged_frames else result


def add_rainfall_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (3, 7, 14, 30),
) -> pd.DataFrame:
    print("\nCreating rainfall features...")
    result = df.sort_values("date").copy()
    rainfall_column = _first_existing_column(
        result,
        ["era5_tp_daily_total_mean", "era5_tp_daily_mean_mean"],
        ["era5", "tp", "mean"],
    )
    if rainfall_column is None:
        raise ValueError("Could not identify an ERA5 rainfall column for feature engineering.")

    for window in windows:
        rolling = result[rainfall_column].rolling(window=window, min_periods=window)
        result[f"rainfall_total_{window}d"] = rolling.sum()
        result[f"rainfall_mean_{window}d"] = rolling.mean()

    climatology = result.groupby(result["date"].dt.dayofyear)[rainfall_column].transform("mean")
    result["rainfall_anomaly"] = result[rainfall_column] - climatology
    result["rainfall_intensity"] = result[rainfall_column].where(result[rainfall_column] > 0, 0)

    monsoon = result["date"].dt.month.between(6, 9)
    monsoon_rain = result[rainfall_column].where(monsoon, 0)
    result["cumulative_monsoon_rainfall"] = monsoon_rain.groupby(result["date"].dt.year).cumsum()
    print("  Added rolling totals, means, anomaly, intensity, and monsoon accumulation.")
    return result


def add_hydrology_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3, 7, 14),
    rolling_windows: tuple[int, ...] = (3, 7, 14, 30),
) -> pd.DataFrame:
    print("\nCreating hydrology features...")
    result = df.sort_values("date").copy()
    discharge_column = _first_existing_column(
        result,
        ["glofas_discharge_max", "glofas_discharge_mean"],
        ["glofas", "discharge"],
    )
    if discharge_column is None:
        raise ValueError("Could not identify a GloFAS discharge column for feature engineering.")

    result = _add_lags(result, [discharge_column], lags, "discharge")
    for window in rolling_windows:
        result[f"discharge_mean_{window}d"] = (
            result[discharge_column].rolling(window=window, min_periods=window).mean()
        )
    result["discharge_change_1d"] = result[discharge_column].diff()
    result["discharge_acceleration_1d"] = result["discharge_change_1d"].diff()

    runoff_columns = _numeric_columns_matching(result, ("sro", "runoff"))
    soil_columns = _numeric_columns_matching(result, ("swvl", "soil"))
    result = _add_lags(result, runoff_columns, (1, 3, 7), "runoff")
    result = _add_lags(result, soil_columns, (1, 3, 7), "soil_moisture")
    print(f"  Added hydrology columns from {1 + len(runoff_columns) + len(soil_columns)} source series.")
    return result


def add_atmospheric_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (7, 14),
) -> pd.DataFrame:
    print("\nCreating atmospheric features...")
    result = df.sort_values("date").copy()

    pressure_columns = _numeric_columns_matching(result, ("pressure", "sp_", "msl"))
    wind_columns = _numeric_columns_matching(result, ("wind", "u10", "v10"))
    evaporation_columns = _numeric_columns_matching(result, ("evap", "era5_e_"))

    for column in pressure_columns:
        climatology = result.groupby(result["date"].dt.dayofyear)[column].transform("mean")
        result[f"{column}_anomaly"] = result[column] - climatology

    for column in wind_columns:
        for window in windows:
            result[f"{column}_mean_{window}d"] = (
                result[column].rolling(window=window, min_periods=window).mean()
            )
            result[f"{column}_std_{window}d"] = (
                result[column].rolling(window=window, min_periods=window).std()
            )

    for column in evaporation_columns:
        result[f"{column}_trend_7d"] = result[column] - result[column].shift(7)

    print(
        "  Added atmospheric derivatives for "
        f"{len(pressure_columns)} pressure, {len(wind_columns)} wind, "
        f"{len(evaporation_columns)} evaporation columns."
    )
    return result


def _terrain_stats(path: Path | None = None) -> dict[str, float]:
    path = path or _default_existing_path(SRTM_CLIPPED, SRTM_MOSAIC)
    if not path.exists():
        print("  Terrain raster not found; terrain features will be omitted.")
        return {}

    with rasterio.open(path) as src:
        sample_size = 100
        rows = np.linspace(0, src.height - 1, num=min(sample_size, src.height)).astype(int)
        cols = np.linspace(0, src.width - 1, num=min(sample_size, src.width)).astype(int)
        coords = [src.xy(row, col) for row in rows for col in cols]
        sampled_values = []
        for value in src.sample(coords, masked=True):
            cell = value[0]
            sampled_values.append(np.nan if np.ma.is_masked(cell) else float(cell))
        sampled = np.array(sampled_values, dtype="float64")
        sampled = sampled.reshape(len(rows), len(cols))

        if src.nodata is not None:
            sampled[sampled == src.nodata] = np.nan
        values = sampled[np.isfinite(sampled)]
        if values.size == 0:
            return {}

        y_grad, x_grad = np.gradient(sampled)
        slope = np.sqrt(x_grad**2 + y_grad**2)
        local_relief = float(np.nanpercentile(sampled, 95) - np.nanpercentile(sampled, 5))

    return {
        "terrain_elevation_mean": float(np.nanmean(values)),
        "terrain_elevation_min": float(np.nanmin(values)),
        "terrain_elevation_max": float(np.nanmax(values)),
        "terrain_elevation_std": float(np.nanstd(values)),
        "terrain_slope_mean": float(np.nanmean(slope)),
        "terrain_slope_max": float(np.nanmax(slope)),
        "terrain_slope_std": float(np.nanstd(slope)),
        "terrain_local_relief_p95_p05": local_relief,
    }


def add_terrain_features(df: pd.DataFrame, terrain_path: Path | None = None) -> pd.DataFrame:
    print("\nCreating terrain features...")
    result = df.copy()
    stats = _terrain_stats(terrain_path)
    for name, value in stats.items():
        result[name] = value
    print(f"  Added {len(stats)} terrain columns.")
    return result


def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["month"] = result["date"].dt.month
    result["day_of_year"] = result["date"].dt.dayofyear
    result["is_monsoon"] = result["month"].between(6, 9).astype(int)
    return result


def add_multi_horizon_labels(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 7, 14),
    train_fraction: float = TRAIN_FRACTION_FOR_Q95,
) -> pd.DataFrame:
    print("\nCreating multi-horizon q95 labels...")
    result = df.sort_values("date").copy()
    discharge_column = _first_existing_column(
        result,
        ["glofas_discharge_max", "glofas_discharge_mean"],
        ["glofas", "discharge"],
    )
    if discharge_column is None:
        raise ValueError("Could not identify a GloFAS discharge column for target labels.")

    split_index = max(1, min(len(result) - 1, int(len(result) * train_fraction)))
    threshold = result.iloc[:split_index][discharge_column].quantile(0.95)
    print(f"  q95 threshold from first {split_index} chronological rows: {threshold:.3f}")

    future_series = result[discharge_column].shift(-1)
    for horizon in horizons:
        future_max = (
            future_series.iloc[::-1]
            .rolling(window=horizon, min_periods=1)
            .max()
            .iloc[::-1]
        )
        target_column = f"target_discharge_max_next_{horizon}d"
        label_column = f"label_discharge_next_{horizon}d_ge_q95"
        result[target_column] = future_max
        result[label_column] = (future_max >= threshold).astype("Int64")
        result.loc[future_max.isna(), label_column] = pd.NA

    return result


def _validate_daily_continuity(df: pd.DataFrame) -> None:
    dates = pd.DatetimeIndex(df["date"])
    duplicate_count = int(dates.duplicated().sum())
    missing = pd.date_range(dates.min(), dates.max(), freq="D").difference(dates)
    diffs = dates.to_series().diff().dropna()
    irregular = int((diffs != pd.Timedelta(days=1)).sum()) if len(diffs) else 0
    print(
        "\nFeature date checks: "
        f"{dates.min().date()} to {dates.max().date()}, "
        f"missing={len(missing)}, duplicates={duplicate_count}, irregular_steps={irregular}"
    )


def build_feature_dataframe(
    era5_path: Path | None = None,
    glofas_path: Path | None = None,
    terrain_path: Path | None = None,
    drop_incomplete_rows: bool = True,
) -> pd.DataFrame:
    era5 = aggregate_era5_daily_stats(era5_path)
    glofas = aggregate_glofas_discharge_stats(glofas_path)

    print("\nJoining daily ERA5 and GloFAS features...")
    df = pd.merge(era5, glofas, on="date", how="inner").sort_values("date")
    print(f"  Joined rows: {len(df)}")
    _validate_daily_continuity(df)

    df = add_seasonality_features(df)
    df = add_rainfall_features(df)
    df = add_atmospheric_features(df)
    df = add_hydrology_features(df)
    df = add_terrain_features(df, terrain_path)
    df = add_multi_horizon_labels(df)

    if drop_incomplete_rows:
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        print(f"\nDropped incomplete rows from lags/rolling windows/targets: {before - len(df)}")

    return df


def select_feature_configuration(df: pd.DataFrame, include_hydrology: bool) -> pd.DataFrame:
    label_columns = [column for column in df.columns if column.startswith("label_")]
    target_columns = [column for column in df.columns if column.startswith("target_")]
    base_columns = ["date", *target_columns, *label_columns]

    if include_hydrology:
        selected = list(df.columns)
    else:
        hydrology_tokens = (
            "glofas",
            "discharge",
            "runoff",
            "sro",
            "swvl",
            "soil",
        )
        selected = [
            column
            for column in df.columns
            if column in base_columns
            or not any(token in column.lower() for token in hydrology_tokens)
        ]

    return df.loc[:, selected]


def build_feature_csv(
    era5_path: Path | None = None,
    glofas_path: Path | None = None,
    rainfall_only_output: Path = RAINFALL_ONLY_OUTPUT,
    hydrology_output: Path = HYDROLOGY_OUTPUT,
    legacy_output: Path = DEFAULT_OUTPUT,
) -> tuple[Path, Path]:
    print("\nBuilding ML-ready flood feature CSVs...")
    full_df = build_feature_dataframe(era5_path=era5_path, glofas_path=glofas_path)
    rainfall_df = select_feature_configuration(full_df, include_hydrology=False)
    hydrology_df = select_feature_configuration(full_df, include_hydrology=True)

    rainfall_only_output.parent.mkdir(parents=True, exist_ok=True)
    rainfall_df.to_csv(rainfall_only_output, index=False)
    hydrology_df.to_csv(hydrology_output, index=False)
    hydrology_df.to_csv(legacy_output, index=False)

    print(f"\nSaved rainfall-only features: {rainfall_only_output.relative_to(PROJECT_ROOT)}")
    print(f"Rows: {len(rainfall_df)} | Columns: {len(rainfall_df.columns)}")
    print(f"Saved hydrology features: {hydrology_output.relative_to(PROJECT_ROOT)}")
    print(f"Rows: {len(hydrology_df)} | Columns: {len(hydrology_df.columns)}")
    print(f"Backward-compatible ML CSV: {legacy_output.relative_to(PROJECT_ROOT)}")
    return rainfall_only_output, hydrology_output


def main() -> None:
    build_feature_csv()


if __name__ == "__main__":
    main()
