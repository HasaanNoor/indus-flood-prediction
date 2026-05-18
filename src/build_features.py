from pathlib import Path

import pandas as pd
import xarray as xr

from paths import CLIPPED_DIR, PROCESSED_FEATURES_DIR, PROJECT_ROOT


ERA5_CLIPPED = CLIPPED_DIR / "era5_sindh_2020_clipped.nc"
GLOFAS_CLIPPED = CLIPPED_DIR / "glofas_sindh_2020_clipped.nc"
DEFAULT_OUTPUT = PROCESSED_FEATURES_DIR / "flood_ml_features.csv"

SPATIAL_DIMS = ("lat", "lon")
ACCUMULATED_ERA5_VARIABLES = {"tp", "sro"}


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


def aggregate_era5_daily_stats(input_path: Path = ERA5_CLIPPED) -> pd.DataFrame:
    print("\nAggregating ERA5 daily statistics...")
    ds = xr.open_dataset(input_path, engine="netcdf4")
    frames: list[pd.DataFrame] = []
    try:
        if "time" not in ds.dims and "time" not in ds.coords:
            raise ValueError(f"ERA5 dataset has no time dimension: {input_path}")

        for var_name, data in ds.data_vars.items():
            if "time" not in data.dims or var_name == "spatial_ref":
                continue

            if var_name in ACCUMULATED_ERA5_VARIABLES:
                daily = data.resample(time="1D").sum(skipna=True)
                aggregation = "daily_total"
            else:
                daily = data.resample(time="1D").mean(skipna=True)
                aggregation = "daily_mean"

            frames.append(_spatial_stats(daily, f"era5_{var_name}_{aggregation}"))
    finally:
        ds.close()

    if not frames:
        raise ValueError(f"No time-varying ERA5 variables found in {input_path}")

    df = pd.concat(frames, axis=1).reset_index()
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").drop_duplicates("date")
    print(f"  ERA5 daily feature rows: {len(df)}")
    return df


def aggregate_glofas_discharge_stats(input_path: Path = GLOFAS_CLIPPED) -> pd.DataFrame:
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
        df = _spatial_stats(daily, "glofas_discharge").reset_index()
    finally:
        ds.close()

    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").drop_duplicates("date")
    print(f"  GloFAS daily feature rows: {len(df)}")
    return df


def add_lag_features(
    df: pd.DataFrame,
    lag_days: tuple[int, ...] = (1, 2, 3, 7),
    exclude_columns: tuple[str, ...] = ("date",),
) -> pd.DataFrame:
    print("\nCreating lag features...")
    result = df.sort_values("date").copy()
    base_columns = [
        column
        for column in result.select_dtypes(include="number").columns
        if column not in exclude_columns and not column.startswith("label_")
        and not column.startswith("target_")
    ]

    lagged_frames = []
    for lag in lag_days:
        lagged = result[base_columns].shift(lag)
        lagged.columns = [f"{column}_lag_{lag}d" for column in base_columns]
        lagged_frames.append(lagged)

    if lagged_frames:
        result = pd.concat([result, *lagged_frames], axis=1)

    print(f"  Added {len(base_columns) * len(lag_days)} lag columns.")
    return result


def add_rolling_rainfall_totals(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (3, 7, 14, 30),
    rainfall_column: str = "era5_tp_daily_total_mean",
) -> pd.DataFrame:
    print("\nCreating rolling rainfall totals...")
    if rainfall_column not in df.columns:
        raise ValueError(
            f"Rainfall column '{rainfall_column}' not found. Available columns: {list(df.columns)}"
        )

    result = df.sort_values("date").copy()
    for window in windows:
        result[f"rainfall_total_{window}d"] = (
            result[rainfall_column].rolling(window=window, min_periods=window).sum()
        )

    print(f"  Added {len(windows)} rolling rainfall columns.")
    return result


def add_discharge_threshold_labels(
    df: pd.DataFrame,
    discharge_column: str = "glofas_discharge_max",
    quantiles: tuple[float, ...] = (0.75, 0.9, 0.95),
    horizon_days: int = 1,
) -> pd.DataFrame:
    print("\nCreating discharge threshold labels...")
    if discharge_column not in df.columns:
        raise ValueError(
            f"Discharge column '{discharge_column}' not found. Available columns: {list(df.columns)}"
        )

    result = df.copy()
    target_column = f"target_{discharge_column}_next_{horizon_days}d"
    result[target_column] = result[discharge_column].shift(-horizon_days)

    thresholds = result[target_column].quantile(list(quantiles))
    for quantile, threshold in thresholds.items():
        label_name = (
            f"label_discharge_next_{horizon_days}d_ge_q{int(round(quantile * 100)):02d}"
        )
        result[label_name] = (result[target_column] >= threshold).astype("Int64")
        print(f"  {label_name}: threshold={threshold:.3f}")

    return result


def build_feature_dataframe(
    era5_path: Path = ERA5_CLIPPED,
    glofas_path: Path = GLOFAS_CLIPPED,
    lag_days: tuple[int, ...] = (1, 2, 3, 7),
    rainfall_windows: tuple[int, ...] = (3, 7, 14, 30),
    label_quantiles: tuple[float, ...] = (0.75, 0.9, 0.95),
    label_horizon_days: int = 1,
    drop_incomplete_rows: bool = True,
) -> pd.DataFrame:
    era5 = aggregate_era5_daily_stats(era5_path)
    glofas = aggregate_glofas_discharge_stats(glofas_path)

    print("\nJoining daily ERA5 and GloFAS features...")
    df = pd.merge(era5, glofas, on="date", how="inner").sort_values("date")
    print(f"  Joined rows: {len(df)}")

    df = add_rolling_rainfall_totals(df, windows=rainfall_windows)
    df = add_lag_features(df, lag_days=lag_days)
    df = add_discharge_threshold_labels(
        df,
        quantiles=label_quantiles,
        horizon_days=label_horizon_days,
    )

    if drop_incomplete_rows:
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        print(f"\nDropped incomplete rows from lags/rolling windows: {before - len(df)}")

    return df


def build_feature_csv(
    era5_path: Path = ERA5_CLIPPED,
    glofas_path: Path = GLOFAS_CLIPPED,
    output_path: Path = DEFAULT_OUTPUT,
) -> Path:
    print("\nBuilding ML-ready flood feature CSV...")
    df = build_feature_dataframe(era5_path=era5_path, glofas_path=glofas_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved ML-ready dataframe: {output_path.relative_to(PROJECT_ROOT)}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    return output_path


def main() -> None:
    build_feature_csv()


if __name__ == "__main__":
    main()
