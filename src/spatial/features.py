from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.spatial.alignment import (
    align_raster_to_grid,
    build_glofas_river_mapping,
    flatten_inside,
    slope_from_elevation,
    write_dataframe_partition,
)
from src.spatial.configuration import MAX_LOOKBACK_DAYS, SpatialPipelineConfig
from src.spatial.grid import CanonicalGrid, grid_cells_dataframe
from src.spatial.temporal import date_range, normalise_daily_index, validate_available_dates, validate_no_future_looking_join


ERA5_SUM_VARIABLES = {"tp", "sro", "e", "pev"}


def _select_dates(ds: xr.Dataset, requested: pd.DatetimeIndex, source_name: str) -> xr.Dataset:
    available = normalise_daily_index(ds["time"].values, source_name)
    validate_available_dates(available, requested, source_name)
    return ds.sel(time=requested.values)


def _time_indexer(ds: xr.Dataset) -> dict[pd.Timestamp, int]:
    dates = normalise_daily_index(ds["time"].values, "dataset")
    return {pd.Timestamp(date): idx for idx, date in enumerate(dates)}


def _era5_array(ds: xr.Dataset, var_name: str, date: pd.Timestamp, indexer: dict[pd.Timestamp, int], grid: CanonicalGrid) -> np.ndarray:
    data = ds[var_name].isel(time=indexer[date])
    if tuple(data.dims[-2:]) != ("lat", "lon"):
        data = data.transpose("lat", "lon")
    selected = data.sel(lat=grid.latitudes, lon=grid.longitudes)
    arr = selected.values
    selected_lats = np.asarray(selected["lat"].values, dtype="float64")
    if not np.allclose(selected_lats, grid.latitudes):
        arr = arr[::-1, :]
    return arr.astype("float32")


def static_feature_frame(config: SpatialPipelineConfig, grid: CanonicalGrid) -> pd.DataFrame:
    cells = grid_cells_dataframe(grid)
    inside = cells.loc[cells["in_sindh"]].reset_index(drop=True)
    elevation = align_raster_to_grid(str(config.srtm_path), grid, "terrain_elevation")
    slope = slope_from_elevation(elevation, grid.resolution[0], grid.resolution[1])
    inside["terrain_elevation_m"] = flatten_inside(grid, elevation)
    inside["terrain_slope_degrees"] = flatten_inside(grid, slope)
    inside["relative_elevation_within_sindh_m"] = inside["terrain_elevation_m"] - inside["terrain_elevation_m"].median()
    return inside


def build_spatial_features(
    config: SpatialPipelineConfig,
    grid: CanonicalGrid,
    start_date: str,
    end_date: str,
    output_path: Path | None = None,
) -> pd.DataFrame:
    requested = date_range(start_date, end_date)
    lookback_start = requested.min() - pd.Timedelta(days=MAX_LOOKBACK_DAYS)
    needed = pd.date_range(lookback_start, requested.max(), freq="D")
    static = static_feature_frame(config, grid)

    era5 = xr.open_dataset(config.era5_path)
    glofas = xr.open_dataset(config.glofas_path)
    try:
        _select_dates(era5, requested, "ERA5")
        _select_dates(glofas, requested, "GloFAS")
        era5_index = _time_indexer(era5)
        glofas_index = _time_indexer(glofas)
        era5_available = set(era5_index)
        glofas_available = set(glofas_index)
        era5_vars = [name for name in era5.data_vars if "time" in era5[name].dims and {"lat", "lon"}.issubset(era5[name].dims)]
        discharge_var = "dis24" if "dis24" in glofas.data_vars else list(glofas.data_vars)[0]
        representative = glofas[discharge_var].isel(time=glofas_index[requested.min()]).values
        river_mapping = build_glofas_river_mapping(
            np.asarray(glofas["lat"].values, dtype="float64"),
            np.asarray(glofas["lon"].values, dtype="float64"),
            representative,
            grid,
        )

        frames: list[pd.DataFrame] = []
        for current_date in requested:
            validate_no_future_looking_join(current_date, [current_date])
            frame = static.copy()
            frame.insert(1, "date", current_date.date().isoformat())
            frame["source_era5_date"] = current_date.date().isoformat()
            frame["source_glofas_date"] = current_date.date().isoformat()
            frame["month"] = current_date.month
            frame["day_of_year"] = current_date.dayofyear
            frame["season_sin"] = np.sin(2 * np.pi * current_date.dayofyear / 366.0)
            frame["season_cos"] = np.cos(2 * np.pi * current_date.dayofyear / 366.0)
            frame["is_monsoon"] = int(6 <= current_date.month <= 9)

            for var_name in era5_vars:
                arr = _era5_array(era5, var_name, current_date, era5_index, grid)
                prefix = f"era5_{var_name}"
                frame[f"{prefix}_current"] = flatten_inside(grid, arr)
                for lag in (1, 3, 7):
                    lag_date = current_date - pd.Timedelta(days=lag)
                    if lag_date in era5_available:
                        frame[f"{prefix}_lag_{lag}d"] = flatten_inside(grid, _era5_array(era5, var_name, lag_date, era5_index, grid))
                    else:
                        frame[f"{prefix}_lag_{lag}d"] = np.nan
                for window in (3, 7):
                    window_dates = [current_date - pd.Timedelta(days=offset) for offset in range(window)]
                    if all(value in era5_available for value in window_dates):
                        arrays = [_era5_array(era5, var_name, value, era5_index, grid) for value in window_dates]
                        reducer = np.nansum if var_name in ERA5_SUM_VARIABLES else np.nanmean
                        frame[f"{prefix}_{'total' if var_name in ERA5_SUM_VARIABLES else 'mean'}_{window}d"] = flatten_inside(grid, reducer(arrays, axis=0))
                    else:
                        frame[f"{prefix}_{'total' if var_name in ERA5_SUM_VARIABLES else 'mean'}_{window}d"] = np.nan

            if {"u10", "v10"}.issubset(era5_vars):
                u = _era5_array(era5, "u10", current_date, era5_index, grid)
                v = _era5_array(era5, "v10", current_date, era5_index, grid)
                frame["era5_wind_speed_current"] = flatten_inside(grid, np.sqrt(u**2 + v**2))

            dis = glofas[discharge_var].isel(time=glofas_index[current_date]).values.astype("float32")
            nearest = dis.ravel()[river_mapping.nearest_flat_indexes].reshape(grid.height, grid.width)
            on_river = np.where(river_mapping.has_river_within_cell, nearest, np.nan)
            frame["glofas_nearest_river_discharge_m3s_current"] = flatten_inside(grid, nearest)
            frame["glofas_river_discharge_m3s_on_river_cell"] = flatten_inside(grid, on_river)
            frame["distance_to_glofas_river_km"] = flatten_inside(grid, river_mapping.distance_km)
            frame["has_glofas_river_cell"] = flatten_inside(grid, river_mapping.has_river_within_cell).astype("int8")
            for lag in (1, 3, 7):
                lag_date = current_date - pd.Timedelta(days=lag)
                if lag_date in glofas_available:
                    lag_dis = glofas[discharge_var].isel(time=glofas_index[lag_date]).values.astype("float32")
                    lag_nearest = lag_dis.ravel()[river_mapping.nearest_flat_indexes].reshape(grid.height, grid.width)
                    frame[f"glofas_nearest_river_discharge_m3s_lag_{lag}d"] = flatten_inside(grid, lag_nearest)
                else:
                    frame[f"glofas_nearest_river_discharge_m3s_lag_{lag}d"] = np.nan
            for window in (3, 7):
                window_dates = [current_date - pd.Timedelta(days=offset) for offset in range(window)]
                if not all(value in glofas_available for value in window_dates):
                    frame[f"glofas_nearest_river_discharge_m3s_mean_{window}d"] = np.nan
                    continue
                vals = []
                for value in window_dates:
                    d = glofas[discharge_var].isel(time=glofas_index[value]).values.astype("float32")
                    vals.append(d.ravel()[river_mapping.nearest_flat_indexes].reshape(grid.height, grid.width))
                frame[f"glofas_nearest_river_discharge_m3s_mean_{window}d"] = flatten_inside(grid, np.nanmean(vals, axis=0))

            frames.append(frame)
    finally:
        era5.close()
        glofas.close()

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["date", "row", "col"]).reset_index(drop=True)
    if output_path is not None:
        write_dataframe_partition(result, output_path)
    return result
