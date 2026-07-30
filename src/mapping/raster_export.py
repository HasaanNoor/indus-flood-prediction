from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin


PROBABILITY_NODATA = -9999.0
RISK_NODATA = 0


@dataclass(frozen=True)
class SpatialGrid:
    latitude_column: str
    longitude_column: str
    crs: str
    transform: object
    width: int
    height: int
    resolution: tuple[float, float]
    nodata: float | int
    latitude_values: tuple[float, ...] = ()
    longitude_values: tuple[float, ...] = ()


def detect_coordinate_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    lat = "latitude" if "latitude" in df.columns else "lat" if "lat" in df.columns else None
    lon = "longitude" if "longitude" in df.columns else "lon" if "lon" in df.columns else None
    if lat and lon:
        return lat, lon
    return None


def reconstruct_regular_grid(
    df: pd.DataFrame,
    crs: str | None = None,
    nodata: float | int = PROBABILITY_NODATA,
) -> SpatialGrid | None:
    columns = detect_coordinate_columns(df)
    if columns is None:
        return None
    if crs is None:
        raise ValueError("CRS is required for raster export and will not be inferred silently.")

    lat_col, lon_col = columns
    coords = df[[lat_col, lon_col]]
    if coords.isna().any().any():
        raise ValueError("Coordinate columns contain missing values.")
    if coords.duplicated().any():
        raise ValueError("Coordinate pairs are duplicated; cannot reconstruct a unique raster grid.")

    lats = np.sort(coords[lat_col].unique())[::-1]
    lons = np.sort(coords[lon_col].unique())
    if len(lats) < 2 or len(lons) < 2:
        raise ValueError("At least two unique latitudes and longitudes are required for raster export.")
    expected = len(lats) * len(lons)
    if len(df) != expected:
        raise ValueError(
            f"Point rows do not form a complete rectangular grid: rows={len(df)}, expected={expected}."
        )

    yres = float(abs(np.diff(lats).mean()))
    xres = float(abs(np.diff(lons).mean()))
    if not np.allclose(np.abs(np.diff(lats)), yres) or not np.allclose(np.abs(np.diff(lons)), xres):
        raise ValueError("Coordinates are not regularly spaced; cannot preserve raster alignment.")

    transform = from_origin(float(lons.min() - xres / 2), float(lats.max() + yres / 2), xres, yres)
    return SpatialGrid(
        latitude_column=lat_col,
        longitude_column=lon_col,
        crs=crs,
        transform=transform,
        width=len(lons),
        height=len(lats),
        resolution=(xres, yres),
        nodata=nodata,
        latitude_values=tuple(float(value) for value in lats),
        longitude_values=tuple(float(value) for value in lons),
    )


def values_to_grid(df: pd.DataFrame, values: np.ndarray, grid: SpatialGrid) -> np.ndarray:
    if len(df) != len(values):
        raise ValueError("Output values length does not match input rows.")
    result = np.full((grid.height, grid.width), grid.nodata, dtype=np.asarray(values).dtype)
    if grid.latitude_column not in df.columns or grid.longitude_column not in df.columns:
        raise ValueError("Input rows do not contain the coordinate columns used by the raster grid.")
    lats = grid.latitude_values or tuple(np.sort(df[grid.latitude_column].unique())[::-1])
    lons = grid.longitude_values or tuple(np.sort(df[grid.longitude_column].unique()))
    lat_index = {value: idx for idx, value in enumerate(lats)}
    lon_index = {value: idx for idx, value in enumerate(lons)}
    for (_, row), value in zip(df.iterrows(), values):
        lat = row[grid.latitude_column]
        lon = row[grid.longitude_column]
        if lat not in lat_index or lon not in lon_index:
            raise ValueError("Input row coordinates are outside the raster grid.")
        result[lat_index[lat], lon_index[lon]] = value
    return result


def validate_raster_dimensions(array: np.ndarray, grid: SpatialGrid) -> None:
    if array.shape != (grid.height, grid.width):
        raise ValueError(
            f"Raster dimensions {array.shape} do not match source grid {(grid.height, grid.width)}."
        )


def write_geotiff(array: np.ndarray, grid: SpatialGrid, output_path: Path, dtype: str, nodata: float | int) -> Path:
    validate_raster_dimensions(array, grid)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=grid.height,
        width=grid.width,
        count=1,
        dtype=dtype,
        crs=grid.crs,
        transform=grid.transform,
        nodata=nodata,
    ) as dst:
        dst.write(array.astype(dtype), 1)
    return output_path


def spatial_metadata_for_grid(grid: SpatialGrid | None) -> dict[str, object]:
    if grid is None:
        return {
            "spatial_output_status": "skipped",
            "spatial_output_reason": (
                "Input prediction dataset has no complete latitude/longitude raster grid metadata. "
                "Existing processed CSVs are temporal aggregate feature tables."
            ),
            "crs": None,
            "affine_transform": None,
            "raster_dimensions": None,
            "resolution": None,
            "nodata_value": None,
        }
    return {
        "spatial_output_status": "available",
        "crs": grid.crs,
        "affine_transform": tuple(grid.transform)[:6],
        "raster_dimensions": {"width": grid.width, "height": grid.height},
        "resolution": list(grid.resolution),
        "nodata_value": grid.nodata,
    }
