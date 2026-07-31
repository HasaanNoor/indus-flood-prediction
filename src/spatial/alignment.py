from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.spatial import cKDTree

from src.spatial.grid import CanonicalGrid


RESAMPLING_RULES = {
    "meteorological_continuous": "native-grid lookup on canonical ERA5 grid",
    "terrain_elevation": "bilinear",
    "terrain_slope": "bilinear after slope derivation",
    "categorical_mask": "nearest",
    "flood_mask": "nearest",
    "glofas_discharge": "nearest valid river-cell feature plus explicit river mask and distance",
}


def resampling_for(kind: str) -> Resampling | str:
    if kind in {"categorical_mask", "flood_mask"}:
        return Resampling.nearest
    if kind in {"terrain_elevation", "terrain_slope", "meteorological_continuous"}:
        return Resampling.bilinear
    if kind == "glofas_discharge":
        return "river-aware-nearest"
    raise ValueError(f"Unsupported resampling kind: {kind}")


def align_raster_to_grid(path: str, grid: CanonicalGrid, kind: str, dst_dtype: str = "float32") -> np.ndarray:
    method = resampling_for(kind)
    if isinstance(method, str):
        raise ValueError(f"Raster reprojection is not valid for {kind}; use source-specific alignment.")
    destination = np.full((grid.height, grid.width), np.nan, dtype=dst_dtype)
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Raster CRS is missing: {path}")
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            resampling=method,
        )
    return destination


def slope_from_elevation(elevation: np.ndarray, xres_degrees: float, yres_degrees: float) -> np.ndarray:
    meters_per_degree = 111_320.0
    y_grad, x_grad = np.gradient(elevation.astype("float64"), yres_degrees * meters_per_degree, xres_degrees * meters_per_degree)
    return np.degrees(np.arctan(np.sqrt(x_grad**2 + y_grad**2))).astype("float32")


@dataclass(frozen=True)
class RiverMapping:
    nearest_flat_indexes: np.ndarray
    distance_km: np.ndarray
    has_river_within_cell: np.ndarray
    valid_river_mask: np.ndarray


def build_glofas_river_mapping(glofas_lats: np.ndarray, glofas_lons: np.ndarray, representative_discharge: np.ndarray, grid: CanonicalGrid) -> RiverMapping:
    valid = np.isfinite(representative_discharge) & (representative_discharge > 0)
    if not valid.any():
        raise ValueError("No valid GloFAS river cells found from representative discharge.")
    lon2d, lat2d = np.meshgrid(glofas_lons, glofas_lats)
    river_points = np.column_stack([lat2d[valid], lon2d[valid]])
    tree = cKDTree(river_points)
    grid_lon2d, grid_lat2d = np.meshgrid(grid.longitudes, grid.latitudes)
    query_points = np.column_stack([grid_lat2d.ravel(), grid_lon2d.ravel()])
    dist_deg, river_index = tree.query(query_points, k=1)
    nearest_lat = river_points[river_index, 0]
    distance_km = dist_deg * 111.32
    half_y = grid.resolution[1] / 2
    half_x = grid.resolution[0] / 2
    has_within = (
        (np.abs(nearest_lat - query_points[:, 0]) <= half_y)
        & (np.abs(river_points[river_index, 1] - query_points[:, 1]) <= half_x)
    )
    flat_valid_indexes = np.flatnonzero(valid.ravel())
    nearest_flat = flat_valid_indexes[river_index]
    return RiverMapping(
        nearest_flat_indexes=nearest_flat.astype("int64"),
        distance_km=distance_km.reshape(grid.height, grid.width).astype("float32"),
        has_river_within_cell=has_within.reshape(grid.height, grid.width),
        valid_river_mask=valid,
    )


def flatten_inside(grid: CanonicalGrid, values: np.ndarray) -> np.ndarray:
    if values.shape != (grid.height, grid.width):
        raise ValueError(f"Aligned array shape {values.shape} does not match canonical grid {(grid.height, grid.width)}.")
    return values.ravel()[grid.boundary_mask.ravel()]


def write_dataframe_partition(df: pd.DataFrame, output_path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(output_path)

