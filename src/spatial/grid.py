from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.features import geometry_mask
from rasterio.transform import Affine, from_origin


@dataclass(frozen=True)
class CanonicalGrid:
    crs: str
    transform: Affine
    width: int
    height: int
    resolution: tuple[float, float]
    bounds: tuple[float, float, float, float]
    latitudes: np.ndarray
    longitudes: np.ndarray
    boundary_mask: np.ndarray
    grid_cell_ids: np.ndarray

    @property
    def cell_count(self) -> int:
        return int(self.width * self.height)

    @property
    def inside_cell_count(self) -> int:
        return int(self.boundary_mask.sum())


def _regular_resolution(values: np.ndarray, name: str) -> float:
    unique = np.asarray(values, dtype="float64")
    if unique.size < 2:
        raise ValueError(f"{name} requires at least two coordinates.")
    diffs = np.abs(np.diff(np.sort(unique)))
    resolution = float(np.median(diffs))
    if not np.allclose(diffs, resolution, atol=1e-9):
        raise ValueError(f"{name} coordinates are not regularly spaced.")
    return resolution


def grid_from_era5(era5_path: Path, boundary_path: Path) -> CanonicalGrid:
    if not era5_path.exists():
        raise FileNotFoundError(f"ERA5 processed dataset not found: {era5_path}")
    if not boundary_path.exists():
        raise FileNotFoundError(f"Sindh boundary not found: {boundary_path}")

    ds = xr.open_dataset(era5_path)
    try:
        if "lat" not in ds.coords or "lon" not in ds.coords:
            raise ValueError("ERA5 dataset must expose lat/lon coordinates.")
        lats = np.sort(ds["lat"].values.astype("float64"))[::-1]
        lons = np.sort(ds["lon"].values.astype("float64"))
    finally:
        ds.close()

    xres = _regular_resolution(lons, "ERA5 longitude")
    yres = _regular_resolution(lats, "ERA5 latitude")
    transform = from_origin(float(lons.min() - xres / 2), float(lats.max() + yres / 2), xres, yres)
    bounds = (
        float(lons.min() - xres / 2),
        float(lats.min() - yres / 2),
        float(lons.max() + xres / 2),
        float(lats.max() + yres / 2),
    )

    boundary = gpd.read_file(boundary_path)
    if boundary.crs is None:
        raise ValueError("Sindh boundary CRS is missing.")
    boundary = boundary.to_crs("EPSG:4326")
    mask = geometry_mask(
        [geom for geom in boundary.geometry if geom is not None],
        out_shape=(len(lats), len(lons)),
        transform=transform,
        invert=True,
        all_touched=True,
    )
    rows, cols = np.indices(mask.shape)
    ids = np.array([f"sindh_era5_r{r:03d}_c{c:03d}" for r, c in zip(rows.ravel(), cols.ravel())], dtype=object)
    ids = ids.reshape(mask.shape)
    return CanonicalGrid("EPSG:4326", transform, len(lons), len(lats), (xres, yres), bounds, lats, lons, mask, ids)


def grid_cells_dataframe(grid: CanonicalGrid) -> pd.DataFrame:
    rows, cols = np.indices((grid.height, grid.width))
    lon2d, lat2d = np.meshgrid(grid.longitudes, grid.latitudes)
    return pd.DataFrame(
        {
            "grid_cell_id": grid.grid_cell_ids.ravel(),
            "row": rows.ravel().astype("int32"),
            "col": cols.ravel().astype("int32"),
            "latitude": lat2d.ravel(),
            "longitude": lon2d.ravel(),
            "in_sindh": grid.boundary_mask.ravel().astype(bool),
        }
    )


def metadata_dict(grid: CanonicalGrid) -> dict[str, object]:
    return {
        "unit_of_analysis": "one row is one canonical grid cell on one daily date",
        "canonical_grid_source": "ERA5 processed daily regular latitude/longitude grid",
        "crs": grid.crs,
        "affine_transform": list(tuple(grid.transform)[:6]),
        "width": grid.width,
        "height": grid.height,
        "resolution_degrees": list(grid.resolution),
        "bounds": {
            "left": grid.bounds[0],
            "bottom": grid.bounds[1],
            "right": grid.bounds[2],
            "top": grid.bounds[3],
        },
        "grid_origin": "upper-left outer corner derived from ERA5 cell centers",
        "cell_dimensions": {"x_degrees": grid.resolution[0], "y_degrees": grid.resolution[1]},
        "temporal_frequency": "daily",
        "nodata_policy": "NaN in tabular features; -9999 only for raster-compatible numeric reports",
        "cell_ordering": "row-major, north-to-south then west-to-east",
        "total_cells": grid.cell_count,
        "sindh_cells": grid.inside_cell_count,
        "grid_cell_id_pattern": "sindh_era5_r{row:03d}_c{col:03d}",
    }


def save_grid_outputs(grid: CanonicalGrid, metadata_path: Path, cells_path: Path) -> tuple[Path, Path]:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata_dict(grid), indent=2) + "\n")
    grid_cells_dataframe(grid).to_csv(cells_path, index=False)
    return metadata_path, cells_path

