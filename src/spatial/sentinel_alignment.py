from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from src.spatial.grid import CanonicalGrid


NODATA_CODE = 255


@dataclass(frozen=True)
class RasterProfile:
    path: str
    crs: str
    width: int
    height: int
    count: int
    dtype: str
    nodata: float | int | None
    bounds: tuple[float, float, float, float]
    transform: tuple[float, float, float, float, float, float]


def inspect_raster(path: Path) -> RasterProfile:
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Raster CRS is missing: {path}")
        if src.count < 1:
            raise ValueError(f"Raster has no bands: {path}")
        if src.transform.b != 0 or src.transform.d != 0:
            raise ValueError(f"Raster transform has rotation/shear and cannot be safely aligned: {path}")
        if src.transform.a <= 0 or src.transform.e >= 0:
            raise ValueError(f"Raster transform orientation is not north-up: {path}")
        return RasterProfile(
            path=str(path),
            crs=src.crs.to_string(),
            width=src.width,
            height=src.height,
            count=src.count,
            dtype=src.dtypes[0],
            nodata=src.nodata,
            bounds=(float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top)),
            transform=tuple(float(v) for v in tuple(src.transform)[:6]),
        )


def _aligned_uint8_band(path: Path, grid: CanonicalGrid, band: int) -> np.ndarray:
    destination = np.full((grid.height, grid.width), NODATA_CODE, dtype="uint8")
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Raster CRS is missing: {path}")
        src_nodata = src.nodata
        reproject(
            source=rasterio.band(src, band),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=NODATA_CODE,
            resampling=Resampling.nearest,
        )
    return destination


def align_label_raster(path: Path, grid: CanonicalGrid, band: int = 1) -> np.ndarray:
    profile = inspect_raster(path)
    if profile.crs != grid.crs:
        raise ValueError(f"Raster CRS {profile.crs} differs from canonical grid CRS {grid.crs}: {path}")
    aligned = _aligned_uint8_band(path, grid, band)
    values = set(np.unique(aligned).tolist())
    allowed = {0, 1, NODATA_CODE}
    if not values.issubset(allowed):
        raise ValueError(f"Aligned label raster contains unsupported values {sorted(values - allowed)} in {path}")
    return aligned
