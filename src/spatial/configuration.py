from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.paths import (
    CLIPPED_DIR,
    DATA_PROCESSED,
    OUTPUTS,
    PROCESSED_BOUNDARIES_DIR,
    PROCESSED_ERA5_DIR,
    PROCESSED_GLOFAS_DIR,
)


SPATIAL_DIR = DATA_PROCESSED / "spatial"
SPATIAL_FEATURES_DIR = SPATIAL_DIR / "features"
SPATIAL_LABELS_DIR = SPATIAL_DIR / "labels"
SPATIAL_VALIDATION_DIR = SPATIAL_DIR / "validation"

GRID_METADATA_PATH = SPATIAL_DIR / "grid_metadata.json"
GRID_CELLS_PATH = SPATIAL_DIR / "grid_cells.csv"

DEFAULT_ERA5_PATH = PROCESSED_ERA5_DIR / "era5_sindh_multiyear_combined.nc"
DEFAULT_GLOFAS_PATH = PROCESSED_GLOFAS_DIR / "glofas_sindh_multiyear_clean.nc"
DEFAULT_SRTM_PATH = CLIPPED_DIR / "srtm_sindh_clipped.tif"
DEFAULT_BOUNDARY_PATH = PROCESSED_BOUNDARIES_DIR / "sindh_boundary.geojson"
DEFAULT_SENTINEL1_DIR = OUTPUTS / "validation" / "sentinel1"

DEFAULT_START_DATE = "2010-01-02"
DEFAULT_END_DATE = "2023-12-29"
DEFAULT_PILOT_START_DATE = "2019-08-01"
DEFAULT_PILOT_END_DATE = "2019-08-15"
MAX_LOOKBACK_DAYS = 7
NODATA_FLOAT = -9999.0


@dataclass(frozen=True)
class SpatialPipelineConfig:
    era5_path: Path = DEFAULT_ERA5_PATH
    glofas_path: Path = DEFAULT_GLOFAS_PATH
    srtm_path: Path = DEFAULT_SRTM_PATH
    boundary_path: Path = DEFAULT_BOUNDARY_PATH
    sentinel1_dir: Path = DEFAULT_SENTINEL1_DIR
    output_dir: Path = SPATIAL_DIR
    grid_metadata_path: Path = GRID_METADATA_PATH
    grid_cells_path: Path = GRID_CELLS_PATH
    nodata_float: float = NODATA_FLOAT
    max_glofas_river_distance_km: float = 75.0
