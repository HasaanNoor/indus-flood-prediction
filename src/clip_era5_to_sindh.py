from pathlib import Path

from clip_utils import DEFAULT_BOUNDARY_PATH, clip_netcdf_to_boundary
from paths import CLIPPED_DIR, PROJECT_ROOT, PROCESSED_ERA5_DIR


def clip_era5_to_sindh(
    input_path: Path = PROCESSED_ERA5_DIR / "era5_sindh_2020_combined.nc",
    output_path: Path = CLIPPED_DIR / "era5_sindh_2020_clipped.nc",
    boundary_path: Path = DEFAULT_BOUNDARY_PATH,
) -> Path:
    print("\nClipping ERA5 to Sindh boundary...")
    path = clip_netcdf_to_boundary(input_path, output_path, boundary_path)
    print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")
    return path


if __name__ == "__main__":
    clip_era5_to_sindh()
