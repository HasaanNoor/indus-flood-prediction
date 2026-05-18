from pathlib import Path
from zipfile import BadZipFile, ZipFile, is_zipfile

import geopandas as gpd
import rasterio
import xarray as xr

from paths import (
    PROJECT_ROOT,
    RAW_BOUNDARIES_DIR,
    RAW_ERA5_DIR,
    RAW_GLOFAS_DIR,
    RAW_SRTM_DIR,
)


def _print_xarray_summary(ds: xr.Dataset) -> None:
    print("    Dimensions:")
    for name, size in ds.sizes.items():
        print(f"      {name}: {size}")

    print("    Coordinates:")
    for name, coord in ds.coords.items():
        print(f"      {name}: shape={coord.shape}, dtype={coord.dtype}")

    print("    Variables:")
    for name, variable in ds.data_vars.items():
        units = variable.attrs.get("units", "no units attribute")
        long_name = variable.attrs.get("long_name", "")
        label = f"{name} ({long_name})" if long_name else name
        print(f"      {label}: dims={variable.dims}, units={units}")


def inspect_era5_files(raw_dir: Path = RAW_ERA5_DIR) -> None:
    print("\nInspecting ERA5 files...")
    files = sorted(raw_dir.glob("*.nc"))
    if not files:
        print(f"  No ERA5 .nc files found in {raw_dir.relative_to(PROJECT_ROOT)}")
        return

    for path in files:
        print(f"\n  File: {path.relative_to(PROJECT_ROOT)}")
        try:
            if is_zipfile(path):
                with ZipFile(path) as archive:
                    members = [name for name in archive.namelist() if name.endswith(".nc")]
                    print("    ZIP archive containing NetCDF members:")
                    for member in members:
                        print(f"      {member}")
                    if not members:
                        print("    No inner .nc files found.")
                continue

            with xr.open_dataset(path) as ds:
                _print_xarray_summary(ds)
        except (OSError, ValueError, BadZipFile) as exc:
            print(f"    Could not inspect ERA5 file: {exc}")


def inspect_glofas_file(raw_dir: Path = RAW_GLOFAS_DIR) -> None:
    print("\nInspecting GloFAS file...")
    files = sorted(raw_dir.glob("*.nc"))
    if not files:
        print(f"  No GloFAS .nc files found in {raw_dir.relative_to(PROJECT_ROOT)}")
        return

    for path in files:
        print(f"\n  File: {path.relative_to(PROJECT_ROOT)}")
        try:
            with xr.open_dataset(path) as ds:
                _print_xarray_summary(ds)
        except (OSError, ValueError) as exc:
            print(f"    Could not inspect GloFAS file: {exc}")


def inspect_srtm_files(raw_dir: Path = RAW_SRTM_DIR) -> None:
    print("\nInspecting SRTM GeoTIFF files...")
    files = sorted(raw_dir.glob("*.tif"))
    if not files:
        print(f"  No SRTM .tif files found in {raw_dir.relative_to(PROJECT_ROOT)}")
        return

    for path in files:
        print(f"\n  File: {path.relative_to(PROJECT_ROOT)}")
        try:
            with rasterio.open(path) as src:
                print(f"    CRS: {src.crs}")
                print(f"    Bounds: {src.bounds}")
                print(f"    Shape: {src.shape}")
                print(f"    Resolution: {src.res}")
        except rasterio.errors.RasterioIOError as exc:
            print(f"    Could not inspect SRTM file: {exc}")


def inspect_boundary_file(raw_dir: Path = RAW_BOUNDARIES_DIR) -> None:
    print("\nInspecting boundary shapefile...")
    candidates = sorted(raw_dir.glob("*ADM1*.shp")) + sorted(raw_dir.glob("*.shp"))
    if not candidates:
        print(f"  No boundary .shp files found in {raw_dir.relative_to(PROJECT_ROOT)}")
        return

    path = candidates[0]
    print(f"  File: {path.relative_to(PROJECT_ROOT)}")
    try:
        gdf = gpd.read_file(path)
        print(f"    CRS: {gdf.crs}")
        print(f"    Columns: {list(gdf.columns)}")
        print(f"    Geometry bounds: {tuple(gdf.total_bounds)}")
    except Exception as exc:
        print(f"    Could not inspect boundary file: {exc}")


def inspect_all_raw_data() -> None:
    inspect_era5_files()
    inspect_glofas_file()
    inspect_srtm_files()
    inspect_boundary_file()
