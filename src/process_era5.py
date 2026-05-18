from pathlib import Path
from zipfile import ZipFile, is_zipfile

import xarray as xr

from paths import PROJECT_ROOT, RAW_ERA5_DIR, PROCESSED_ERA5_DIR


def _standardize_coordinate_names(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {}
    if "valid_time" in ds.coords or "valid_time" in ds.dims:
        rename_map["valid_time"] = "time"
    if "latitude" in ds.coords or "latitude" in ds.dims:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.coords or "longitude" in ds.dims:
        rename_map["longitude"] = "lon"
    return ds.rename(rename_map) if rename_map else ds


def _extract_zipped_netcdf_files(files: list[Path], extract_dir: Path) -> list[Path]:
    extracted_files = []
    extract_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        if not is_zipfile(path):
            extracted_files.append(path)
            continue

        month_dir = extract_dir / path.stem
        month_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.endswith(".nc")]
            if not members:
                print(f"  Warning: no inner NetCDF files found in {path.name}")
                continue

            for member in members:
                target_path = month_dir / Path(member).name
                if not target_path.exists():
                    with archive.open(member) as source, target_path.open("wb") as target:
                        target.write(source.read())
                extracted_files.append(target_path)

    return sorted(extracted_files)


def _open_many_era5_files(files: list[Path]) -> xr.Dataset:
    try:
        return xr.open_mfdataset(
            files,
            combine="by_coords",
            parallel=False,
            engine="netcdf4",
        )
    except ImportError as exc:
        if "dask" not in str(exc).lower():
            raise

        print("  dask is not installed; falling back to eager xarray combination.")
        datasets = []
        try:
            for path in files:
                datasets.append(xr.open_dataset(path, engine="netcdf4").load())
            return xr.combine_by_coords(
                datasets,
                combine_attrs="override",
                compat="no_conflicts",
            )
        finally:
            for dataset in datasets:
                dataset.close()


def process_era5(
    raw_dir: Path = RAW_ERA5_DIR,
    output_path: Path = PROCESSED_ERA5_DIR / "era5_sindh_2020_combined.nc",
) -> Path:
    print("\nProcessing ERA5 data...")
    raw_files = sorted(raw_dir.glob("*.nc"))
    if not raw_files:
        raise FileNotFoundError(f"No ERA5 .nc files found in {raw_dir}")

    print(f"  Found {len(raw_files)} monthly ERA5 files.")
    extracted_dir = output_path.parent / "_extracted_era5_netcdf"
    netcdf_files = _extract_zipped_netcdf_files(raw_files, extracted_dir)
    if not netcdf_files:
        raise FileNotFoundError("No readable ERA5 NetCDF files found after checking raw files.")

    print(f"  Combining {len(netcdf_files)} NetCDF files with xarray.open_mfdataset.")
    try:
        ds = _open_many_era5_files(netcdf_files)
    except Exception as exc:
        raise RuntimeError(f"Could not combine ERA5 files: {exc}") from exc

    ds = _standardize_coordinate_names(ds)
    if "time" in ds.coords:
        ds = ds.sortby("time")

    if "tp" in ds.data_vars:
        print("  Converting total precipitation 'tp' from meters to millimeters.")
        ds["tp"] = ds["tp"] * 1000.0
        ds["tp"].attrs["units"] = "mm"
        ds["tp"].attrs["description"] = "Total precipitation converted from meters to millimeters"
    else:
        print("  Variable 'tp' not found; skipping precipitation unit conversion.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    ds.close()
    print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")
    return output_path
