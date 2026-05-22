from pathlib import Path
import re
from zipfile import ZipFile, is_zipfile

import numpy as np
import xarray as xr

from paths import PROJECT_ROOT, RAW_ERA5_DIR, PROCESSED_ERA5_DIR


MULTIYEAR_OUTPUT = PROCESSED_ERA5_DIR / "era5_sindh_multiyear_combined.nc"
LEGACY_2020_OUTPUT = PROCESSED_ERA5_DIR / "era5_sindh_2020_combined.nc"
DAILY_SUM_KEYWORDS = (
    "precipitation",
    "runoff",
    "evaporation",
)
DAILY_SUM_NAMES = {"tp", "sro", "ssro", "ro", "e", "evaporation"}


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


def _infer_file_date(path: Path) -> tuple[int, int, str]:
    text = str(path)
    match = re.search(r"(20\d{2})[_/-](0[1-9]|1[0-2])", text)
    if match:
        return int(match.group(1)), int(match.group(2)), str(path)

    year_match = re.search(r"(20\d{2})", text)
    if year_match:
        return int(year_match.group(1)), 1, str(path)

    return 9999, 12, str(path)


def discover_era5_files(raw_dir: Path = RAW_ERA5_DIR) -> list[Path]:
    files = sorted(raw_dir.rglob("*.nc"), key=_infer_file_date)
    if not files:
        raise FileNotFoundError(f"No ERA5 .nc files found recursively under {raw_dir}")

    years = sorted({year for year, _, _ in (_infer_file_date(path) for path in files) if year != 9999})
    print(f"  Total ERA5 files discovered: {len(files)}")
    if years:
        print(f"  Detected ERA5 year range: {years[0]}-{years[-1]}")
    else:
        print("  Detected ERA5 year range: unknown")
    return files


def _open_many_era5_files(files: list[Path]) -> xr.Dataset:
    try:
        return xr.open_mfdataset(
            files,
            combine="by_coords",
            parallel=False,
            engine="netcdf4",
            join="outer",
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
                join="outer",
            )
        finally:
            for dataset in datasets:
                dataset.close()


def _sort_and_deduplicate_time(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords:
        return ds

    ds = ds.sortby("time")
    _, unique_indexes = np.unique(ds["time"].values, return_index=True)
    if len(unique_indexes) != ds.sizes.get("time", len(unique_indexes)):
        duplicate_count = ds.sizes["time"] - len(unique_indexes)
        print(f"  Removing {duplicate_count} duplicate ERA5 timestamps.")
        ds = ds.isel(time=np.sort(unique_indexes))
    return ds


def _uses_daily_sum(name: str, data: xr.DataArray) -> bool:
    if name in DAILY_SUM_NAMES:
        return True

    text = " ".join(
        str(data.attrs.get(key, "")).lower()
        for key in ["standard_name", "long_name", "description", "GRIB_name"]
    )
    return any(keyword in text for keyword in DAILY_SUM_KEYWORDS)


def _convert_depth_variables_to_mm(ds: xr.Dataset) -> xr.Dataset:
    for name in ["tp", "sro", "ssro", "ro", "e"]:
        if name not in ds.data_vars:
            continue
        units = str(ds[name].attrs.get("units", "")).lower()
        if units in {"m", "metre", "meter", "m of water equivalent"} or name in {"tp", "sro"}:
            print(f"  Converting ERA5 '{name}' from meters to millimeters.")
            ds[name] = ds[name] * 1000.0
            ds[name].attrs["units"] = "mm"
    return ds


def aggregate_era5_to_daily(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords:
        raise ValueError("ERA5 dataset has no time coordinate after standardization.")

    daily_vars = {}
    for name, data in ds.data_vars.items():
        if "time" not in data.dims:
            daily_vars[name] = data
            continue

        if _uses_daily_sum(name, data):
            daily_vars[name] = data.resample(time="1D").sum(skipna=True, min_count=1)
            daily_vars[name].attrs["daily_aggregation"] = "sum"
        else:
            daily_vars[name] = data.resample(time="1D").mean(skipna=True)
            daily_vars[name].attrs["daily_aggregation"] = "mean"

    daily = xr.Dataset(daily_vars, attrs=ds.attrs)
    for coord_name in ["lat", "lon", "latitude", "longitude"]:
        if coord_name in ds.coords and coord_name not in daily.coords:
            daily = daily.assign_coords({coord_name: ds[coord_name]})
    daily.attrs["temporal_resolution"] = "daily"
    return daily


def process_era5(
    raw_dir: Path = RAW_ERA5_DIR,
    output_path: Path = MULTIYEAR_OUTPUT,
    legacy_output_path: Path | None = LEGACY_2020_OUTPUT,
) -> Path:
    print("\nProcessing ERA5 data...")
    raw_files = discover_era5_files(raw_dir)
    extracted_dir = output_path.parent / "_extracted_era5_netcdf"
    netcdf_files = sorted(_extract_zipped_netcdf_files(raw_files, extracted_dir), key=_infer_file_date)
    if not netcdf_files:
        raise FileNotFoundError("No readable ERA5 NetCDF files found after checking raw files.")

    print(f"  Combining {len(netcdf_files)} ERA5 NetCDF streams.")
    try:
        ds = _open_many_era5_files(netcdf_files)
    except Exception as exc:
        raise RuntimeError(f"Could not combine ERA5 files: {exc}") from exc

    ds = _standardize_coordinate_names(ds)
    ds = _sort_and_deduplicate_time(ds)
    ds = _convert_depth_variables_to_mm(ds)
    print("  Aggregating ERA5 variables to daily resolution.")
    daily = aggregate_era5_to_daily(ds)
    ds.close()
    daily = _sort_and_deduplicate_time(daily)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_netcdf(output_path)
    print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")

    if legacy_output_path and legacy_output_path != output_path:
        daily.to_netcdf(legacy_output_path)
        print(f"  Backward-compatible copy saved: {legacy_output_path.relative_to(PROJECT_ROOT)}")

    daily.close()
    return output_path
