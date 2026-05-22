from pathlib import Path
import re

import numpy as np
import xarray as xr

from paths import PROJECT_ROOT, RAW_GLOFAS_DIR, PROCESSED_GLOFAS_DIR


MULTIYEAR_OUTPUT = PROCESSED_GLOFAS_DIR / "glofas_sindh_multiyear_clean.nc"
LEGACY_2020_OUTPUT = PROCESSED_GLOFAS_DIR / "glofas_sindh_2020_clean.nc"


def _standardize_coordinate_names(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {}
    if "valid_time" in ds.coords or "valid_time" in ds.dims:
        rename_map["valid_time"] = "time"
    if "latitude" in ds.coords or "latitude" in ds.dims:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.coords or "longitude" in ds.dims:
        rename_map["longitude"] = "lon"
    return ds.rename(rename_map) if rename_map else ds


def _identify_discharge_variable(ds: xr.Dataset) -> str | None:
    preferred_names = ["dis24", "dis", "discharge", "river_discharge"]
    for name in preferred_names:
        if name in ds.data_vars:
            return name

    for name, variable in ds.data_vars.items():
        text = " ".join(
            str(variable.attrs.get(key, "")).lower()
            for key in ["standard_name", "long_name", "shortName", "units"]
        )
        if "discharge" in text or "river" in text or "m**3" in text or "m3" in text:
            return name

    return None


def _infer_file_year(path: Path) -> tuple[int, str]:
    match = re.search(r"(20\d{2})", str(path))
    if match:
        return int(match.group(1)), str(path)
    return 9999, str(path)


def discover_glofas_files(raw_dir: Path = RAW_GLOFAS_DIR) -> list[Path]:
    files = sorted(raw_dir.glob("*.nc"), key=_infer_file_year)
    if not files:
        raise FileNotFoundError(f"No GloFAS .nc files found in {raw_dir}")

    years = sorted({year for year, _ in (_infer_file_year(path) for path in files) if year != 9999})
    print(f"  Total GloFAS yearly files discovered: {len(files)}")
    if years:
        print(f"  Detected GloFAS year range: {years[0]}-{years[-1]}")
    else:
        print("  Detected GloFAS year range: unknown")
    return files


def _open_many_glofas_files(files: list[Path]) -> xr.Dataset:
    datasets = []
    try:
        for path in files:
            ds = xr.open_dataset(path, engine="netcdf4").load()
            ds = _standardize_coordinate_names(ds)
            if "lat" in ds.coords:
                ds = ds.sortby("lat")
            if "lon" in ds.coords:
                ds = ds.sortby("lon")
            datasets.append(ds)

        return xr.concat(
            datasets,
            dim="time",
            data_vars="all",
            coords="minimal",
            compat="override",
            combine_attrs="override",
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
        print(f"  Removing {duplicate_count} duplicate GloFAS timestamps.")
        ds = ds.isel(time=np.sort(unique_indexes))
    return ds


def process_glofas(
    raw_dir: Path = RAW_GLOFAS_DIR,
    output_path: Path = MULTIYEAR_OUTPUT,
    legacy_output_path: Path | None = LEGACY_2020_OUTPUT,
) -> Path:
    print("\nProcessing GloFAS data...")
    files = discover_glofas_files(raw_dir)
    print("  Merging yearly GloFAS files into one continuous dataset.")
    try:
        ds = _open_many_glofas_files(files)
    except Exception as exc:
        raise RuntimeError(f"Could not combine GloFAS files: {exc}") from exc

    ds = _standardize_coordinate_names(ds)
    ds = _sort_and_deduplicate_time(ds)

    discharge_var = _identify_discharge_variable(ds)
    if discharge_var:
        print(f"  Identified discharge variable: {discharge_var}")
        ds[discharge_var].attrs.setdefault("description", "GloFAS discharge variable")
    else:
        print("  Warning: could not identify a discharge variable automatically.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")

    if legacy_output_path and legacy_output_path != output_path:
        ds.to_netcdf(legacy_output_path)
        print(f"  Backward-compatible copy saved: {legacy_output_path.relative_to(PROJECT_ROOT)}")

    ds.close()
    return output_path
