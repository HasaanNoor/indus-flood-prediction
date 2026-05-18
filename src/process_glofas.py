from pathlib import Path

import xarray as xr

from paths import PROJECT_ROOT, RAW_GLOFAS_DIR, PROCESSED_GLOFAS_DIR


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


def process_glofas(
    raw_dir: Path = RAW_GLOFAS_DIR,
    output_path: Path = PROCESSED_GLOFAS_DIR / "glofas_sindh_2020_clean.nc",
) -> Path:
    print("\nProcessing GloFAS data...")
    files = sorted(raw_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No GloFAS .nc files found in {raw_dir}")
    if len(files) > 1:
        print(f"  Found {len(files)} files; using the first one: {files[0].name}")

    input_path = files[0]
    print(f"  Reading: {input_path.relative_to(PROJECT_ROOT)}")
    ds = xr.open_dataset(input_path, engine="netcdf4")
    ds = _standardize_coordinate_names(ds)
    if "time" in ds.coords:
        ds = ds.sortby("time")

    discharge_var = _identify_discharge_variable(ds)
    if discharge_var:
        print(f"  Identified discharge variable: {discharge_var}")
        ds[discharge_var].attrs.setdefault("description", "GloFAS discharge variable")
    else:
        print("  Warning: could not identify a discharge variable automatically.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    ds.close()
    print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")
    return output_path
