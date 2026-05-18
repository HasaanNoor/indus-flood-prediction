from pathlib import Path

import geopandas as gpd
import rioxarray  # noqa: F401  Required for the .rio accessor.
import xarray as xr

from paths import PROCESSED_BOUNDARIES_DIR


DEFAULT_BOUNDARY_PATH = PROCESSED_BOUNDARIES_DIR / "sindh_boundary.geojson"


def read_boundary(boundary_path: Path = DEFAULT_BOUNDARY_PATH) -> gpd.GeoDataFrame:
    """Read the Sindh boundary and return it in EPSG:4326."""
    if not boundary_path.exists():
        raise FileNotFoundError(
            f"Sindh boundary not found at {boundary_path}. Run process_boundary.py first."
        )

    boundary = gpd.read_file(boundary_path)
    if boundary.empty:
        raise ValueError(f"Boundary file is empty: {boundary_path}")

    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326")
    elif boundary.crs.to_epsg() != 4326:
        boundary = boundary.to_crs("EPSG:4326")

    return boundary


def standardize_lat_lon_names(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {}
    if "latitude" in ds.coords or "latitude" in ds.dims:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.coords or "longitude" in ds.dims:
        rename_map["longitude"] = "lon"
    if "valid_time" in ds.coords or "valid_time" in ds.dims:
        rename_map["valid_time"] = "time"
    return ds.rename(rename_map) if rename_map else ds


def clip_netcdf_to_boundary(
    input_path: Path,
    output_path: Path,
    boundary_path: Path = DEFAULT_BOUNDARY_PATH,
) -> Path:
    """Clip a lat/lon NetCDF dataset to the Sindh boundary polygon."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input NetCDF not found: {input_path}")

    boundary = read_boundary(boundary_path)
    ds = xr.open_dataset(input_path, engine="netcdf4")
    try:
        ds = standardize_lat_lon_names(ds)
        if "lat" not in ds.coords or "lon" not in ds.coords:
            raise ValueError(
                f"Expected lat/lon coordinates in {input_path}. Found: {list(ds.coords)}"
            )

        ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
        ds = ds.rio.write_crs("EPSG:4326", inplace=False)
        clipped = ds.rio.clip(
            boundary.geometry,
            boundary.crs,
            drop=True,
            all_touched=True,
        )
        clipped.attrs.update(ds.attrs)
        clipped.attrs["clip_boundary"] = "Sindh, Pakistan"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        clipped.to_netcdf(output_path)
        clipped.close()
    finally:
        ds.close()

    return output_path
