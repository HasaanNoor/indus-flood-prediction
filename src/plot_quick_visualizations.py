from pathlib import Path
import os

from paths import CLIPPED_DIR, FIGURES_DIR, PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
import xarray as xr

from clip_utils import DEFAULT_BOUNDARY_PATH, read_boundary


ERA5_CLIPPED = CLIPPED_DIR / "era5_sindh_2020_clipped.nc"
GLOFAS_CLIPPED = CLIPPED_DIR / "glofas_sindh_2020_clipped.nc"
SRTM_CLIPPED = CLIPPED_DIR / "srtm_sindh_clipped.tif"


def _find_variable(ds: xr.Dataset, preferred: list[str], keywords: list[str]) -> str:
    for name in preferred:
        if name in ds.data_vars:
            return name

    for name, variable in ds.data_vars.items():
        text = " ".join(
            str(variable.attrs.get(key, "")).lower()
            for key in ["standard_name", "long_name", "description", "units"]
        )
        if any(keyword in text for keyword in keywords):
            return name

    raise ValueError(f"Could not identify variable. Available variables: {list(ds.data_vars)}")


def _plot_boundary(ax: plt.Axes, boundary: gpd.GeoDataFrame) -> None:
    boundary.boundary.plot(ax=ax, edgecolor="black", linewidth=1.1)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def plot_precipitation_map(
    input_path: Path = ERA5_CLIPPED,
    output_path: Path = FIGURES_DIR / "precipitation_map.png",
    boundary_path: Path = DEFAULT_BOUNDARY_PATH,
) -> Path:
    print("\nPlotting precipitation map...")
    boundary = read_boundary(boundary_path)
    ds = xr.open_dataset(input_path, engine="netcdf4")
    try:
        var_name = _find_variable(ds, ["tp", "precipitation"], ["precip", "rain"])
        data = ds[var_name]
        if "time" in data.dims:
            data = data.sum("time", skipna=True)

        fig, ax = plt.subplots(figsize=(8, 8))
        data.plot(ax=ax, cmap="Blues", cbar_kwargs={"label": "Total precipitation (mm)"})
        _plot_boundary(ax, boundary)
        ax.set_title("ERA5 Total Precipitation Over Sindh")
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    finally:
        ds.close()

    print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")
    return output_path


def plot_discharge_snapshot(
    input_path: Path = GLOFAS_CLIPPED,
    output_path: Path = FIGURES_DIR / "discharge_snapshot.png",
    boundary_path: Path = DEFAULT_BOUNDARY_PATH,
) -> Path:
    print("\nPlotting discharge snapshot...")
    boundary = read_boundary(boundary_path)
    ds = xr.open_dataset(input_path, engine="netcdf4")
    try:
        var_name = _find_variable(ds, ["dis24", "dis", "discharge"], ["discharge", "river"])
        data = ds[var_name]
        timestamp = None
        if "time" in data.dims:
            data = data.isel(time=0)
            timestamp = np.datetime_as_string(ds["time"].values[0], unit="D")

        fig, ax = plt.subplots(figsize=(8, 8))
        data.plot(ax=ax, cmap="viridis", cbar_kwargs={"label": "Discharge (m3 s-1)"})
        _plot_boundary(ax, boundary)
        title = "GloFAS Discharge Snapshot"
        if timestamp:
            title = f"{title} ({timestamp})"
        ax.set_title(title)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    finally:
        ds.close()

    print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")
    return output_path


def plot_elevation_map(
    input_path: Path = SRTM_CLIPPED,
    output_path: Path = FIGURES_DIR / "elevation_map.png",
    boundary_path: Path = DEFAULT_BOUNDARY_PATH,
) -> Path:
    print("\nPlotting elevation map...")
    boundary = read_boundary(boundary_path)
    with rasterio.open(input_path) as src:
        scale = max(1, int(np.ceil(max(src.width, src.height) / 1600)))
        out_shape = (max(1, src.height // scale), max(1, src.width // scale))
        elevation = src.read(1, masked=True, out_shape=out_shape, resampling=Resampling.bilinear)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    fig, ax = plt.subplots(figsize=(8, 9))
    image = ax.imshow(elevation, extent=extent, origin="upper", cmap="terrain")
    fig.colorbar(image, ax=ax, label="Elevation (m)")
    _plot_boundary(ax, boundary)
    ax.set_title("SRTM Elevation Over Sindh")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")
    return output_path


def main() -> None:
    plot_precipitation_map()
    plot_discharge_snapshot()
    plot_elevation_map()


if __name__ == "__main__":
    main()
