from pathlib import Path
import os

from paths import CLIPPED_DIR, FIGURES_DIR, PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy import ndimage
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
    boundary.boundary.plot(ax=ax, edgecolor="#111827", linewidth=1.2, zorder=5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def _format_date(value: np.datetime64) -> str:
    return np.datetime_as_string(value, unit="D")


def _format_time_range(data: xr.DataArray) -> str | None:
    if "time" not in data.coords or data.sizes.get("time", 0) == 0:
        return None

    start = _format_date(data["time"].values[0])
    end = _format_date(data["time"].values[-1])
    if start == end:
        return start

    year = start[:4] if start[:4] == end[:4] else None
    return f"{year} ({start} to {end})" if year else f"{start} to {end}"


def _sort_spatial(data: xr.DataArray) -> xr.DataArray:
    if "lat" in data.coords:
        data = data.sortby("lat")
    if "lon" in data.coords:
        data = data.sortby("lon")
    return data


def _interpolate_grid(data: xr.DataArray, factor: int = 5) -> xr.DataArray:
    data = _sort_spatial(data)
    if not {"lat", "lon"}.issubset(data.coords):
        return data

    lat = data["lat"].values
    lon = data["lon"].values
    if len(lat) < 2 or len(lon) < 2:
        return data

    smooth_lat = np.linspace(float(lat.min()), float(lat.max()), len(lat) * factor)
    smooth_lon = np.linspace(float(lon.min()), float(lon.max()), len(lon) * factor)
    return data.interp(lat=smooth_lat, lon=smooth_lon, method="linear")


def _finite_values(data: xr.DataArray) -> np.ndarray:
    values = np.asarray(data.values, dtype=float)
    return values[np.isfinite(values)]


def _lightly_smooth_masked(data: xr.DataArray, sigma: float = 1.0) -> xr.DataArray:
    values = np.asarray(data.values, dtype=float)
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    weights = valid.astype(float)

    smoothed = ndimage.gaussian_filter(filled, sigma=sigma)
    smoothed_weights = ndimage.gaussian_filter(weights, sigma=sigma)

    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = smoothed / smoothed_weights
    smoothed[smoothed_weights <= 0] = np.nan

    return xr.DataArray(smoothed, coords=data.coords, dims=data.dims, attrs=data.attrs)


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


def plot_precipitation_map_clean(
    input_path: Path = ERA5_CLIPPED,
    output_path: Path = FIGURES_DIR / "precipitation_map_clean.png",
    boundary_path: Path = DEFAULT_BOUNDARY_PATH,
) -> Path:
    print("\nPlotting clean precipitation map...")
    boundary = read_boundary(boundary_path)
    ds = xr.open_dataset(input_path, engine="netcdf4")
    try:
        var_name = _find_variable(ds, ["tp", "precipitation"], ["precip", "rain"])
        raw = ds[var_name]
        date_range = _format_time_range(raw)
        data = raw.sum("time", skipna=True) if "time" in raw.dims else raw
        data = _interpolate_grid(data, factor=5)

        finite = _finite_values(data)
        vmax = float(np.nanpercentile(finite, 98)) if finite.size else None
        if vmax and vmax > 0:
            levels = np.linspace(0, vmax, 18)
            extend = "max"
        else:
            levels = 18
            extend = "neither"

        fig, ax = plt.subplots(figsize=(8.5, 8.5))
        image = ax.contourf(
            data["lon"],
            data["lat"],
            data,
            levels=levels,
            cmap="YlGnBu",
            extend=extend,
            antialiased=True,
        )
        _plot_boundary(ax, boundary)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.042, pad=0.025)
        colorbar.set_label("Total precipitation (mm)")
        colorbar.ax.tick_params(labelsize=9)

        title = "ERA5 Total Precipitation Over Sindh"
        if date_range:
            title = f"{title} - {date_range}"
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#d1d5db", linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=250)
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


def plot_discharge_snapshot_clean(
    input_path: Path = GLOFAS_CLIPPED,
    output_path: Path = FIGURES_DIR / "discharge_snapshot_clean.png",
    boundary_path: Path = DEFAULT_BOUNDARY_PATH,
) -> Path:
    print("\nPlotting clean discharge snapshot...")
    boundary = read_boundary(boundary_path)
    ds = xr.open_dataset(input_path, engine="netcdf4")
    try:
        var_name = _find_variable(ds, ["dis24", "dis", "discharge"], ["discharge", "river"])
        raw = ds[var_name]
        timestamp = None
        if "time" in raw.dims:
            raw = raw.isel(time=0)
            timestamp = _format_date(ds["time"].values[0])

        data = _interpolate_grid(raw, factor=3)
        positive_values = _finite_values(data.where(data > 0))
        if positive_values.size:
            near_zero_threshold = max(1e-6, float(np.nanpercentile(positive_values, 15)))
            upper_clip = float(np.nanpercentile(positive_values, 99))
        else:
            near_zero_threshold = 1e-6
            upper_clip = 1.0

        data = data.where(data >= near_zero_threshold)
        data = data.clip(max=upper_clip)
        data = _lightly_smooth_masked(data, sigma=0.8)

        finite = _finite_values(data)
        vmin = max(near_zero_threshold, float(np.nanpercentile(finite, 5))) if finite.size else near_zero_threshold
        vmax = max(vmin * 1.01, float(np.nanpercentile(finite, 99))) if finite.size else upper_clip

        fig, ax = plt.subplots(figsize=(8.5, 8.5))
        image = ax.pcolormesh(
            data["lon"],
            data["lat"],
            data,
            cmap="magma_r",
            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            shading="gouraud",
        )
        _plot_boundary(ax, boundary)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.042, pad=0.025)
        colorbar.set_label("Discharge (m3 s-1, log scale)")
        colorbar.ax.tick_params(labelsize=9)

        title = "GloFAS Discharge Snapshot Over Sindh"
        if timestamp:
            title = f"{title} - {timestamp}"
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#d1d5db", linewidth=0.5, alpha=0.5)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=250)
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
    plot_precipitation_map_clean()
    plot_discharge_snapshot_clean()
    if not (FIGURES_DIR / "elevation_map.png").exists():
        plot_elevation_map()


if __name__ == "__main__":
    main()
