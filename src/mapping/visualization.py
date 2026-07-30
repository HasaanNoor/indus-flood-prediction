from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    from src.paths import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover
    from paths import PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from src.mapping.configuration import RISK_CLASSES
from src.mapping.raster_export import PROBABILITY_NODATA, RISK_NODATA, SpatialGrid


def _extent(grid: SpatialGrid) -> tuple[float, float, float, float]:
    left = grid.transform.c
    top = grid.transform.f
    right = left + grid.width * grid.resolution[0]
    bottom = top - grid.height * grid.resolution[1]
    return left, right, bottom, top


def _plot_boundary(ax: plt.Axes, boundary_path: Path | None, crs: str) -> None:
    if boundary_path is None or not boundary_path.exists():
        return
    boundary = gpd.read_file(boundary_path)
    if boundary.empty:
        return
    if boundary.crs is None:
        boundary = boundary.set_crs(crs)
    elif str(boundary.crs) != crs:
        boundary = boundary.to_crs(crs)
    boundary.boundary.plot(ax=ax, edgecolor="#111827", linewidth=0.9)


def plot_probability_map(
    array: np.ndarray,
    grid: SpatialGrid,
    output_path: Path,
    horizon: str,
    boundary_path: Path | None = None,
) -> Path:
    masked = np.ma.masked_where(array == PROBABILITY_NODATA, array)
    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(masked, extent=_extent(grid), origin="upper", cmap="YlGnBu", vmin=0, vmax=1)
    _plot_boundary(ax, boundary_path, grid.crs)
    ax.set_title(f"Model-estimated flood probability ({horizon})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.042, pad=0.025)
    colorbar.set_label("Flood probability")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_risk_map(
    array: np.ndarray,
    grid: SpatialGrid,
    output_path: Path,
    horizon: str,
    boundary_path: Path | None = None,
) -> Path:
    masked = np.ma.masked_where(array == RISK_NODATA, array)
    cmap = ListedColormap(["#2c7bb6", "#abd9e9", "#fdae61", "#d7191c"])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(masked, extent=_extent(grid), origin="upper", cmap=cmap, norm=norm)
    _plot_boundary(ax, boundary_path, grid.crs)
    ax.set_title(f"Model-based flood probability category ({horizon})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=9, color=cmap(index - 1), label=label)
        for index, label in RISK_CLASSES.items()
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
