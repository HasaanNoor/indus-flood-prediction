from pathlib import Path
import os

from paths import CLIPPED_DIR, FIGURES_DIR, PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import xarray as xr

from plot_quick_visualizations import _find_variable


ERA5_CLIPPED = CLIPPED_DIR / "era5_sindh_2020_clipped.nc"
GLOFAS_CLIPPED = CLIPPED_DIR / "glofas_sindh_2020_clipped.nc"


def plot_average_rainfall_over_time(
    input_path: Path = ERA5_CLIPPED,
    output_path: Path = FIGURES_DIR / "average_rainfall_over_time.png",
) -> Path:
    print("\nPlotting average rainfall over time...")
    ds = xr.open_dataset(input_path, engine="netcdf4")
    try:
        var_name = _find_variable(ds, ["tp", "precipitation"], ["precip", "rain"])
        rainfall = ds[var_name].mean(dim=[dim for dim in ["lat", "lon"] if dim in ds[var_name].dims])
        if "time" in rainfall.dims:
            rainfall = rainfall.resample(time="1D").sum(skipna=True)

        fig, ax = plt.subplots(figsize=(12, 4.8))
        rainfall.plot(ax=ax, color="#2563eb", linewidth=1.4)
        ax.set_title("Average Rainfall Over Sindh")
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily area-average rainfall (mm)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    finally:
        ds.close()

    print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")
    return output_path


def plot_average_discharge_over_time(
    input_path: Path = GLOFAS_CLIPPED,
    output_path: Path = FIGURES_DIR / "average_discharge_over_time.png",
) -> Path:
    print("\nPlotting average discharge over time...")
    ds = xr.open_dataset(input_path, engine="netcdf4")
    try:
        var_name = _find_variable(ds, ["dis24", "dis", "discharge"], ["discharge", "river"])
        discharge = ds[var_name].mean(dim=[dim for dim in ["lat", "lon"] if dim in ds[var_name].dims])

        fig, ax = plt.subplots(figsize=(12, 4.8))
        discharge.plot(ax=ax, color="#047857", linewidth=1.4)
        ax.set_title("Average Discharge Over Sindh")
        ax.set_xlabel("Date")
        ax.set_ylabel("Area-average discharge (m3 s-1)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    finally:
        ds.close()

    print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")
    return output_path


def main() -> None:
    plot_average_rainfall_over_time()
    plot_average_discharge_over_time()


if __name__ == "__main__":
    main()
