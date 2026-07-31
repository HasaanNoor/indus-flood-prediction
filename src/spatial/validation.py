from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

from src.spatial.grid import CanonicalGrid


def validate_grid(grid: CanonicalGrid) -> list[str]:
    issues: list[str] = []
    if grid.crs != "EPSG:4326":
        issues.append(f"Unexpected CRS: {grid.crs}")
    if grid.boundary_mask.shape != (grid.height, grid.width):
        issues.append("Boundary mask shape does not match grid dimensions.")
    if len(set(grid.grid_cell_ids.ravel())) != grid.cell_count:
        issues.append("Duplicate grid_cell_id values found.")
    if not np.all(np.diff(grid.longitudes) > 0):
        issues.append("Longitudes are not west-to-east.")
    if not np.all(np.diff(grid.latitudes) < 0):
        issues.append("Latitudes are not north-to-south.")
    return issues


def validate_feature_frame(df: pd.DataFrame, grid: CanonicalGrid) -> list[str]:
    issues: list[str] = []
    required = {"grid_cell_id", "date", "row", "col", "latitude", "longitude", "has_glofas_river_cell"}
    missing = required.difference(df.columns)
    if missing:
        issues.append(f"Missing required columns: {sorted(missing)}")
        return issues
    if df.duplicated(["grid_cell_id", "date"]).any():
        issues.append("Duplicate grid_cell_id/date rows found.")
    expected_per_date = grid.inside_cell_count
    counts = df.groupby("date")["grid_cell_id"].nunique()
    bad = counts[counts != expected_per_date]
    if not bad.empty:
        issues.append(f"Incomplete grid for dates: {bad.head().to_dict()}")
    dates = pd.DatetimeIndex(pd.to_datetime(df["date"]).sort_values().unique())
    if len(dates):
        missing_dates = pd.date_range(dates.min(), dates.max(), freq="D").difference(dates)
        if len(missing_dates):
            issues.append(f"Missing dates: {[str(value.date()) for value in missing_dates[:5]]}")
    coords = df[["grid_cell_id", "latitude", "longitude"]].drop_duplicates()
    if coords["grid_cell_id"].duplicated().any():
        issues.append("Coordinate drift: same grid_cell_id appears with multiple coordinates.")
    if not set(df["has_glofas_river_cell"].dropna().unique()).issubset({0, 1}):
        issues.append("River-cell indicator is not binary.")
    static_cols = [col for col in df.columns if col.startswith("terrain_") or col.startswith("relative_elevation")]
    for col in static_cols:
        nunique = df.groupby("grid_cell_id")[col].nunique(dropna=False)
        if int((nunique > 1).sum()):
            issues.append(f"Static feature drift detected: {col}")
    return issues


def validate_raster_alignment(path: Path, grid: CanonicalGrid) -> list[str]:
    issues: list[str] = []
    with rasterio.open(path) as src:
        if src.crs is None:
            issues.append(f"{path} has no CRS.")
        if src.crs and src.crs.to_string() != grid.crs:
            issues.append(f"{path} CRS differs from canonical grid: {src.crs}")
    return issues


def validate_transform_signature(
    transform: Affine,
    width: int,
    height: int,
    grid: CanonicalGrid,
    atol: float = 1e-9,
) -> None:
    if width != grid.width or height != grid.height:
        raise ValueError(f"Grid dimensions differ: {(width, height)} != {(grid.width, grid.height)}.")
    if not np.allclose(tuple(transform)[:6], tuple(grid.transform)[:6], atol=atol):
        raise ValueError("Affine transform differs from canonical grid.")


def validation_summary(grid: CanonicalGrid, features: pd.DataFrame | None = None, labels: pd.DataFrame | None = None) -> dict[str, object]:
    issues = validate_grid(grid)
    nodata_rates = {}
    if features is not None:
        issues.extend(validate_feature_frame(features, grid))
        numeric = features.select_dtypes(include="number")
        nodata_rates = numeric.isna().mean().sort_values(ascending=False).head(20).to_dict()
    if labels is not None:
        if labels.duplicated(["grid_cell_id", "event_id"]).any():
            issues.append("Duplicate label rows found for grid_cell_id/event_id.")
        if "observed_inundation_label" in labels.columns and not set(labels["observed_inundation_label"].dropna().unique()).issubset({0, 1}):
            issues.append("Observed inundation label is not binary.")
    return {
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "grid": {
            "crs": grid.crs,
            "width": grid.width,
            "height": grid.height,
            "sindh_cells": grid.inside_cell_count,
            "transform": list(tuple(grid.transform)[:6]),
            "bounds": list(grid.bounds),
        },
        "feature_rows": None if features is None else len(features),
        "feature_dates": None if features is None else int(pd.to_datetime(features["date"]).nunique()),
        "label_rows": None if labels is None else len(labels),
        "nodata_rates_top20": nodata_rates,
    }


def write_reports(summary: dict[str, object], json_path: Path, md_path: Path) -> tuple[Path, Path]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Spatial Grid Validation Report",
        "",
        f"Status: {summary['status']}",
        f"Grid: {summary['grid']['width']} x {summary['grid']['height']} ({summary['grid']['sindh_cells']} Sindh cells)",
        f"Feature rows: {summary.get('feature_rows')}",
        f"Feature dates: {summary.get('feature_dates')}",
        f"Label rows: {summary.get('label_rows')}",
        "",
        "## Issues",
    ]
    issues = summary.get("issues", [])
    lines.extend([f"- {issue}" for issue in issues] if issues else ["- None"])
    lines.append("")
    md_path.write_text("\n".join(lines))
    return json_path, md_path
