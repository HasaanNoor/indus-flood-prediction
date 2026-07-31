from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.spatial.configuration import (
    DEFAULT_END_DATE,
    DEFAULT_PILOT_END_DATE,
    DEFAULT_PILOT_START_DATE,
    DEFAULT_START_DATE,
    SPATIAL_FEATURES_DIR,
    SPATIAL_LABELS_DIR,
    SPATIAL_VALIDATION_DIR,
    SpatialPipelineConfig,
)
from src.spatial.features import build_spatial_features
from src.spatial.grid import grid_from_era5, save_grid_outputs
from src.spatial.labels import build_sentinel1_labels, discover_sentinel1_events, label_availability_report
from src.spatial.temporal import date_range, partition_name
from src.spatial.validation import validation_summary, write_reports


def feature_output_path(start_date: str, end_date: str, partition_by_year: bool = False) -> Path:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    name = partition_name(start, end, partition_by_year=partition_by_year)
    safe_name = name.replace("=", "_")
    return SPATIAL_FEATURES_DIR / f"spatial_features_{safe_name}.parquet"


def run_pipeline(
    start_date: str = DEFAULT_PILOT_START_DATE,
    end_date: str = DEFAULT_PILOT_END_DATE,
    build_grid: bool = False,
    include_labels: bool = False,
    partition_by_year: bool = False,
    overwrite: bool = False,
    config: SpatialPipelineConfig | None = None,
) -> dict[str, object]:
    config = config or SpatialPipelineConfig()
    grid = grid_from_era5(config.era5_path, config.boundary_path)
    if build_grid or not config.grid_metadata_path.exists() or not config.grid_cells_path.exists():
        save_grid_outputs(grid, config.grid_metadata_path, config.grid_cells_path)

    outputs = {"grid_metadata": str(config.grid_metadata_path), "grid_cells": str(config.grid_cells_path), "features": [], "labels": []}
    label_frame = None
    ranges = [(pd.Timestamp(start_date), pd.Timestamp(end_date))]
    if partition_by_year:
        requested = date_range(start_date, end_date)
        ranges = []
        for year, values in pd.Series(requested).groupby(requested.year):
            ranges.append((pd.Timestamp(values.min()), pd.Timestamp(values.max())))

    feature_frames = []
    for start, end in ranges:
        path = feature_output_path(str(start.date()), str(end.date()), partition_by_year=partition_by_year)
        if path.exists() and not overwrite:
            frame = pd.read_parquet(path)
        else:
            frame = build_spatial_features(config, grid, str(start.date()), str(end.date()), output_path=path)
        outputs["features"].append(str(path))
        feature_frames.append(frame)
    features = pd.concat(feature_frames, ignore_index=True) if feature_frames else None

    if include_labels:
        events = discover_sentinel1_events(config.sentinel1_dir)
        report = label_availability_report(events)
        (SPATIAL_LABELS_DIR / "label_availability_report.json").write_text(json.dumps(report, indent=2) + "\n")
        label_frames = []
        for event in events:
            path = SPATIAL_LABELS_DIR / f"sentinel1_labels_{event.event_id}.parquet"
            if path.exists() and not overwrite:
                current_labels = pd.read_parquet(path)
            else:
                current_labels = build_sentinel1_labels(event, grid, path)
            outputs["labels"].append(str(path))
            label_frames.append(current_labels)
        if label_frames:
            label_frame = pd.concat(label_frames, ignore_index=True)

    summary = validation_summary(grid, features, label_frame)
    json_path = SPATIAL_VALIDATION_DIR / "spatial_grid_report.json"
    md_path = SPATIAL_VALIDATION_DIR / "spatial_grid_report.md"
    write_reports(summary, json_path, md_path)
    outputs["validation_report_json"] = str(json_path)
    outputs["validation_report_md"] = str(md_path)
    outputs["validation_status"] = summary["status"]
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Sindh spatial-temporal feature grid.")
    parser.add_argument("--build-grid", action="store_true")
    parser.add_argument("--start-date", default=DEFAULT_PILOT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_PILOT_END_DATE)
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--partition-by-year", action="store_true")
    parser.add_argument("--include-labels", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = args.start_date
    end_date = args.end_date
    if args.start_year is not None or args.end_year is not None:
        if args.start_year is None or args.end_year is None:
            raise ValueError("--start-year and --end-year must be provided together.")
        start_date = f"{args.start_year}-01-01"
        end_date = f"{args.end_year}-12-31"
        if args.start_year == 2010:
            start_date = DEFAULT_START_DATE
        if args.end_year == 2023:
            end_date = DEFAULT_END_DATE
    outputs = run_pipeline(
        start_date=start_date,
        end_date=end_date,
        build_grid=args.build_grid,
        include_labels=args.include_labels,
        partition_by_year=args.partition_by_year,
        overwrite=args.overwrite,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
