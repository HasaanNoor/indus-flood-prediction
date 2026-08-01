from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.paths import OUTPUTS, PROJECT_ROOT
from src.spatial.configuration import SpatialPipelineConfig
from src.spatial.grid import grid_from_era5
from src.spatial.sentinel_ingestion import ingest_event_labels, valid_existing_event_outputs, write_event_metadata
from src.spatial.sentinel_inventory import DEFAULT_INVENTORY_PATH, SentinelEventConfig, load_event_inventory
from src.spatial.sentinel_validation import build_inventory_report, write_combined_outputs, write_inventory_reports


LABEL_ROOT = PROJECT_ROOT / "data_processed" / "spatial" / "labels"
EVENTS_DIR = LABEL_ROOT / "events"
COMBINED_DIR = LABEL_ROOT / "combined"
VALIDATION_DIR = OUTPUTS / "validation"


def _event_paths(event_id: str) -> tuple[Path, Path, Path]:
    event_dir = EVENTS_DIR / event_id
    return event_dir / "labels.parquet", event_dir / "metadata.json", event_dir / "validation.json"


def _read_existing_summary(validation_path: Path, event_id: str) -> dict[str, object]:
    summary = json.loads(validation_path.read_text())
    summary["status"] = "skipped_existing"
    summary["event_id"] = event_id
    return summary


def process_event(event: SentinelEventConfig, grid, overwrite: bool = False) -> tuple[pd.DataFrame | None, dict[str, object] | None, dict[str, object] | None]:
    labels_path, metadata_path, validation_path = _event_paths(event.event_id)
    if not event.mask_available or event.source_mask_path is None:
        return None, None, {
            "event_id": event.event_id,
            "status": "skipped_unavailable",
            "reason": event.validation_status,
            "sentinel1_available": event.sentinel1_available,
            "mask_available": event.mask_available,
        }
    if not overwrite and valid_existing_event_outputs(labels_path, metadata_path, validation_path, event.event_id):
        return pd.read_parquet(labels_path), _read_existing_summary(validation_path, event.event_id), None
    try:
        labels, summary = ingest_event_labels(event, grid, labels_path)
        write_event_metadata(event, summary, metadata_path, validation_path)
        return labels, summary, None
    except Exception as exc:
        return None, None, {"event_id": event.event_id, "status": "failed", "error": str(exc)}


def run_sentinel_pipeline(
    inventory_path: Path = DEFAULT_INVENTORY_PATH,
    event_id: str | None = None,
    all_events: bool = False,
    overwrite: bool = False,
    config: SpatialPipelineConfig | None = None,
) -> dict[str, object]:
    config = config or SpatialPipelineConfig()
    events = load_event_inventory(inventory_path)
    if event_id:
        selected = [event for event in events if event.event_id == event_id]
        if not selected:
            raise ValueError(f"Unknown Sentinel event_id: {event_id}")
    elif all_events:
        selected = events
    else:
        selected = []

    grid = grid_from_era5(config.era5_path, config.boundary_path)
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for event in selected:
        frame, summary, failure = process_event(event, grid, overwrite=overwrite)
        if frame is not None:
            frames.append(frame)
        if summary is not None:
            summaries.append(summary)
        if failure is not None:
            failures.append(failure)

    combined_outputs = write_combined_outputs(frames, summaries, COMBINED_DIR)
    report = build_inventory_report(events, summaries, failures)
    report_json, report_md = write_inventory_reports(
        report,
        VALIDATION_DIR / "sentinel_label_inventory_report.json",
        VALIDATION_DIR / "sentinel_label_inventory_report.md",
    )
    return {
        "selected_event_count": len(selected),
        "processed_event_count": len(summaries),
        "failure_count": len(failures),
        "combined_outputs": combined_outputs,
        "validation_report_json": str(report_json),
        "validation_report_md": str(report_md),
        "status": report["status"],
    }


def list_events(inventory_path: Path = DEFAULT_INVENTORY_PATH) -> list[dict[str, object]]:
    return [
        {
            "event_id": event.event_id,
            "event_name": event.event_name,
            "event_year": event.event_year,
            "validation_status": event.validation_status,
            "independent_event": event.independent_event,
            "mask_available": event.mask_available,
            "source_mask_path": None if event.source_mask_path is None else str(event.source_mask_path),
        }
        for event in load_event_inventory(inventory_path)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and validate multi-event Sentinel-1 spatial label inventories.")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY_PATH)
    parser.add_argument("--list-events", action="store_true")
    parser.add_argument("--event")
    parser.add_argument("--all-events", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_events:
        print(json.dumps(list_events(args.inventory), indent=2))
        return
    if bool(args.event) == bool(args.all_events):
        raise ValueError("Provide exactly one of --event or --all-events, or use --list-events.")
    outputs = run_sentinel_pipeline(args.inventory, event_id=args.event, all_events=args.all_events, overwrite=args.overwrite)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
