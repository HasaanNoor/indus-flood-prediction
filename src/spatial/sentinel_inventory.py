from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.paths import PROJECT_ROOT


DEFAULT_INVENTORY_PATH = PROJECT_ROOT / "data_processed" / "spatial" / "labels" / "inventory" / "sentinel_event_inventory.json"


@dataclass(frozen=True)
class SentinelEventConfig:
    event_id: str
    event_name: str
    event_year: int
    event_start_date: str
    event_end_date: str
    representative_date: str
    baseline_start_date: str | None
    baseline_end_date: str | None
    data_source: str
    sentinel1_collection: str
    polarization: str
    orbit_pass: str | None
    orbit_strategy: str
    threshold_method: str
    selected_threshold: float | None
    threshold_alternatives_tested: list[float]
    flood_area_estimate_km2: float | None
    permanent_water_mask_source: str
    permanent_water_occurrence_threshold: int | None
    slope_threshold_degrees: float | None
    connected_pixel_threshold: int | None
    source_mask_path: Path | None
    permanent_water_mask_path: Path | None
    sentinel1_available: bool
    mask_available: bool
    validation_status: str
    independent_event: bool
    parent_event_id: str | None
    notes: str
    source_references: list[str]
    processed_status: str | None
    export_timestamp: str | None
    baseline_image_count: int | None
    event_image_count: int | None
    flood_cell_count: int | None
    non_flood_count: int | None
    permanent_water_count: int | None
    raster_sha256: str | None
    label_sha256: str | None
    processing_version: str | None
    raw: dict[str, Any]

    @property
    def independence_key(self) -> str:
        return self.parent_event_id or self.event_id


def _resolve_optional_path(value: str | None, base_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _event_from_dict(record: dict[str, Any], base_dir: Path) -> SentinelEventConfig:
    required = {
        "event_id",
        "event_name",
        "event_year",
        "event_start_date",
        "event_end_date",
        "representative_date",
        "data_source",
        "sentinel1_collection",
        "polarization",
        "orbit_strategy",
        "threshold_method",
        "permanent_water_mask_source",
        "sentinel1_available",
        "mask_available",
        "validation_status",
        "independent_event",
        "notes",
        "source_references",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"Sentinel event inventory row is missing required fields: {missing}")
    return SentinelEventConfig(
        event_id=str(record["event_id"]),
        event_name=str(record["event_name"]),
        event_year=int(record["event_year"]),
        event_start_date=str(record["event_start_date"]),
        event_end_date=str(record["event_end_date"]),
        representative_date=str(record["representative_date"]),
        baseline_start_date=record.get("baseline_start_date"),
        baseline_end_date=record.get("baseline_end_date"),
        data_source=str(record["data_source"]),
        sentinel1_collection=str(record["sentinel1_collection"]),
        polarization=str(record["polarization"]),
        orbit_pass=record.get("orbit_pass"),
        orbit_strategy=str(record["orbit_strategy"]),
        threshold_method=str(record["threshold_method"]),
        selected_threshold=None if record.get("selected_threshold") is None else float(record["selected_threshold"]),
        threshold_alternatives_tested=[float(v) for v in record.get("threshold_alternatives_tested", [])],
        flood_area_estimate_km2=None if record.get("flood_area_estimate_km2") is None else float(record["flood_area_estimate_km2"]),
        permanent_water_mask_source=str(record["permanent_water_mask_source"]),
        permanent_water_occurrence_threshold=(
            None if record.get("permanent_water_occurrence_threshold") is None else int(record["permanent_water_occurrence_threshold"])
        ),
        slope_threshold_degrees=None if record.get("slope_threshold_degrees") is None else float(record["slope_threshold_degrees"]),
        connected_pixel_threshold=None if record.get("connected_pixel_threshold") is None else int(record["connected_pixel_threshold"]),
        source_mask_path=_resolve_optional_path(record.get("source_mask_path"), base_dir),
        permanent_water_mask_path=_resolve_optional_path(record.get("permanent_water_mask_path"), base_dir),
        sentinel1_available=bool(record["sentinel1_available"]),
        mask_available=bool(record["mask_available"]),
        validation_status=str(record["validation_status"]),
        independent_event=bool(record["independent_event"]),
        parent_event_id=record.get("parent_event_id"),
        notes=str(record["notes"]),
        source_references=[str(v) for v in record.get("source_references", [])],
        processed_status=record.get("processed_status"),
        export_timestamp=record.get("export_timestamp"),
        baseline_image_count=None if record.get("baseline_image_count") is None else int(record["baseline_image_count"]),
        event_image_count=None if record.get("event_image_count") is None else int(record["event_image_count"]),
        flood_cell_count=None if record.get("flood_cell_count") is None else int(record["flood_cell_count"]),
        non_flood_count=None if record.get("non_flood_count") is None else int(record["non_flood_count"]),
        permanent_water_count=None if record.get("permanent_water_count") is None else int(record["permanent_water_count"]),
        raster_sha256=record.get("raster_sha256"),
        label_sha256=record.get("label_sha256"),
        processing_version=record.get("processing_version"),
        raw=dict(record),
    )


def load_event_inventory(path: Path = DEFAULT_INVENTORY_PATH, base_dir: Path = PROJECT_ROOT) -> list[SentinelEventConfig]:
    if not path.exists():
        raise FileNotFoundError(f"Sentinel event inventory not found: {path}")
    payload = json.loads(path.read_text())
    events = [_event_from_dict(record, base_dir) for record in payload.get("events", [])]
    validate_event_inventory(events)
    return events


def load_inventory_payload(path: Path = DEFAULT_INVENTORY_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Sentinel event inventory not found: {path}")
    return json.loads(path.read_text())


def write_inventory_payload(payload: dict[str, Any], path: Path = DEFAULT_INVENTORY_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _event_status(record: dict[str, Any]) -> str:
    if not record.get("sentinel1_available", False):
        return "unavailable"
    if not record.get("mask_available", False):
        return "pending_gee_export"
    return "verified_export_available"


def update_inventory_records(
    path: Path,
    summaries: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> Path:
    payload = load_inventory_payload(path)
    payload["schema_version"] = "phase15_sentinel_event_inventory_v2"
    summary_by_id = {str(summary["event_id"]): summary for summary in summaries}
    failure_by_id = {str(failure["event_id"]): failure for failure in failures}
    for record in payload.get("events", []):
        event_id = str(record["event_id"])
        if event_id in summary_by_id:
            summary = summary_by_id[event_id]
            record["processed_status"] = "processed"
            record["validation_status"] = "processed"
            record["source_mask_sha256"] = summary.get("source_mask_sha256")
            record["raster_sha256"] = summary.get("source_mask_sha256")
            record["label_sha256"] = summary.get("label_array_sha256")
            record["label_array_sha256"] = summary.get("label_array_sha256")
            record["flood_cell_count"] = summary.get("flood_cells")
            record["non_flood_count"] = summary.get("non_flood_cells")
            record["permanent_water_count"] = summary.get("permanent_water_cells")
            record["validation_result"] = summary.get("validation_result", "passed")
            record["processing_version"] = summary.get("processing_version")
            record["export_timestamp"] = summary.get("export_timestamp", record.get("export_timestamp"))
            record["baseline_image_count"] = summary.get("baseline_image_count", record.get("baseline_image_count"))
            record["event_image_count"] = summary.get("event_image_count", record.get("event_image_count"))
        elif event_id in failure_by_id:
            record["processed_status"] = "failed"
            record["validation_status"] = "failed"
            record["validation_result"] = failure_by_id[event_id].get("error")
        else:
            status = _event_status(record)
            if record.get("processed_status") in {None, "failed"}:
                record["processed_status"] = status
            if record.get("validation_status") == "failed":
                record["validation_status"] = status
    return write_inventory_payload(payload, path)


def validate_event_inventory(events: list[SentinelEventConfig]) -> None:
    ids = [event.event_id for event in events]
    duplicates = sorted({event_id for event_id in ids if ids.count(event_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate Sentinel event IDs found: {duplicates}")
    known = set(ids)
    missing_parents = sorted({event.parent_event_id for event in events if event.parent_event_id and event.parent_event_id not in known})
    if missing_parents:
        raise ValueError(f"Threshold variants reference missing parent events: {missing_parents}")
    for event in events:
        if event.mask_available and event.source_mask_path is None:
            raise ValueError(f"{event.event_id} is marked mask_available but has no source_mask_path.")
        if not event.independent_event and not event.parent_event_id:
            raise ValueError(f"{event.event_id} is a non-independent variant but has no parent_event_id.")


def independent_available_events(events: list[SentinelEventConfig]) -> list[SentinelEventConfig]:
    return [event for event in events if event.independent_event and event.sentinel1_available]
