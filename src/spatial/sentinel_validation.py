from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from src.spatial.sentinel_inventory import SentinelEventConfig


def build_inventory_report(
    events: list[SentinelEventConfig],
    processed_summaries: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> dict[str, object]:
    independent = [event for event in events if event.independent_event]
    processable = [event for event in independent if event.mask_available and event.source_mask_path is not None]
    unavailable = [event for event in events if not event.sentinel1_available or not event.mask_available]
    processed_ids = {str(s["event_id"]) for s in processed_summaries}
    pending = [event for event in independent if event.sentinel1_available and not event.mask_available]
    unavailable_only = [event for event in independent if not event.sentinel1_available]
    source_hashes = [str(s.get("source_mask_sha256")) for s in processed_summaries if s.get("source_mask_sha256")]
    label_hashes = [str(s.get("label_array_sha256")) for s in processed_summaries if s.get("label_array_sha256")]
    duplicate_source_hashes = sorted([h for h, count in Counter(source_hashes).items() if count > 1])
    duplicate_label_hashes = sorted([h for h, count in Counter(label_hashes).items() if count > 1])
    grouped_variants: dict[str, list[str]] = defaultdict(list)
    for event in events:
        grouped_variants[event.parent_event_id or event.event_id].append(event.event_id)
    threshold_variant_groups = {k: v for k, v in grouped_variants.items() if len(v) > 1}

    totals = {
        "rows": sum(int(s.get("rows", 0)) for s in processed_summaries),
        "valid_rows": sum(int(s.get("valid_rows", 0)) for s in processed_summaries),
        "flood_cells": sum(int(s.get("flood_cells", 0)) for s in processed_summaries),
        "non_flood_cells": sum(int(s.get("non_flood_cells", 0)) for s in processed_summaries),
        "permanent_water_cells": sum(int(s.get("permanent_water_cells", 0)) for s in processed_summaries),
        "nodata_cells": sum(int(s.get("nodata_cells", 0)) for s in processed_summaries),
    }
    class_balance_by_event = [
        {
            "event_id": s.get("event_id"),
            "valid_rows": int(s.get("valid_rows", 0)),
            "flood_cells": int(s.get("flood_cells", 0)),
            "non_flood_cells": int(s.get("non_flood_cells", 0)),
            "permanent_water_cells": int(s.get("permanent_water_cells", 0)),
        }
        for s in processed_summaries
    ]
    issues = []
    if duplicate_source_hashes:
        issues.append("Duplicate source raster hashes detected.")
    if duplicate_label_hashes:
        issues.append("Duplicate label arrays detected.")
    return {
        "status": "passed" if not issues and not failures else "partial" if processed_summaries else "failed",
        "issues": issues,
        "inventory_event_count": len(events),
        "independent_event_count": len(independent),
        "sentinel1_available_independent_event_count": len([e for e in independent if e.sentinel1_available]),
        "processable_mask_event_count": len(processable),
        "processed_event_count": len(processed_summaries),
        "processed_events": sorted(processed_ids),
        "pending_events": [event.event_id for event in pending],
        "unavailable_events": [event.event_id for event in unavailable_only],
        "unavailable_or_pending_events": [
            {
                "event_id": event.event_id,
                "validation_status": event.validation_status,
                "sentinel1_available": event.sentinel1_available,
                "mask_available": event.mask_available,
                "notes": event.notes,
            }
            for event in unavailable
        ],
        "threshold_variant_groups": threshold_variant_groups,
        "duplicate_source_hashes": duplicate_source_hashes,
        "duplicate_label_array_hashes": duplicate_label_hashes,
        "duplicate_events": sorted(set(duplicate_source_hashes + duplicate_label_hashes)),
        "class_balance_by_event": class_balance_by_event,
        "events": processed_summaries,
        "failures": failures,
        "totals": totals,
        "scientific_guardrails": [
            "Threshold variants are not counted as independent events.",
            "Permanent water is tracked separately from candidate observed inundation.",
            "No model retraining is performed in Phase 14.",
            "No model retraining is performed in Phase 15.",
            "Unavailable and pending events remain visible in the inventory."
        ],
    }


def write_inventory_reports(report: dict[str, object], json_path: Path, md_path: Path) -> tuple[Path, Path]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Sentinel-1 Label Inventory Validation Report",
        "",
        f"Status: {report['status']}",
        f"Inventory events: {report['inventory_event_count']}",
        f"Independent events: {report['independent_event_count']}",
        f"Processable mask events: {report['processable_mask_event_count']}",
        f"Processed events: {report['processed_event_count']}",
        f"Pending events: {len(report['pending_events'])}",
        f"Unavailable events: {len(report['unavailable_events'])}",
        "",
        "## Label Totals",
    ]
    totals = report["totals"]
    for key in ["rows", "valid_rows", "flood_cells", "non_flood_cells", "permanent_water_cells", "nodata_cells"]:
        lines.append(f"- {key}: {totals[key]}")
    lines.extend(["", "## Processed Events"])
    if report["events"]:
        for event in report["events"]:
            lines.append(
                f"- {event['event_id']}: rows={event['rows']}, valid={event['valid_rows']}, "
                f"flood={event['flood_cells']}, non_flood={event['non_flood_cells']}, "
                f"permanent_water={event['permanent_water_cells']}, nodata={event['nodata_cells']}"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Class Balance By Event"])
    if report["class_balance_by_event"]:
        for event in report["class_balance_by_event"]:
            lines.append(
                f"- {event['event_id']}: flood={event['flood_cells']}, "
                f"non_flood={event['non_flood_cells']}, permanent_water={event['permanent_water_cells']}"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Unavailable Or Pending Events"])
    pending = report["unavailable_or_pending_events"]
    lines.extend([f"- {event['event_id']}: {event['validation_status']}" for event in pending] if pending else ["- None"])
    lines.extend(["", "## Duplicate And Independence Checks"])
    lines.append(f"- Duplicate source raster hashes: {len(report['duplicate_source_hashes'])}")
    lines.append(f"- Duplicate label array hashes: {len(report['duplicate_label_array_hashes'])}")
    lines.append(f"- Threshold variant groups: {len(report['threshold_variant_groups'])}")
    lines.extend(["", "## Issues"])
    lines.extend([f"- {issue}" for issue in report["issues"]] if report["issues"] else ["- None"])
    lines.append("")
    md_path.write_text("\n".join(lines))
    return json_path, md_path


def write_combined_outputs(frames: list[pd.DataFrame], summaries: list[dict[str, object]], combined_dir: Path) -> dict[str, str | None]:
    combined_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str | None] = {"labels": None, "summary": None}
    if frames:
        labels = pd.concat(frames, ignore_index=True)
        if labels.duplicated(["event_id", "grid_cell_id"]).any():
            raise ValueError("Combined Sentinel labels contain duplicate event_id/grid_cell_id pairs.")
        required = {
            "event_id",
            "grid_cell_id",
            "observed_inundation_label",
            "permanent_water_label",
            "event_date",
            "threshold",
            "processing_version",
        }
        missing = required.difference(labels.columns)
        if missing:
            raise ValueError(f"Combined Sentinel labels are missing required columns: {sorted(missing)}")
        labels = labels.sort_values(["event_id", "grid_cell_id"]).reset_index(drop=True)
        labels_path = combined_dir / "sentinel_labels_all_events.parquet"
        labels.to_parquet(labels_path, index=False)
        outputs["labels"] = str(labels_path)
    if summaries:
        summary_path = combined_dir / "label_inventory_summary.csv"
        pd.DataFrame(summaries).to_csv(summary_path, index=False)
        outputs["summary"] = str(summary_path)
    return outputs
