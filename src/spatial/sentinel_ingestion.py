from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from src.spatial.alignment import write_dataframe_partition
from src.spatial.grid import CanonicalGrid, grid_cells_dataframe
from src.spatial.sentinel_alignment import NODATA_CODE, align_label_raster, inspect_raster
from src.spatial.sentinel_inventory import SentinelEventConfig


@dataclass(frozen=True)
class EventIngestionResult:
    event_id: str
    status: str
    labels_path: Path | None
    metadata_path: Path | None
    validation_path: Path | None
    summary: dict[str, object]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_array_sha256(observed: np.ndarray, permanent: np.ndarray, valid: np.ndarray) -> str:
    stacked = np.stack([observed.astype("uint8"), permanent.astype("uint8"), valid.astype("uint8")])
    return hashlib.sha256(stacked.tobytes()).hexdigest()


def _read_optional_permanent_water(event: SentinelEventConfig, grid: CanonicalGrid, source_profile_count: int) -> tuple[np.ndarray, str]:
    if event.permanent_water_mask_path is not None:
        return align_label_raster(event.permanent_water_mask_path, grid, 1), "separate_raster"
    if event.source_mask_path is not None and source_profile_count >= 2:
        return align_label_raster(event.source_mask_path, grid, 2), "source_band_2"
    return np.zeros((grid.height, grid.width), dtype="uint8"), "not_supplied"


def ingest_event_labels(event: SentinelEventConfig, grid: CanonicalGrid, labels_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    if event.source_mask_path is None:
        raise FileNotFoundError(f"{event.event_id} has no source_mask_path in the inventory.")
    if not event.source_mask_path.exists():
        raise FileNotFoundError(f"Sentinel mask not found for {event.event_id}: {event.source_mask_path}")

    profile = inspect_raster(event.source_mask_path)
    source_hash = file_sha256(event.source_mask_path)
    aligned_flood = align_label_raster(event.source_mask_path, grid, 1)
    aligned_perm, permanent_mode = _read_optional_permanent_water(event, grid, profile.count)

    inside = grid.boundary_mask
    flood_inside = aligned_flood[inside]
    perm_inside = aligned_perm[inside]
    valid_inside = (flood_inside != NODATA_CODE) & (perm_inside != NODATA_CODE) & (perm_inside != 1)
    observed_values = np.where(valid_inside, flood_inside, 0).astype("uint8")
    permanent_values = np.where(perm_inside == NODATA_CODE, 0, perm_inside).astype("uint8")

    cells = grid_cells_dataframe(grid)
    frame = cells.loc[cells["in_sindh"], ["grid_cell_id", "row", "col", "latitude", "longitude"]].reset_index(drop=True)
    frame["event_id"] = event.event_id
    frame["event_date"] = event.representative_date
    observed = pd.Series(observed_values, dtype="Int8")
    observed.loc[~valid_inside] = pd.NA
    frame["observed_inundation_label"] = observed
    frame["permanent_water_label"] = pd.Series(permanent_values, dtype="Int8")
    frame["label_valid"] = valid_inside.astype(bool)
    frame["threshold"] = event.selected_threshold
    frame["polarization"] = event.polarization
    frame["orbit_strategy"] = event.orbit_strategy
    frame["orbit_pass"] = event.orbit_pass
    frame["flood_area_km2"] = event.flood_area_estimate_km2
    frame["source_mask_path"] = str(event.source_mask_path)
    frame["source_mask_sha256"] = source_hash

    if frame.duplicated(["event_id", "grid_cell_id"]).any():
        raise ValueError(f"Duplicate grid_cell_id/event_id label rows generated for {event.event_id}.")
    label_hash = label_array_sha256(observed_values, permanent_values, valid_inside.astype("uint8"))
    summary = {
        "event_id": event.event_id,
        "status": "processed",
        "source_mask_path": str(event.source_mask_path),
        "source_mask_sha256": source_hash,
        "label_array_sha256": label_hash,
        "raster_profile": profile.__dict__,
        "permanent_water_mode": permanent_mode,
        "rows": int(len(frame)),
        "valid_rows": int(frame["label_valid"].sum()),
        "flood_cells": int((frame["observed_inundation_label"] == 1).sum()),
        "non_flood_cells": int((frame["observed_inundation_label"] == 0).sum()),
        "permanent_water_cells": int((frame["permanent_water_label"] == 1).sum()),
        "nodata_cells": int((~frame["label_valid"] & (frame["permanent_water_label"] != 1)).sum()),
        "threshold": event.selected_threshold,
        "threshold_method": event.threshold_method,
        "threshold_alternatives_tested": event.threshold_alternatives_tested,
        "independent_event": event.independent_event,
        "parent_event_id": event.parent_event_id,
    }
    write_dataframe_partition(frame, labels_path)
    return frame, summary


def write_event_metadata(event: SentinelEventConfig, summary: dict[str, object], metadata_path: Path, validation_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": event.raw,
        "ingestion": summary,
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n")
    validation_path.write_text(json.dumps(summary, indent=2) + "\n")


def valid_existing_event_outputs(labels_path: Path, metadata_path: Path, validation_path: Path, event_id: str) -> bool:
    if not labels_path.exists() or not metadata_path.exists() or not validation_path.exists():
        return False
    try:
        labels = pd.read_parquet(labels_path)
        validation = json.loads(validation_path.read_text())
    except Exception:
        return False
    required = {"event_id", "grid_cell_id", "observed_inundation_label", "permanent_water_label", "label_valid"}
    if required.difference(labels.columns):
        return False
    if labels.empty or set(labels["event_id"].unique()) != {event_id}:
        return False
    if labels.duplicated(["event_id", "grid_cell_id"]).any():
        return False
    return validation.get("event_id") == event_id and validation.get("status") in {"processed", "skipped_existing"}
