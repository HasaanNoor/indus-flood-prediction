from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.spatial.alignment import align_raster_to_grid, flatten_inside, write_dataframe_partition
from src.spatial.grid import CanonicalGrid, grid_cells_dataframe


@dataclass(frozen=True)
class Sentinel1Event:
    event_id: str
    path: Path
    threshold_db: float | None
    observed_date: str | None = None


def discover_sentinel1_events(directory: Path) -> list[Sentinel1Event]:
    events = []
    for path in sorted(directory.glob("*.tif")):
        match = re.search(r"(20\d{2}).*?event(\d+).*?threshold_?(\d+)", path.name)
        if match:
            event_id = f"{match.group(1)}_event{match.group(2)}_threshold_{match.group(3)}"
            threshold = -float(match.group(3)) / 10.0
        else:
            event_id = path.stem
            threshold = None
        if "Version" in path.name:
            event_id = f"{event_id}_versioned"
        events.append(Sentinel1Event(event_id=event_id, path=path, threshold_db=threshold))
    return events


def build_sentinel1_labels(event: Sentinel1Event, grid: CanonicalGrid, output_path: Path | None = None) -> pd.DataFrame:
    cells = grid_cells_dataframe(grid)
    frame = cells.loc[cells["in_sindh"], ["grid_cell_id", "row", "col", "latitude", "longitude"]].reset_index(drop=True)
    mask = align_raster_to_grid(str(event.path), grid, "flood_mask", dst_dtype="float32")
    observed = flatten_inside(grid, mask)
    frame["event_id"] = event.event_id
    frame["observed_inundation_label"] = (observed >= 0.5).astype("int8")
    frame["permanent_water_label"] = pd.NA
    frame["model_estimated_flood_probability"] = pd.NA
    frame["sentinel1_threshold_db"] = event.threshold_db
    frame["source_raster"] = str(event.path)
    frame["label_limitations"] = "Sentinel-1 threshold mask; permanent water not supplied as a separate raster in this repository."
    if output_path is not None:
        write_dataframe_partition(frame, output_path)
    return frame


def label_availability_report(events: list[Sentinel1Event]) -> dict[str, object]:
    unique_events = {event.event_id.replace("_versioned", "") for event in events}
    return {
        "raster_count": len(events),
        "unique_event_count": len(unique_events),
        "events": [
            {"event_id": event.event_id, "path": str(event.path), "threshold_db": event.threshold_db}
            for event in events
        ],
        "training_readiness": (
            "insufficient_for_robust_spatial_model_training"
            if len(unique_events) < 3
            else "candidate_multi_event_label_set"
        ),
        "limitations": [
            "Sentinel-1 masks are observed inundation candidates, not perfect ground truth.",
            "Permanent water must remain separate when available and should not be treated as flood.",
            "A single event cannot support a general spatial flood-event classifier.",
        ],
    }
