from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.transform import Affine, from_origin
from shapely.geometry import box

from src.spatial.configuration import SpatialPipelineConfig
from src.spatial.grid import grid_from_era5
from src.spatial.sentinel_ingestion import ingest_event_labels
from src.spatial.sentinel_inventory import load_event_inventory
from src.spatial.sentinel_pipeline import run_sentinel_pipeline
from src.spatial.sentinel_validation import build_inventory_report


def _write_era5(path: Path) -> None:
    times = pd.date_range("2020-01-01", "2020-01-03", freq="D")
    ds = xr.Dataset(
        {"tp": (("time", "lat", "lon"), np.ones((3, 2, 2), dtype="float32"))},
        coords={"time": times, "lat": np.array([1.0, 0.0]), "lon": np.array([10.0, 11.0])},
    )
    ds.to_netcdf(path)
    ds.close()


def _write_boundary(path: Path) -> None:
    gpd.GeoDataFrame({"name": ["test"]}, geometry=[box(9.75, -0.25, 11.25, 1.25)], crs="EPSG:4326").to_file(path, driver="GeoJSON")


def _write_raster(
    path: Path,
    values: np.ndarray,
    transform=None,
    crs="EPSG:4326",
    nodata=None,
    dtype="uint8",
) -> None:
    transform = transform or from_origin(9.5, 1.5, 0.5, 0.5)
    count = 1 if values.ndim == 2 else values.shape[0]
    height = values.shape[-2]
    width = values.shape[-1]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        if count == 1:
            dst.write(values.astype(dtype), 1)
        else:
            dst.write(values.astype(dtype))


def _inventory(path: Path, mask_path: Path | None, extra_events: list[dict] | None = None) -> None:
    event = {
        "event_id": "2020_event_a",
        "event_name": "Synthetic 2020 event",
        "event_year": 2020,
        "event_start_date": "2020-01-02",
        "event_end_date": "2020-01-03",
        "representative_date": "2020-01-03",
        "baseline_start_date": "2019-12-01",
        "baseline_end_date": "2019-12-31",
        "data_source": "synthetic",
        "sentinel1_collection": "COPERNICUS/S1_GRD",
        "polarization": "VH",
        "orbit_pass": "DESCENDING",
        "orbit_strategy": "same_pass_pre_and_during_required",
        "threshold_method": "synthetic",
        "selected_threshold": -3.0,
        "threshold_alternatives_tested": [-2.4, -3.0],
        "flood_area_estimate_km2": 1.0,
        "permanent_water_mask_source": "synthetic",
        "permanent_water_occurrence_threshold": 90,
        "slope_threshold_degrees": 5,
        "connected_pixel_threshold": 8,
        "source_mask_path": None if mask_path is None else str(mask_path),
        "permanent_water_mask_path": None,
        "sentinel1_available": True,
        "mask_available": mask_path is not None,
        "validation_status": "candidate_export_available" if mask_path else "pending_gee_export",
        "independent_event": True,
        "parent_event_id": None,
        "notes": "test",
        "source_references": ["test"],
    }
    payload = {"schema_version": "test", "events": [event] + (extra_events or [])}
    path.write_text(json.dumps(payload, indent=2))


@pytest.fixture()
def phase14_sources(tmp_path: Path):
    era5 = tmp_path / "era5.nc"
    boundary = tmp_path / "boundary.geojson"
    _write_era5(era5)
    _write_boundary(boundary)
    grid = grid_from_era5(era5, boundary)
    config = SpatialPipelineConfig(
        era5_path=era5,
        boundary_path=boundary,
        glofas_path=tmp_path / "unused_glofas.nc",
        srtm_path=tmp_path / "unused_srtm.tif",
        grid_metadata_path=tmp_path / "grid_metadata.json",
        grid_cells_path=tmp_path / "grid_cells.csv",
    )
    return tmp_path, grid, config


def test_event_inventory_parsing_and_duplicate_rejection(phase14_sources) -> None:
    tmp_path, _, _ = phase14_sources
    inventory = tmp_path / "inventory.json"
    _inventory(inventory, None)
    assert load_event_inventory(inventory, base_dir=Path("/"))[0].event_id == "2020_event_a"
    payload = json.loads(inventory.read_text())
    payload["events"].append(dict(payload["events"][0]))
    inventory.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Duplicate Sentinel event IDs"):
        load_event_inventory(inventory)


def test_threshold_variant_not_independent(phase14_sources) -> None:
    tmp_path, _, _ = phase14_sources
    inventory = tmp_path / "inventory.json"
    variant = {
        **json.loads(json.dumps({
            "event_id": "2020_event_a_threshold_variant",
            "event_name": "Synthetic variant",
            "event_year": 2020,
            "event_start_date": "2020-01-02",
            "event_end_date": "2020-01-03",
            "representative_date": "2020-01-03",
            "baseline_start_date": "2019-12-01",
            "baseline_end_date": "2019-12-31",
            "data_source": "synthetic",
            "sentinel1_collection": "COPERNICUS/S1_GRD",
            "polarization": "VH",
            "orbit_pass": "DESCENDING",
            "orbit_strategy": "same_pass_pre_and_during_required",
            "threshold_method": "synthetic",
            "selected_threshold": -2.4,
            "threshold_alternatives_tested": [-2.4, -3.0],
            "flood_area_estimate_km2": 1.5,
            "permanent_water_mask_source": "synthetic",
            "permanent_water_occurrence_threshold": 90,
            "slope_threshold_degrees": 5,
            "connected_pixel_threshold": 8,
            "source_mask_path": None,
            "permanent_water_mask_path": None,
            "sentinel1_available": True,
            "mask_available": False,
            "validation_status": "threshold_variant_not_independent",
            "independent_event": False,
            "parent_event_id": "2020_event_a",
            "notes": "test",
            "source_references": ["test"],
        }))
    }
    _inventory(inventory, None, [variant])
    events = load_event_inventory(inventory)
    report = build_inventory_report(events, [], [])
    assert report["independent_event_count"] == 1
    assert "2020_event_a" in report["threshold_variant_groups"]


def test_multiband_ingestion_preserves_nodata_and_permanent_water(phase14_sources) -> None:
    tmp_path, grid, _ = phase14_sources
    mask = tmp_path / "mask.tif"
    flood = np.array([[1, 0], [255, 1]], dtype="uint8")
    perm = np.array([[0, 1], [0, 0]], dtype="uint8")
    _write_raster(mask, np.stack([flood, perm]), transform=grid.transform, nodata=255)
    inventory = tmp_path / "inventory.json"
    _inventory(inventory, mask)
    event = load_event_inventory(inventory)[0]
    labels, summary = ingest_event_labels(event, grid, tmp_path / "labels.parquet")
    assert summary["rows"] == 4
    assert summary["permanent_water_cells"] == 1
    assert summary["nodata_cells"] == 1
    assert labels["observed_inundation_label"].isna().sum() == 2
    assert labels["grid_cell_id"].tolist() == ["sindh_era5_r000_c000", "sindh_era5_r000_c001", "sindh_era5_r001_c000", "sindh_era5_r001_c001"]


def test_crs_and_transform_rejections(phase14_sources) -> None:
    tmp_path, grid, _ = phase14_sources
    bad_crs = tmp_path / "bad_crs.tif"
    _write_raster(bad_crs, np.zeros((4, 4), dtype="uint8"), crs="EPSG:3857")
    inventory = tmp_path / "inventory.json"
    _inventory(inventory, bad_crs)
    with pytest.raises(ValueError, match="differs from canonical grid CRS"):
        ingest_event_labels(load_event_inventory(inventory)[0], grid, tmp_path / "labels.parquet")

    rotated = tmp_path / "rotated.tif"
    _write_raster(rotated, np.zeros((4, 4), dtype="uint8"), transform=Affine(0.25, 0.01, 9.75, 0, -0.25, 1.25))
    _inventory(inventory, rotated)
    with pytest.raises(ValueError, match="rotation/shear"):
        ingest_event_labels(load_event_inventory(inventory)[0], grid, tmp_path / "labels.parquet")


def test_duplicate_hash_detection_and_partial_failure_reporting(phase14_sources) -> None:
    tmp_path, _, _ = phase14_sources
    inventory = tmp_path / "inventory.json"
    _inventory(inventory, None)
    events = load_event_inventory(inventory)
    summaries = [
        {"event_id": "a", "source_mask_sha256": "same", "label_array_sha256": "labels", "rows": 1},
        {"event_id": "b", "source_mask_sha256": "same", "label_array_sha256": "labels", "rows": 1},
    ]
    report = build_inventory_report(events, summaries, [{"event_id": "c", "status": "failed"}])
    assert report["status"] == "partial"
    assert report["duplicate_source_hashes"] == ["same"]
    assert report["duplicate_label_array_hashes"] == ["labels"]


def test_restart_safe_skip_and_deterministic_repeated_execution(phase14_sources, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path, _, config = phase14_sources
    mask = tmp_path / "mask.tif"
    _write_raster(mask, np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype="uint8"))
    inventory = tmp_path / "inventory.json"
    _inventory(inventory, mask)
    monkeypatch.setattr("src.spatial.sentinel_pipeline.EVENTS_DIR", tmp_path / "events")
    monkeypatch.setattr("src.spatial.sentinel_pipeline.COMBINED_DIR", tmp_path / "combined")
    monkeypatch.setattr("src.spatial.sentinel_pipeline.VALIDATION_DIR", tmp_path / "validation")
    first = run_sentinel_pipeline(inventory, all_events=True, config=config, overwrite=True)
    labels_path = tmp_path / "events" / "2020_event_a" / "labels.parquet"
    first_bytes = labels_path.read_bytes()
    second = run_sentinel_pipeline(inventory, all_events=True, config=config, overwrite=False)
    assert first["processed_event_count"] == 1
    assert second["processed_event_count"] == 1
    assert labels_path.read_bytes() == first_bytes
    assert (tmp_path / "validation" / "sentinel_label_inventory_report.md").exists()
