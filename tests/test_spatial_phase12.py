from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from shapely.geometry import box

from src.spatial.alignment import (
    build_glofas_river_mapping,
    flatten_inside,
    resampling_for,
    write_dataframe_partition,
)
from src.spatial.configuration import SpatialPipelineConfig
from src.spatial.features import build_spatial_features
from src.spatial.grid import grid_from_era5, save_grid_outputs
from src.spatial.labels import Sentinel1Event, build_sentinel1_labels
from src.spatial.temporal import normalise_daily_index, partition_name, validate_no_future_looking_join
from src.spatial.validation import validation_summary, write_reports
from src.spatial.validation import validate_transform_signature


def _write_era5(path: Path) -> None:
    times = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    lats = np.array([1.0, 0.0])
    lons = np.array([10.0, 11.0])
    shape = (len(times), len(lats), len(lons))
    ds = xr.Dataset(
        {
            "tp": (("time", "lat", "lon"), np.ones(shape, dtype="float32")),
            "u10": (("time", "lat", "lon"), np.ones(shape, dtype="float32") * 3),
            "v10": (("time", "lat", "lon"), np.ones(shape, dtype="float32") * 4),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    ds.to_netcdf(path)
    ds.close()


def _write_glofas(path: Path) -> None:
    times = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    lats = np.array([0.875, 0.625, 0.375, 0.125])
    lons = np.array([9.875, 10.125, 10.875, 11.125])
    data = np.zeros((len(times), len(lats), len(lons)), dtype="float32")
    data[:, 1, 1] = np.arange(len(times), dtype="float32") + 10
    data[:, 2, 2] = np.arange(len(times), dtype="float32") + 20
    ds = xr.Dataset({"dis24": (("time", "lat", "lon"), data)}, coords={"time": times, "lat": lats, "lon": lons})
    ds.to_netcdf(path)
    ds.close()


def _write_boundary(path: Path) -> None:
    gpd.GeoDataFrame({"name": ["test"]}, geometry=[box(9.75, -0.25, 11.25, 1.25)], crs="EPSG:4326").to_file(path, driver="GeoJSON")


def _write_raster(path: Path, values: np.ndarray, transform=None, crs="EPSG:4326", dtype="float32", nodata=None) -> None:
    transform = transform or from_origin(9.75, 1.25, 0.5, 0.5)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(values.astype(dtype), 1)


@pytest.fixture()
def synthetic_sources(tmp_path: Path) -> tuple[SpatialPipelineConfig, Path]:
    era5 = tmp_path / "era5.nc"
    glofas = tmp_path / "glofas.nc"
    boundary = tmp_path / "boundary.geojson"
    srtm = tmp_path / "srtm.tif"
    sentinel_dir = tmp_path / "sentinel1"
    sentinel_dir.mkdir()
    _write_era5(era5)
    _write_glofas(glofas)
    _write_boundary(boundary)
    _write_raster(srtm, np.array([[2, 4], [6, 8]], dtype="float32"))
    _write_raster(sentinel_dir / "sentinel1_flood_mask_sindh_2020_event1_threshold_24.tif", np.array([[0, 1], [1, 0]], dtype="uint8"), dtype="uint8")
    config = SpatialPipelineConfig(
        era5_path=era5,
        glofas_path=glofas,
        srtm_path=srtm,
        boundary_path=boundary,
        sentinel1_dir=sentinel_dir,
        output_dir=tmp_path / "spatial",
        grid_metadata_path=tmp_path / "spatial" / "grid_metadata.json",
        grid_cells_path=tmp_path / "spatial" / "grid_cells.csv",
    )
    return config, sentinel_dir


def test_canonical_grid_creation_and_deterministic_ids(synthetic_sources) -> None:
    config, _ = synthetic_sources
    grid = grid_from_era5(config.era5_path, config.boundary_path)
    assert (grid.height, grid.width) == (2, 2)
    assert grid.grid_cell_ids[0, 0] == "sindh_era5_r000_c000"
    assert grid.crs == "EPSG:4326"
    assert grid.boundary_mask.all()
    assert list(grid.latitudes) == [1.0, 0.0]
    assert list(grid.longitudes) == [10.0, 11.0]


def test_boundary_masking_excludes_outside_cells(tmp_path: Path) -> None:
    era5 = tmp_path / "era5.nc"
    boundary = tmp_path / "boundary.geojson"
    _write_era5(era5)
    gpd.GeoDataFrame({"name": ["test"]}, geometry=[box(9.75, 0.75, 10.25, 1.25)], crs="EPSG:4326").to_file(boundary, driver="GeoJSON")
    grid = grid_from_era5(era5, boundary)
    assert grid.boundary_mask.sum() == 1


def test_save_grid_outputs(synthetic_sources) -> None:
    config, _ = synthetic_sources
    grid = grid_from_era5(config.era5_path, config.boundary_path)
    save_grid_outputs(grid, config.grid_metadata_path, config.grid_cells_path)
    metadata = json.loads(config.grid_metadata_path.read_text())
    assert metadata["unit_of_analysis"].startswith("one row")
    assert pd.read_csv(config.grid_cells_path)["grid_cell_id"].is_unique


def test_resampling_mode_selection() -> None:
    assert resampling_for("flood_mask") == Resampling.nearest
    assert resampling_for("terrain_elevation") == Resampling.bilinear
    assert resampling_for("glofas_discharge") == "river-aware-nearest"
    with pytest.raises(ValueError, match="Unsupported"):
        resampling_for("unknown")


def test_temporal_alignment_rejects_duplicates_and_future_joins() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        normalise_daily_index(pd.to_datetime(["2020-01-01", "2020-01-01"]), "test")
    with pytest.raises(ValueError, match="Future-looking"):
        validate_no_future_looking_join(pd.Timestamp("2020-01-01"), [pd.Timestamp("2020-01-02")])


def test_river_cell_handling(synthetic_sources) -> None:
    config, _ = synthetic_sources
    grid = grid_from_era5(config.era5_path, config.boundary_path)
    with xr.open_dataset(config.glofas_path) as ds:
        mapping = build_glofas_river_mapping(ds.lat.values, ds.lon.values, ds.dis24.isel(time=0).values, grid)
    assert mapping.has_river_within_cell.any()
    assert np.isfinite(mapping.distance_km).all()


def test_lightweight_spatial_feature_integration(synthetic_sources, tmp_path: Path) -> None:
    config, _ = synthetic_sources
    grid = grid_from_era5(config.era5_path, config.boundary_path)
    output = tmp_path / "features.parquet"
    df = build_spatial_features(config, grid, "2020-01-08", "2020-01-09", output)
    assert output.exists()
    assert len(df) == grid.inside_cell_count * 2
    assert df.duplicated(["grid_cell_id", "date"]).sum() == 0
    assert "era5_wind_speed_current" in df.columns
    assert np.allclose(df["era5_wind_speed_current"], 5.0)
    assert "glofas_river_discharge_m3s_on_river_cell" in df.columns
    assert df.groupby("grid_cell_id")["terrain_elevation_m"].nunique().max() == 1


def test_nearest_neighbor_label_resampling(synthetic_sources, tmp_path: Path) -> None:
    config, sentinel_dir = synthetic_sources
    grid = grid_from_era5(config.era5_path, config.boundary_path)
    event = Sentinel1Event("event", sentinel_dir / "sentinel1_flood_mask_sindh_2020_event1_threshold_24.tif", -2.4)
    labels = build_sentinel1_labels(event, grid, tmp_path / "labels.parquet")
    assert set(labels["observed_inundation_label"].unique()).issubset({0, 1})
    assert labels["permanent_water_label"].isna().all()


def test_validation_report_generation(synthetic_sources, tmp_path: Path) -> None:
    config, _ = synthetic_sources
    grid = grid_from_era5(config.era5_path, config.boundary_path)
    df = build_spatial_features(config, grid, "2020-01-08", "2020-01-08")
    summary = validation_summary(grid, df)
    assert summary["status"] == "passed"
    json_path, md_path = write_reports(summary, tmp_path / "report.json", tmp_path / "report.md")
    assert json_path.exists()
    assert md_path.exists()


def test_incomplete_grid_rejection(synthetic_sources) -> None:
    config, _ = synthetic_sources
    grid = grid_from_era5(config.era5_path, config.boundary_path)
    df = build_spatial_features(config, grid, "2020-01-08", "2020-01-08").iloc[:-1]
    summary = validation_summary(grid, df)
    assert summary["status"] == "failed"
    assert any("Incomplete grid" in issue for issue in summary["issues"])


def test_chunk_partition_naming_and_deterministic_write(tmp_path: Path) -> None:
    assert partition_name(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"), True) == "year=2020"
    with pytest.raises(ValueError, match="within one calendar year"):
        partition_name(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01"), True)
    df = pd.DataFrame({"a": [1, 2]})
    output = tmp_path / "partition.parquet"
    write_dataframe_partition(df, output)
    first = output.read_bytes()
    write_dataframe_partition(df, output)
    assert output.read_bytes() == first


def test_crs_mismatch_rejection(tmp_path: Path) -> None:
    raster = tmp_path / "bad.tif"
    _write_raster(raster, np.ones((2, 2), dtype="float32"), crs=None)
    from src.spatial.alignment import align_raster_to_grid

    era5 = tmp_path / "era5.nc"
    boundary = tmp_path / "boundary.geojson"
    _write_era5(era5)
    _write_boundary(boundary)
    grid = grid_from_era5(era5, boundary)
    with pytest.raises(ValueError, match="CRS is missing"):
        align_raster_to_grid(str(raster), grid, "terrain_elevation")


def test_flatten_inside_rejects_transform_dimension_mismatch(synthetic_sources) -> None:
    config, _ = synthetic_sources
    grid = grid_from_era5(config.era5_path, config.boundary_path)
    with pytest.raises(ValueError, match="shape"):
        flatten_inside(grid, np.ones((1, 1), dtype="float32"))
    with pytest.raises(ValueError, match="Affine transform"):
        validate_transform_signature(from_origin(0, 0, 1, 1), grid.width, grid.height, grid)
