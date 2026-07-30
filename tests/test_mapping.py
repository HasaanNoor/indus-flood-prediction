from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio

from src.mapping import pipeline as mapping_pipeline
from src.mapping.classification import classify_probabilities, validate_thresholds
from src.mapping.configuration import (
    HorizonConfig,
    RISK_CLASSES,
    get_horizon_config,
    model_path_for_horizon,
    output_dir_for_horizon,
)
from src.mapping.inference import (
    build_predictions_frame,
    build_metadata,
    model_feature_columns,
    run_tabular_inference,
    validate_horizon_binding,
    validate_prediction_features,
)
from src.mapping.pipeline import _guard_material_overwrite
from src.mapping.raster_export import (
    PROBABILITY_NODATA,
    SpatialGrid,
    reconstruct_regular_grid,
    validate_raster_dimensions,
    values_to_grid,
    write_geotiff,
)


class DummyProbabilityModel:
    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names_in_ = np.array(feature_names)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        scores = np.clip(X.sum(axis=1).to_numpy(dtype=float) / 10.0, 0.0, 1.0)
        return np.column_stack([1.0 - scores, scores])


def test_valid_threshold_configuration() -> None:
    assert validate_thresholds((0.2, 0.5, 0.8)) == (0.2, 0.5, 0.8)


@pytest.mark.parametrize("thresholds", [(0.5, 0.5, 0.8), (0.7, 0.5, 0.8)])
def test_rejects_unordered_thresholds(thresholds: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_thresholds(thresholds)


@pytest.mark.parametrize("thresholds", [(-0.1, 0.5, 0.8), (0.1, 0.5, 1.2)])
def test_rejects_thresholds_outside_unit_interval(thresholds: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_thresholds(thresholds)


def test_classification_boundaries_are_left_closed() -> None:
    probabilities = np.array([0.0, 0.249, 0.25, 0.499, 0.5, 0.749, 0.75, 1.0])
    assert classify_probabilities(probabilities, (0.25, 0.5, 0.75)).tolist() == [
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
    ]


def test_rejects_probability_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        classify_probabilities(np.array([0.1, 1.1]), (0.25, 0.5, 0.75))


def test_rejects_missing_model_features() -> None:
    df = pd.DataFrame({"a": [1.0], "date": ["2020-01-01"]})
    with pytest.raises(ValueError, match="Missing model features"):
        validate_prediction_features(df, ["a", "b"])


def test_rejects_unexpected_numeric_features() -> None:
    df = pd.DataFrame({"a": [1.0], "b": [2.0], "date": ["2020-01-01"]})
    with pytest.raises(ValueError, match="Unexpected numeric prediction features"):
        validate_prediction_features(df, ["a"])


def test_stable_feature_ordering() -> None:
    model = DummyProbabilityModel(["b", "a"])
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    X = validate_prediction_features(df, model_feature_columns(model))
    assert list(X.columns) == ["b", "a"]


def test_horizon_to_model_mapping() -> None:
    assert model_path_for_horizon("7day").name == "hydrology_label_discharge_next_7d_ge_q95_xgboost.pkl"


def test_horizon_to_dataset_mapping() -> None:
    config = get_horizon_config("14day")
    assert config.dataset_path.name == "flood_features_hydrology.csv"
    assert config.label_column == "label_discharge_next_14d_ge_q95"
    validate_horizon_binding(config)


def test_horizon_binding_rejects_mismatched_model() -> None:
    config = replace(get_horizon_config("1day"), model_path=Path("hydrology_label_discharge_next_7d_ge_q95_xgboost.pkl"))
    with pytest.raises(ValueError, match="does not match horizon"):
        validate_horizon_binding(config)


def test_regular_grid_validation_and_output_naming(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "latitude": [2.0, 2.0, 1.0, 1.0],
            "longitude": [10.0, 11.0, 10.0, 11.0],
        }
    )
    grid = reconstruct_regular_grid(df, crs="EPSG:4326")
    assert grid is not None
    assert (grid.height, grid.width) == (2, 2)
    assert output_dir_for_horizon("1day", tmp_path) == tmp_path / "1day"


def test_rejects_incomplete_regular_grid() -> None:
    df = pd.DataFrame({"lat": [2.0, 2.0, 1.0], "lon": [10.0, 11.0, 10.0]})
    with pytest.raises(ValueError, match="complete rectangular grid"):
        reconstruct_regular_grid(df, crs="EPSG:4326")


def test_raster_dimension_validation() -> None:
    grid = SpatialGrid("lat", "lon", "EPSG:4326", object(), 2, 2, (1.0, 1.0), PROBABILITY_NODATA)
    with pytest.raises(ValueError, match="dimensions"):
        validate_raster_dimensions(np.zeros((1, 2)), grid)


def test_nodata_preservation_in_geotiff(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "lat": [2.0, 2.0, 1.0, 1.0],
            "lon": [10.0, 11.0, 10.0, 11.0],
        }
    )
    grid = reconstruct_regular_grid(df, crs="EPSG:4326")
    assert grid is not None
    array = values_to_grid(df, np.array([0.1, PROBABILITY_NODATA, 0.3, 0.4], dtype="float32"), grid)
    output = write_geotiff(array, grid, tmp_path / "probability_map.tif", "float32", PROBABILITY_NODATA)
    with rasterio.open(output) as src:
        assert src.crs.to_string() == "EPSG:4326"
        assert src.width == 2
        assert src.height == 2
        assert src.nodata == PROBABILITY_NODATA
        assert src.dtypes == ("float32",)


def test_values_to_grid_uses_full_grid_axes_for_retained_rows() -> None:
    original = pd.DataFrame(
        {
            "lat": [2.0, 2.0, 1.0, 1.0],
            "lon": [10.0, 11.0, 10.0, 11.0],
        }
    )
    retained = original.drop(index=[1]).reset_index(drop=True)
    grid = reconstruct_regular_grid(original, crs="EPSG:4326")
    assert grid is not None

    array = values_to_grid(retained, np.array([0.2, 0.6, 0.8], dtype="float32"), grid)

    np.testing.assert_allclose(array, [[0.2, PROBABILITY_NODATA], [0.6, 0.8]])


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"latitude": [1.0], "lat": [1.0]}),
        pd.DataFrame({"longitude": [10.0], "lon": [10.0]}),
        pd.DataFrame({"date": ["2020-01-01"], "time": ["00:00"]}),
        pd.DataFrame(
            {"date": ["2020-01-01"], "timestamp": ["2020-01-01T00:00:00"]}
        ),
    ],
)
def test_build_predictions_frame_rejects_ambiguous_aliases(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Ambiguous aliases"):
        build_predictions_frame(frame, np.array([0.1]), np.array([1]), "1day")


def test_metadata_generation_records_inputs(tmp_path: Path) -> None:
    model_path = tmp_path / "hydrology_label_discharge_next_1d_ge_q95_xgboost.pkl"
    dataset_path = tmp_path / "flood_features_hydrology.csv"
    model_path.write_bytes(b"model")
    dataset_path.write_text("date,a\n2020-01-01,1\n")
    config = replace(get_horizon_config("1day"), model_path=model_path, dataset_path=dataset_path)
    metadata = build_metadata(config, (0.25, 0.5, 0.75), ["a"], 1, {"crs": None})
    assert metadata["forecast_horizon"] == "1day"
    assert metadata["feature_count"] == 1
    assert metadata["model_sha256"]
    assert RISK_CLASSES[4] == "Very High"


def test_material_metadata_overwrite_guard(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text('{"forecast_horizon": "1day", "risk_thresholds": [0.25, 0.5, 0.75]}')
    with pytest.raises(FileExistsError, match="materially different"):
        _guard_material_overwrite(
            metadata_path,
            {"forecast_horizon": "1day", "risk_thresholds": [0.2, 0.5, 0.8]},
            overwrite=False,
        )
    _guard_material_overwrite(
        metadata_path,
        {"forecast_horizon": "1day", "risk_thresholds": [0.2, 0.5, 0.8]},
        overwrite=True,
    )


def test_deterministic_repeated_inference(tmp_path: Path) -> None:
    model_path = tmp_path / "hydrology_label_discharge_next_1d_ge_q95_xgboost.pkl"
    dataset_path = tmp_path / "flood_features_hydrology.csv"
    with model_path.open("wb") as handle:
        pickle.dump(DummyProbabilityModel(["a", "b"]), handle)
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "a": [1.0, 3.0],
            "b": [2.0, 4.0],
            "label_discharge_next_1d_ge_q95": [0, 1],
        }
    ).to_csv(dataset_path, index=False)
    config = replace(get_horizon_config("1day"), model_path=model_path, dataset_path=dataset_path)
    first = run_tabular_inference(config, (0.25, 0.5, 0.75), {"crs": None})
    second = run_tabular_inference(config, (0.25, 0.5, 0.75), {"crs": None})
    assert first.predictions.equals(second.predictions)
    assert np.array_equal(first.risk_classes, second.risk_classes)


def test_drop_invalid_rows_retains_coordinate_alignment_and_rejects_incomplete_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "hydrology_label_discharge_next_1d_ge_q95_xgboost.pkl"
    dataset_path = tmp_path / "flood_features_hydrology.csv"
    with model_path.open("wb") as handle:
        pickle.dump(DummyProbabilityModel(["a", "b"]), handle)

    original = pd.DataFrame(
        {
            "lat": [2.0, 2.0, 1.0, 1.0],
            "lon": [10.0, 11.0, 10.0, 11.0],
            "a": [1.0, np.nan, 3.0, 4.0],
            "b": [1.0, 2.0, 3.0, 4.0],
            "label_discharge_next_1d_ge_q95": [0, 0, 1, 1],
        }
    )
    original.to_csv(dataset_path, index=False)
    config = HorizonConfig(
        horizon="1day",
        label_column="label_discharge_next_1d_ge_q95",
        model_path=model_path,
        dataset_path=dataset_path,
        dataset_type="hydrology",
    )

    result = run_tabular_inference(
        config,
        (0.25, 0.5, 0.75),
        {"crs": None},
        drop_invalid_rows=True,
    )

    assert result.metadata["dropped_non_finite_rows"] == 1
    assert result.retained_rows[["lat", "lon"]].to_records(index=False).tolist() == [
        (2.0, 10.0),
        (1.0, 10.0),
        (1.0, 11.0),
    ]
    assert result.predictions[["latitude", "longitude"]].to_records(index=False).tolist() == [
        (2.0, 10.0),
        (1.0, 10.0),
        (1.0, 11.0),
    ]
    full_grid = reconstruct_regular_grid(original, crs="EPSG:4326")
    assert full_grid is not None
    probability_grid = values_to_grid(
        result.retained_rows,
        result.probabilities.astype("float32"),
        full_grid,
    )
    np.testing.assert_allclose(probability_grid, [[0.2, PROBABILITY_NODATA], [0.6, 0.8]])

    monkeypatch.setattr(
        mapping_pipeline,
        "get_horizon_config",
        lambda horizon, dataset_type="hydrology": config,
    )
    with pytest.raises(ValueError, match="complete rectangular grid"):
        mapping_pipeline.run_horizon(
            horizon="1day",
            thresholds=(0.25, 0.5, 0.75),
            output_root=tmp_path / "outputs",
            raster_crs="EPSG:4326",
            boundary_path=None,
            drop_invalid_rows=True,
            overwrite=True,
        )
