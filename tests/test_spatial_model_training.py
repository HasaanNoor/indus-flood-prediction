from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.spatial import model_training as mt


def _spatial_training_files(tmp_path: Path) -> tuple[Path, Path]:
    rows = []
    labels = []
    for row in range(6):
        for col in range(4):
            grid_cell_id = f"cell_{row}_{col}"
            flooded = int((row in {0, 1, 4, 5}) and col == 0)
            rows.append(
                {
                    "grid_cell_id": grid_cell_id,
                    "date": "2020-01-10",
                    "row": row,
                    "col": col,
                    "latitude": 30.0 - row,
                    "longitude": 65.0 + col,
                    "in_sindh": True,
                    "terrain_elevation_m": float(row + col),
                    "terrain_slope_degrees": float(row + 1),
                    "relative_elevation_within_sindh_m": float(row - 3),
                    "era5_tp_current": float(flooded + row / 10.0),
                    "glofas_nearest_river_discharge_m3s_current": float(flooded + col / 10.0),
                    "glofas_river_discharge_m3s_on_river_cell": np.nan if col else float(flooded),
                    "has_glofas_river_cell": int(col == 0),
                }
            )
            labels.append(
                {
                    "grid_cell_id": grid_cell_id,
                    "row": row,
                    "col": col,
                    "latitude": 30.0 - row,
                    "longitude": 65.0 + col,
                    "event_id": "2020_event1_threshold_24",
                    "observed_inundation_label": flooded,
                }
            )
    feature_path = tmp_path / "features.parquet"
    label_path = tmp_path / "labels.parquet"
    pd.DataFrame(rows).to_parquet(feature_path, index=False)
    pd.DataFrame(labels).to_parquet(label_path, index=False)
    return feature_path, label_path


def test_spatial_feature_ordering_excludes_metadata(tmp_path: Path) -> None:
    feature_path, label_path = _spatial_training_files(tmp_path)
    dataset = mt.build_spatial_training_dataset(feature_path, label_path, "2020-01-10")
    assert "row" not in dataset.feature_columns
    assert "latitude" not in dataset.feature_columns
    assert dataset.feature_columns == [
        "terrain_elevation_m",
        "terrain_slope_degrees",
        "relative_elevation_within_sindh_m",
        "era5_tp_current",
        "glofas_nearest_river_discharge_m3s_current",
        "glofas_river_discharge_m3s_on_river_cell",
        "has_glofas_river_cell",
    ]


def test_label_alignment_rejects_coordinate_mismatch(tmp_path: Path) -> None:
    feature_path, label_path = _spatial_training_files(tmp_path)
    labels = pd.read_parquet(label_path)
    labels.loc[0, "latitude"] = -99.0
    labels.to_parquet(label_path, index=False)
    with pytest.raises(ValueError, match="coordinate alignment"):
        mt.build_spatial_training_dataset(feature_path, label_path, "2020-01-10")


def test_spatial_block_split_prevents_grid_cell_leakage(tmp_path: Path) -> None:
    feature_path, label_path = _spatial_training_files(tmp_path)
    dataset = mt.build_spatial_training_dataset(feature_path, label_path, "2020-01-10")
    split = mt.make_spatial_block_split(dataset.frame)
    assert set(split.train["grid_cell_id"]).isdisjoint(split.test["grid_cell_id"])
    assert split.train["observed_inundation_label"].nunique() == 2
    assert split.test["observed_inundation_label"].nunique() == 2


def test_deterministic_training_and_prediction_dimensions(tmp_path: Path) -> None:
    feature_path, label_path = _spatial_training_files(tmp_path)
    dataset = mt.build_spatial_training_dataset(feature_path, label_path, "2020-01-10")
    split = mt.make_spatial_block_split(dataset.frame)
    X_train = split.train[dataset.feature_columns]
    y_train = split.train["observed_inundation_label"]
    X_test = split.test[dataset.feature_columns]

    first = mt.build_spatial_model_candidates(y_train, random_state=7)
    second = mt.build_spatial_model_candidates(y_train, random_state=7)
    for name in first:
        first[name].fit(X_train, y_train)
        second[name].fit(X_train, y_train)
        p1 = first[name].predict_proba(X_test)[:, 1]
        p2 = second[name].predict_proba(X_test)[:, 1]
        assert p1.shape == (len(X_test),)
        np.testing.assert_allclose(p1, p2)


def test_model_serialization_preserves_predictions(tmp_path: Path) -> None:
    feature_path, label_path = _spatial_training_files(tmp_path)
    dataset = mt.build_spatial_training_dataset(feature_path, label_path, "2020-01-10")
    split = mt.make_spatial_block_split(dataset.frame)
    X_train = split.train[dataset.feature_columns]
    y_train = split.train["observed_inundation_label"]
    X_test = split.test[dataset.feature_columns]
    model = mt.build_spatial_model_candidates(y_train, random_state=11)["random_forest"].fit(X_train, y_train)
    model_path = tmp_path / "random_forest.pkl"
    mt._save_model(model, model_path)
    with model_path.open("rb") as handle:
        loaded = pickle.load(handle)
    np.testing.assert_allclose(model.predict_proba(X_test), loaded.predict_proba(X_test))


def test_train_spatial_models_writes_requested_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    feature_path, label_path = _spatial_training_files(tmp_path)
    monkeypatch.setattr(mt, "SPATIAL_MODELS_DIR", tmp_path / "spatial_models")
    monkeypatch.setattr(mt, "METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(mt, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(mt, "SPATIAL_TRAINING_SUMMARY_PATH", tmp_path / "metrics" / "spatial_training_summary.json")
    monkeypatch.setattr(mt, "SPATIAL_PREDICTIONS_PATH", tmp_path / "metrics" / "spatial_test_predictions.csv")
    metrics = mt.train_spatial_models(feature_path, label_path, "2020-01-10", random_state=3, generate_shap=False)
    assert metrics["model"].tolist() == ["logistic_regression", "random_forest", "xgboost"]
    assert (tmp_path / "spatial_models" / "logistic_regression.pkl").exists()
    assert (tmp_path / "spatial_models" / "random_forest.pkl").exists()
    assert (tmp_path / "spatial_models" / "xgboost.pkl").exists()
    assert (tmp_path / "metrics" / "spatial_model_metrics.csv").exists()
    assert (tmp_path / "figures" / "spatial_confusion_matrix.png").exists()


def test_spatial_shap_generation_where_practical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("shap")
    feature_path, label_path = _spatial_training_files(tmp_path)
    dataset = mt.build_spatial_training_dataset(feature_path, label_path, "2020-01-10")
    split = mt.make_spatial_block_split(dataset.frame)
    X_train = split.train[dataset.feature_columns]
    y_train = split.train["observed_inundation_label"]
    X_test = split.test[dataset.feature_columns]
    model = mt.build_spatial_model_candidates(y_train, random_state=13)["xgboost"].fit(X_train, y_train)
    monkeypatch.setattr(mt, "SPATIAL_SHAP_IMPORTANCE_PATH", tmp_path / "spatial_shap_feature_importance.csv")
    importance = mt.generate_spatial_shap_outputs(model, X_test, tmp_path, random_state=13)
    assert not importance.empty
    assert (tmp_path / "spatial_shap_summary.png").exists()
    assert (tmp_path / "spatial_shap_bar.png").exists()
