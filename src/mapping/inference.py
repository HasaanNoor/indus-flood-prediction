from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from src.mapping.classification import classify_probabilities, labels_for_classes, validate_probabilities
from src.mapping.configuration import HorizonConfig


METADATA_COLUMNS = {"date", "time", "timestamp", "lat", "latitude", "lon", "longitude", "x", "y"}
PROBABILITY_COLUMN = "flood_probability"
RISK_CLASS_COLUMN = "risk_class"
RISK_LABEL_COLUMN = "risk_label"
HORIZON_COLUMN = "forecast_horizon"


@dataclass(frozen=True)
class InferenceResult:
    predictions: pd.DataFrame
    probabilities: np.ndarray
    risk_classes: np.ndarray
    feature_columns: list[str]
    metadata: dict[str, object]


def load_model(model_path: Path) -> object:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def load_prediction_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Prediction dataset not found: {dataset_path}")
    df = pd.read_csv(dataset_path, parse_dates=["date"] if _has_date_column(dataset_path) else None)
    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Dataset contains duplicated columns: {duplicated}")
    return df


def _has_date_column(dataset_path: Path) -> bool:
    header = pd.read_csv(dataset_path, nrows=0)
    return "date" in header.columns


def model_feature_columns(model: object) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(name) for name in names]
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                return [str(name) for name in names]
    raise ValueError("Model artifact does not expose feature_names_in_; cannot reconstruct feature order.")


def validate_horizon_binding(config: HorizonConfig) -> None:
    expected = config.label_column
    model_name = config.model_path.name
    dataset_name = config.dataset_path.name
    if expected not in model_name:
        raise ValueError(f"Model artifact {model_name} does not match horizon label {expected}.")
    if config.dataset_type not in model_name:
        raise ValueError(f"Model artifact {model_name} does not match dataset type {config.dataset_type}.")
    if config.dataset_type == "hydrology" and "hydrology" not in dataset_name:
        raise ValueError(f"Dataset {dataset_name} does not match hydrology mapping.")
    if config.dataset_type == "rainfall_only" and "rainfall_only" not in dataset_name:
        raise ValueError(f"Dataset {dataset_name} does not match rainfall-only mapping.")


def validate_prediction_features(df: pd.DataFrame, required_features: list[str]) -> pd.DataFrame:
    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Dataset contains duplicated columns: {duplicated}")

    missing = [column for column in required_features if column not in df.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    allowed_non_features = {
        column
        for column in df.columns
        if column in METADATA_COLUMNS
        or column.startswith("label_")
        or column.startswith("target_")
    }
    numeric_columns = set(df.select_dtypes(include="number").columns)
    unexpected = sorted(numeric_columns.difference(required_features).difference(allowed_non_features))
    if unexpected:
        raise ValueError(f"Unexpected numeric prediction features not used by model: {unexpected}")

    X = df.loc[:, required_features]
    non_numeric = [column for column in X.columns if not pd.api.types.is_numeric_dtype(X[column])]
    if non_numeric:
        raise ValueError(f"Model features must be numeric: {non_numeric}")
    if not np.isfinite(X.to_numpy(dtype="float64")).all():
        raise ValueError("Prediction features contain missing or non-finite values.")
    return X


def predict_positive_class(model: object, X: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba.")
    probabilities = np.asarray(model.predict_proba(X)[:, 1], dtype="float64")
    return validate_probabilities(probabilities)


def checksum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_predictions_frame(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    risk_classes: np.ndarray,
    horizon: str,
) -> pd.DataFrame:
    columns = [
        column
        for column in ["latitude", "lat", "longitude", "lon", "date", "time", "timestamp"]
        if column in df.columns
    ]
    output = df.loc[:, columns].copy()
    rename = {"lat": "latitude", "lon": "longitude", "time": "timestamp", "date": "timestamp"}
    output = output.rename(columns=rename)
    output[PROBABILITY_COLUMN] = probabilities
    output[RISK_CLASS_COLUMN] = risk_classes
    output[RISK_LABEL_COLUMN] = labels_for_classes(risk_classes)
    output[HORIZON_COLUMN] = horizon
    return output


def build_metadata(
    config: HorizonConfig,
    thresholds: tuple[float, float, float],
    feature_columns: list[str],
    row_count: int,
    spatial_metadata: dict[str, object],
    dropped_non_finite_rows: int = 0,
) -> dict[str, object]:
    return {
        "forecast_horizon": config.horizon,
        "dataset_type": config.dataset_type,
        "model_name": config.model_name,
        "model_artifact": str(config.model_path),
        "input_dataset": str(config.dataset_path),
        "model_sha256": checksum(config.model_path),
        "input_dataset_sha256": checksum(config.dataset_path),
        "risk_thresholds": list(thresholds),
        "risk_class_boundary_behavior": (
            "Low: p < t1; Moderate: t1 <= p < t2; "
            "High: t2 <= p < t3; Very High: p >= t3"
        ),
        "feature_count": len(feature_columns),
        "prediction_row_count": row_count,
        "dropped_non_finite_rows": dropped_non_finite_rows,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        **spatial_metadata,
    }


def atomic_write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=output_path.parent, suffix=".tmp") as handle:
        temp_path = Path(handle.name)
        df.to_csv(handle, index=False)
    temp_path.replace(output_path)


def atomic_write_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=output_path.parent, suffix=".tmp") as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temp_path.replace(output_path)


def run_tabular_inference(
    config: HorizonConfig,
    thresholds: tuple[float, float, float],
    spatial_metadata: dict[str, object],
    drop_invalid_rows: bool = False,
) -> InferenceResult:
    validate_horizon_binding(config)
    model = load_model(config.model_path)
    df = load_prediction_dataset(config.dataset_path)
    features = model_feature_columns(model)
    dropped = 0
    if drop_invalid_rows:
        missing = [column for column in features if column not in df.columns]
        if missing:
            raise ValueError(f"Missing model features: {missing}")
        feature_values = df.loc[:, features].to_numpy(dtype="float64")
        valid_rows = np.isfinite(feature_values).all(axis=1)
        dropped = int((~valid_rows).sum())
        df = df.loc[valid_rows].reset_index(drop=True)
    X = validate_prediction_features(df, features)
    probabilities = predict_positive_class(model, X)
    risk_classes = classify_probabilities(probabilities, thresholds)
    predictions = build_predictions_frame(df, probabilities, risk_classes, config.horizon)
    metadata = build_metadata(
        config=config,
        thresholds=thresholds,
        feature_columns=features,
        row_count=len(predictions),
        spatial_metadata=spatial_metadata,
        dropped_non_finite_rows=dropped,
    )
    return InferenceResult(predictions, probabilities, risk_classes, features, metadata)
