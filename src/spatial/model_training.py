from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.paths import FIGURES_DIR, METRICS_DIR, OUTPUTS, PROJECT_ROOT
from src.spatial.configuration import SPATIAL_FEATURES_DIR, SPATIAL_LABELS_DIR

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


RANDOM_STATE = 42
TARGET_COLUMN = "observed_inundation_label"
DEFAULT_EVENT_DATE = "2019-08-15"
DEFAULT_FEATURE_PATH = SPATIAL_FEATURES_DIR / "spatial_features_2019-08-01_2019-08-15.parquet"
DEFAULT_LABEL_PATH = SPATIAL_LABELS_DIR / "sentinel1_labels_2019_event1_threshold_24.parquet"
SPATIAL_MODELS_DIR = OUTPUTS / "spatial_models"
SPATIAL_TRAINING_SUMMARY_PATH = METRICS_DIR / "spatial_training_summary.json"
SPATIAL_PREDICTIONS_PATH = METRICS_DIR / "spatial_test_predictions.csv"
SPATIAL_SHAP_IMPORTANCE_PATH = METRICS_DIR / "spatial_shap_feature_importance.csv"

METADATA_COLUMNS = {
    "grid_cell_id",
    "date",
    "row",
    "col",
    "latitude",
    "longitude",
    "in_sindh",
    "source_era5_date",
    "source_glofas_date",
    "event_id",
    "permanent_water_label",
    "model_estimated_flood_probability",
    "sentinel1_threshold_db",
    "source_raster",
    "label_limitations",
    TARGET_COLUMN,
}
STRUCTURAL_MISSING_FEATURES = {"glofas_river_discharge_m3s_on_river_cell"}

SPATIAL_MODEL_HYPERPARAMETERS: dict[str, dict[str, object]] = {
    "logistic_regression": {
        "class_weight": "balanced",
        "max_iter": 5000,
        "random_state": RANDOM_STATE,
    },
    "random_forest": {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
    },
    "xgboost": {
        "n_estimators": 150,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
        "tree_method": "hist",
    },
}


@dataclass(frozen=True)
class SpatialTrainingDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    summary: dict[str, object]


@dataclass(frozen=True)
class SpatialSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    strategy: str
    summary: dict[str, object]


def _positive_class_weight(y_train: pd.Series) -> float:
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    return 1.0 if positives == 0 else negatives / positives


def build_spatial_model_candidates(
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> OrderedDict[str, object]:
    params = {
        name: {**values, "random_state": random_state if "random_state" in values else values.get("random_state")}
        for name, values in SPATIAL_MODEL_HYPERPARAMETERS.items()
    }
    models: OrderedDict[str, object] = OrderedDict()
    models["logistic_regression"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**params["logistic_regression"])),
        ]
    )
    models["random_forest"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(**params["random_forest"])),
        ]
    )
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover - environment-dependent optional dependency.
        raise RuntimeError(f"XGBoost is required for Phase 13 spatial training: {exc}") from exc
    xgb_params = dict(params["xgboost"])
    xgb_params["scale_pos_weight"] = _positive_class_weight(y_train)
    models["xgboost"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(**xgb_params)),
        ]
    )
    return models


def select_spatial_feature_columns(df: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS
        and not column.startswith("label_")
        and not column.startswith("target_")
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not columns:
        raise ValueError("No numeric spatial feature columns were found.")
    return columns


def _read_feature_frame(path: Path, event_date: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Spatial feature grid not found: {path}")
    df = pd.read_parquet(path)
    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Spatial feature grid contains duplicated columns: {duplicated}")
    required = {"grid_cell_id", "date", "row", "col", "latitude", "longitude"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Spatial feature grid is missing required columns: {sorted(missing)}")
    if df.duplicated(["grid_cell_id", "date"]).any():
        raise ValueError("Spatial feature grid contains duplicate grid_cell_id/date rows.")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    selected = df.loc[df["date"] == event_date].copy()
    if selected.empty:
        raise ValueError(f"No spatial feature rows found for event_date={event_date}.")
    return selected.sort_values(["row", "col"]).reset_index(drop=True)


def _read_label_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Spatial label dataset not found: {path}")
    labels = pd.read_parquet(path)
    required = {"grid_cell_id", "row", "col", "latitude", "longitude", "event_id", TARGET_COLUMN}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"Spatial labels are missing required columns: {sorted(missing)}")
    if labels.duplicated(["grid_cell_id", "event_id"]).any():
        raise ValueError("Spatial labels contain duplicate grid_cell_id/event_id rows.")
    if not set(labels[TARGET_COLUMN].dropna().unique()).issubset({0, 1}):
        raise ValueError("Spatial inundation labels must be binary 0/1.")
    if labels["event_id"].nunique() != 1:
        raise ValueError("Use one de-duplicated Sentinel-1 event label file for this training run.")
    return labels.sort_values(["row", "col"]).reset_index(drop=True)


def build_spatial_training_dataset(
    feature_path: Path = DEFAULT_FEATURE_PATH,
    label_path: Path = DEFAULT_LABEL_PATH,
    event_date: str = DEFAULT_EVENT_DATE,
) -> SpatialTrainingDataset:
    features = _read_feature_frame(feature_path, event_date)
    labels = _read_label_frame(label_path)
    merged = features.merge(
        labels.drop(columns=["row", "col", "latitude", "longitude"], errors="ignore"),
        on="grid_cell_id",
        how="inner",
        validate="one_to_one",
    )
    missing_features = sorted(set(labels["grid_cell_id"]).difference(features["grid_cell_id"]))
    missing_labels = sorted(set(features["grid_cell_id"]).difference(labels["grid_cell_id"]))
    if missing_features or missing_labels:
        raise ValueError(
            "Feature/label alignment is incomplete: "
            f"missing_features={len(missing_features)}, missing_labels={len(missing_labels)}"
        )

    label_coords = labels.set_index("grid_cell_id")[["latitude", "longitude"]]
    merged_coords = merged.set_index("grid_cell_id")[["latitude", "longitude"]]
    if not np.allclose(
        merged_coords.loc[label_coords.index].to_numpy(dtype="float64"),
        label_coords.to_numpy(dtype="float64"),
        atol=1e-9,
        equal_nan=False,
    ):
        raise ValueError("Feature/label coordinate alignment failed.")

    feature_columns = select_spatial_feature_columns(merged)
    required_finite = [column for column in feature_columns if column not in STRUCTURAL_MISSING_FEATURES]
    finite_mask = np.isfinite(merged[required_finite].to_numpy(dtype="float64")).all(axis=1)
    dropped = merged.loc[~finite_mask].copy()
    retained = merged.loc[finite_mask].sort_values(["row", "col"]).reset_index(drop=True)
    if retained[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Retained spatial training rows contain fewer than two label classes.")

    missing_rates = merged[feature_columns].isna().mean().sort_values(ascending=False)
    summary = {
        "feature_path": str(feature_path),
        "label_path": str(label_path),
        "event_date": event_date,
        "event_id": str(labels["event_id"].iloc[0]),
        "source_event_window": "2019-08-01 to 2019-08-15",
        "raw_feature_rows_for_date": int(len(features)),
        "raw_label_rows": int(len(labels)),
        "retained_training_rows": int(len(retained)),
        "dropped_invalid_rows": int(len(dropped)),
        "dropped_invalid_positive_rows": int((dropped[TARGET_COLUMN] == 1).sum()),
        "class_counts": {str(k): int(v) for k, v in retained[TARGET_COLUMN].value_counts().sort_index().items()},
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "structural_missing_features_imputed": sorted(STRUCTURAL_MISSING_FEATURES.intersection(feature_columns)),
        "missing_rate_top10": {str(k): float(v) for k, v in missing_rates.head(10).items()},
        "limitations": [
            "Only one de-duplicated Sentinel-1 event is available, so this is a same-event spatial classifier.",
            "Sentinel-1 threshold labels are treated as candidate observed inundation, not perfect ground truth.",
            "Rows with non-finite required predictors are excluded; structural river-cell missingness is median-imputed inside each model pipeline.",
        ],
    }
    return SpatialTrainingDataset(retained, feature_columns, summary)


def make_spatial_block_split(
    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    min_positive_per_split: int = 2,
) -> SpatialSplit:
    row_median = float(frame["row"].median())
    col_median = float(frame["col"].median())
    candidates = OrderedDict(
        [
            ("north_holdout", frame["row"] <= row_median),
            ("south_holdout", frame["row"] > row_median),
            ("east_holdout", frame["col"] > col_median),
            ("west_holdout", frame["col"] <= col_median),
            ("northwest_holdout", (frame["row"] <= row_median) & (frame["col"] <= col_median)),
            ("southwest_holdout", (frame["row"] > row_median) & (frame["col"] <= col_median)),
        ]
    )

    scored: list[tuple[int, float, str, pd.Series]] = []
    for name, test_mask in candidates.items():
        train = frame.loc[~test_mask]
        test = frame.loc[test_mask]
        train_pos = int((train[target_column] == 1).sum())
        test_pos = int((test[target_column] == 1).sum())
        if train[target_column].nunique() < 2 or test[target_column].nunique() < 2:
            continue
        if train_pos < min_positive_per_split or test_pos < min_positive_per_split:
            continue
        fraction_penalty = abs((len(test) / len(frame)) - 0.35)
        scored.append((min(train_pos, test_pos), -fraction_penalty, name, test_mask))

    if not scored:
        raise ValueError("No spatial block split kept both classes with enough positives in train and test.")
    _, _, strategy, test_mask = sorted(scored, reverse=True)[0]
    train = frame.loc[~test_mask].sort_values(["row", "col"]).reset_index(drop=True)
    test = frame.loc[test_mask].sort_values(["row", "col"]).reset_index(drop=True)
    overlap = set(train["grid_cell_id"]).intersection(test["grid_cell_id"])
    if overlap:
        raise ValueError("Spatial split leakage: train and test share grid cells.")
    summary = {
        "strategy": strategy,
        "reason": (
            "Single-event labels rule out temporal and event-based holdouts; a contiguous spatial block "
            "holdout prevents identical grid cells from appearing in both train and test."
        ),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_class_counts": {str(k): int(v) for k, v in train[target_column].value_counts().sort_index().items()},
        "test_class_counts": {str(k): int(v) for k, v in test[target_column].value_counts().sort_index().items()},
        "train_grid_cells": int(train["grid_cell_id"].nunique()),
        "test_grid_cells": int(test["grid_cell_id"].nunique()),
    }
    return SpatialSplit(train, test, strategy, summary)


def _predict_scores(model: object, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype="float64")


def _metrics(y_true: pd.Series, y_score: np.ndarray) -> dict[str, object]:
    y_pred = (y_score >= 0.5).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    result: dict[str, object] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None if pd.Series(y_true).nunique() < 2 else float(roc_auc_score(y_true, y_score)),
        "pr_auc": None if pd.Series(y_true).nunique() < 2 else float(average_precision_score(y_true, y_score)),
        "confusion_matrix": matrix.tolist(),
    }
    return result


def _save_model(model: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _plot_confusion_matrices(results: dict[str, dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5.3 * len(results), 4.8))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, metrics) in zip(axes, results.items()):
        matrix = np.asarray(metrics["confusion_matrix"])
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks([0, 1], ["No flood", "Flood"])
        ax.set_yticks([0, 1], ["No flood", "Flood"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, int(value), ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_roc(y_true: pd.Series, scores: dict[str, np.ndarray], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, y_score in scores.items():
        RocCurveDisplay.from_predictions(y_true, y_score, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("Spatial Model ROC Curves")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_pr(y_true: pd.Series, scores: dict[str, np.ndarray], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, y_score in scores.items():
        PrecisionRecallDisplay.from_predictions(y_true, y_score, name=name, ax=ax)
    ax.set_title("Spatial Model Precision-Recall Curves")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _xgboost_shap_matrix(model: object, X: pd.DataFrame) -> tuple[object, pd.DataFrame]:
    estimator = model.named_steps["model"]
    transformed = pd.DataFrame(model.named_steps["imputer"].transform(X), columns=X.columns, index=X.index)
    return estimator, transformed


def _as_2d_shap_values(shap_values: object) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1] if len(shap_values) == 2 else shap_values[0])
    values = np.asarray(shap_values)
    if values.ndim == 3 and values.shape[2] == 2:
        return values[:, :, 1]
    if values.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values, got shape {values.shape}")
    return values


def generate_spatial_shap_outputs(
    model: object,
    X: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
    random_state: int = RANDOM_STATE,
    max_rows: int = 200,
) -> pd.DataFrame:
    import shap

    output_dir.mkdir(parents=True, exist_ok=True)
    sample = X if len(X) <= max_rows else X.sample(n=max_rows, random_state=random_state).sort_index()
    estimator, X_shap = _xgboost_shap_matrix(model, sample)
    explainer = shap.TreeExplainer(estimator)
    shap_values = _as_2d_shap_values(explainer.shap_values(X_shap, check_additivity=False))
    importance = (
        pd.DataFrame(
            {
                "feature": X_shap.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                "mean_shap": shap_values.mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.insert(0, "rank", range(1, len(importance) + 1))
    importance.to_csv(SPATIAL_SHAP_IMPORTANCE_PATH, index=False)

    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
    plt.title("Spatial XGBoost SHAP Summary")
    plt.tight_layout()
    plt.savefig(output_dir / "spatial_shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close("all")

    plt.figure()
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=20)
    plt.title("Spatial XGBoost SHAP Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "spatial_shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close("all")

    for feature in importance["feature"].head(3):
        plt.figure()
        shap.dependence_plot(feature, shap_values, X_shap, show=False)
        plt.title(f"Spatial SHAP Dependence: {feature}")
        plt.tight_layout()
        safe = feature.replace("/", "_").replace(" ", "_").lower()
        plt.savefig(output_dir / f"spatial_shap_dependence_{safe}.png", dpi=200, bbox_inches="tight")
        plt.close("all")
    return importance


def train_spatial_models(
    feature_path: Path = DEFAULT_FEATURE_PATH,
    label_path: Path = DEFAULT_LABEL_PATH,
    event_date: str = DEFAULT_EVENT_DATE,
    random_state: int = RANDOM_STATE,
    generate_shap: bool = True,
) -> pd.DataFrame:
    dataset = build_spatial_training_dataset(feature_path, label_path, event_date)
    split = make_spatial_block_split(dataset.frame)
    X_train = split.train[dataset.feature_columns]
    y_train = split.train[TARGET_COLUMN].astype(int)
    X_test = split.test[dataset.feature_columns]
    y_test = split.test[TARGET_COLUMN].astype(int)

    SPATIAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    models = build_spatial_model_candidates(y_train, random_state=random_state)
    results: dict[str, dict[str, object]] = {}
    scores: dict[str, np.ndarray] = {}
    prediction_frames: list[pd.DataFrame] = []
    for name, model in models.items():
        fitted = model.fit(X_train, y_train)
        _save_model(fitted, SPATIAL_MODELS_DIR / f"{name}.pkl")
        y_score = _predict_scores(fitted, X_test)
        scores[name] = y_score
        results[name] = _metrics(y_test, y_score)
        predictions = split.test[["grid_cell_id", "date", "row", "col", "latitude", "longitude", TARGET_COLUMN]].copy()
        predictions["model"] = name
        predictions["predicted_probability"] = y_score
        predictions["predicted_label"] = (y_score >= 0.5).astype(int)
        prediction_frames.append(predictions)

    metrics_df = pd.DataFrame(
        [{"model": name, **values} for name, values in results.items()]
    )
    metrics_df.to_csv(METRICS_DIR / "spatial_model_metrics.csv", index=False)
    metrics_df.to_json(METRICS_DIR / "spatial_model_metrics.json", orient="records", indent=2)
    pd.concat(prediction_frames, ignore_index=True).to_csv(SPATIAL_PREDICTIONS_PATH, index=False)
    _plot_confusion_matrices(results, FIGURES_DIR / "spatial_confusion_matrix.png")
    _plot_roc(y_test, scores, FIGURES_DIR / "spatial_roc_curve.png")
    _plot_pr(y_test, scores, FIGURES_DIR / "spatial_pr_curve.png")

    shap_top_features: list[str] = []
    if generate_shap:
        shap_importance = generate_spatial_shap_outputs(models["xgboost"], X_test, FIGURES_DIR, random_state=random_state)
        shap_top_features = shap_importance["feature"].head(5).tolist()

    summary = {
        "phase": "13_spatial_flood_model_training",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "dataset": dataset.summary,
        "split": split.summary,
        "class_imbalance_strategy": {
            "logistic_regression": "class_weight='balanced'",
            "random_forest": "class_weight='balanced_subsample'",
            "xgboost": f"scale_pos_weight={_positive_class_weight(y_train):.6f}",
        },
        "hyperparameters": SPATIAL_MODEL_HYPERPARAMETERS,
        "metrics": results,
        "shap_top_features": shap_top_features,
    }
    SPATIAL_TRAINING_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    return metrics_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Phase 13 spatial flood classification models.")
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--label-path", type=Path, default=DEFAULT_LABEL_PATH)
    parser.add_argument("--event-date", default=DEFAULT_EVENT_DATE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--skip-shap", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = train_spatial_models(
        feature_path=args.feature_path,
        label_path=args.label_path,
        event_date=args.event_date,
        random_state=args.random_state,
        generate_shap=not args.skip_shap,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
