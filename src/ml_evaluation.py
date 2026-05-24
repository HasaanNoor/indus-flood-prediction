from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

from paths import PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def predict_scores(model: object, X_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(X_test)


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if pd.Series(y_true).nunique() < 2:
        metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))

    return metrics


def _final_estimator(model: object) -> object:
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]
    return model


def get_feature_importance(model: object, feature_columns: list[str]) -> pd.DataFrame:
    estimator = _final_estimator(model)

    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance = np.abs(estimator.coef_).ravel()
    else:
        importance = np.zeros(len(feature_columns))

    return (
        pd.DataFrame({"feature": feature_columns, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["No flood", "Flood"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix: {model_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_feature_importance(
    importance: pd.DataFrame,
    model_name: str,
    output_path: Path,
    top_n: int = 20,
) -> None:
    top = importance.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance"], color="#2563eb")
    ax.set_title(f"Feature Importance: {model_name}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_curve(
    y_true: pd.Series,
    model_scores: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    if pd.Series(y_true).nunique() < 2:
        ax.text(0.5, 0.5, "ROC curve unavailable: test set has one class", ha="center")
        ax.set_axis_off()
    else:
        for model_name, y_score in model_scores.items():
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_grouped_roc_curves(
    curves: list[dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7))
    plotted = False

    for curve in curves:
        y_true = pd.Series(curve["y_true"])
        y_score = np.asarray(curve["y_score"])
        label = str(curve["label"])

        if y_true.nunique() < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, linewidth=1.8, label=f"{label} (AUC={auc:.3f})")
        plotted = True

    if plotted:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.25)
    else:
        ax.text(0.5, 0.5, "ROC curves unavailable: test sets have one class", ha="center")
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_model(model: object, output_path: Path) -> None:
    with output_path.open("wb") as handle:
        pickle.dump(model, handle)


def save_metrics(metrics: dict[str, dict[str, float | None]], output_path: Path) -> None:
    with output_path.open("w") as handle:
        json.dump(metrics, handle, indent=2)
