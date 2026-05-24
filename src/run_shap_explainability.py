from __future__ import annotations

import argparse
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml_data import load_dataset, make_chronological_split
from paths import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PROCESSED_FEATURES_DIR, PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SHAP_FIGURES_DIR = FIGURES_DIR / "shap"
SHAP_IMPORTANCE_PATH = METRICS_DIR / "shap_feature_importance.csv"
MODEL_NAME = "xgboost"


@dataclass(frozen=True)
class ShapExperiment:
    dataset_type: str
    dataset_path: Path
    horizon: str


EXPERIMENTS = [
    ShapExperiment(
        dataset_type="hydrology",
        dataset_path=PROCESSED_FEATURES_DIR / "flood_features_hydrology.csv",
        horizon="label_discharge_next_1d_ge_q95",
    ),
    ShapExperiment(
        dataset_type="hydrology",
        dataset_path=PROCESSED_FEATURES_DIR / "flood_features_hydrology.csv",
        horizon="label_discharge_next_7d_ge_q95",
    ),
    ShapExperiment(
        dataset_type="hydrology",
        dataset_path=PROCESSED_FEATURES_DIR / "flood_features_hydrology.csv",
        horizon="label_discharge_next_14d_ge_q95",
    ),
    ShapExperiment(
        dataset_type="rainfall_only",
        dataset_path=PROCESSED_FEATURES_DIR / "flood_features_rainfall_only.csv",
        horizon="label_discharge_next_14d_ge_q95",
    ),
]


def _safe_name(*parts: str) -> str:
    return "_".join(part.replace("/", "_").replace(" ", "_").lower() for part in parts)


def _load_model(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(
            f"Required XGBoost model not found: {path}. "
            "Run src/train_flood_models.py before running SHAP explainability."
        )

    with path.open("rb") as handle:
        return pickle.load(handle)


def _prepare_xgboost_matrix(model: object, X: pd.DataFrame) -> tuple[object, pd.DataFrame]:
    if not hasattr(model, "named_steps"):
        return model, X

    if "model" not in model.named_steps:
        raise ValueError("Expected a pipeline with a 'model' step.")

    estimator = model.named_steps["model"]
    if "xgb" not in estimator.__class__.__name__.lower():
        raise ValueError(f"SHAP explainability is only supported for XGBoost models: {estimator}")

    transformed = X
    if "imputer" in model.named_steps:
        transformed = pd.DataFrame(
            model.named_steps["imputer"].transform(X),
            columns=X.columns,
            index=X.index,
        )

    return estimator, transformed


def _sample_test_rows(
    X_test: pd.DataFrame,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if len(X_test) <= max_rows:
        return X_test.copy()
    return X_test.sample(n=max_rows, random_state=random_state).sort_index()


def _as_2d_shap_values(shap_values: object) -> np.ndarray:
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            return np.asarray(shap_values[1])
        return np.asarray(shap_values[0])

    values = np.asarray(shap_values)
    if values.ndim == 3 and values.shape[2] == 2:
        return values[:, :, 1]
    if values.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values, got shape {values.shape}")
    return values


def _save_summary_plot(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    import shap

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close("all")


def _save_bar_plot(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    import shap

    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close("all")


def _save_dependence_plot(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    feature: str,
    output_path: Path,
    title: str,
) -> None:
    import shap

    plt.figure()
    shap.dependence_plot(feature, shap_values, X_sample, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close("all")


def _calculate_importance(
    shap_values: np.ndarray,
    feature_columns: list[str],
    experiment: ShapExperiment,
    rows_sampled: int,
) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "dataset_type": experiment.dataset_type,
            "horizon": experiment.horizon,
            "model": MODEL_NAME,
            "feature": feature_columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
            "std_abs_shap": np.abs(shap_values).std(axis=0),
            "rows_sampled": rows_sampled,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.insert(0, "rank", range(1, len(importance) + 1))
    return importance.reset_index(drop=True)


def run_shap_explainability(
    max_sample_rows: int = 500,
    train_fraction: float = 0.65,
    random_state: int = 42,
) -> pd.DataFrame:
    import shap

    logging.info("Using SHAP version: %s", shap.__version__)
    SHAP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    importance_frames: list[pd.DataFrame] = []

    for experiment in EXPERIMENTS:
        horizon_label = experiment.horizon.replace("label_discharge_next_", "").replace("_ge_q95", "")
        logging.info(
            "Starting SHAP analysis | dataset_type=%s | forecast_horizon=%s | target=%s",
            experiment.dataset_type,
            horizon_label,
            experiment.horizon,
        )

        df = load_dataset(experiment.dataset_path)
        split = make_chronological_split(
            df,
            target_column=experiment.horizon,
            train_fraction=train_fraction,
        )

        X_test_sample = _sample_test_rows(
            split.X_test,
            max_rows=max_sample_rows,
            random_state=random_state,
        )
        logging.info(
            "Sampled test rows | dataset_type=%s | forecast_horizon=%s | rows_sampled=%d/%d",
            experiment.dataset_type,
            horizon_label,
            len(X_test_sample),
            len(split.X_test),
        )

        stem = _safe_name(experiment.dataset_type, experiment.horizon, MODEL_NAME)
        model_path = MODELS_DIR / f"{stem}.pkl"
        pipeline = _load_model(model_path)
        xgb_model, X_shap = _prepare_xgboost_matrix(pipeline, X_test_sample)

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = _as_2d_shap_values(
            explainer.shap_values(X_shap, check_additivity=False)
        )

        importance = _calculate_importance(
            shap_values,
            feature_columns=list(X_shap.columns),
            experiment=experiment,
            rows_sampled=len(X_shap),
        )
        importance_frames.append(importance)

        top_features = importance["feature"].head(5).tolist()
        logging.info(
            "Top SHAP features | dataset_type=%s | forecast_horizon=%s | features=%s",
            experiment.dataset_type,
            horizon_label,
            ", ".join(top_features),
        )

        summary_path = SHAP_FIGURES_DIR / f"summary_{stem}.png"
        bar_path = SHAP_FIGURES_DIR / f"bar_importance_{stem}.png"
        _save_summary_plot(
            shap_values,
            X_shap,
            summary_path,
            f"SHAP Summary: {experiment.dataset_type} | {experiment.horizon}",
        )
        _save_bar_plot(
            shap_values,
            X_shap,
            bar_path,
            f"SHAP Importance: {experiment.dataset_type} | {experiment.horizon}",
        )
        logging.info("Saved SHAP summary plot: %s", summary_path.relative_to(PROJECT_ROOT))
        logging.info("Saved SHAP bar importance plot: %s", bar_path.relative_to(PROJECT_ROOT))

        for feature in top_features:
            dependence_path = SHAP_FIGURES_DIR / f"dependence_{stem}_{_safe_name(feature)}.png"
            _save_dependence_plot(
                shap_values,
                X_shap,
                feature,
                dependence_path,
                f"SHAP Dependence: {feature}",
            )
            logging.info(
                "Saved SHAP dependence plot | feature=%s | path=%s",
                feature,
                dependence_path.relative_to(PROJECT_ROOT),
            )

    all_importance = pd.concat(importance_frames, ignore_index=True)
    all_importance.to_csv(SHAP_IMPORTANCE_PATH, index=False)
    logging.info(
        "Saved ranked SHAP feature importance table: %s",
        SHAP_IMPORTANCE_PATH.relative_to(PROJECT_ROOT),
    )
    return all_importance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SHAP explainability outputs for selected XGBoost flood models."
    )
    parser.add_argument(
        "--max-sample-rows",
        type=int,
        default=500,
        help="Maximum number of chronological test rows to sample per experiment.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_shap_explainability(
        max_sample_rows=args.max_sample_rows,
        train_fraction=args.train_fraction,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
