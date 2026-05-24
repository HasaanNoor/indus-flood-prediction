from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml_data import DEFAULT_FEATURES_PATH, DEFAULT_TARGET, load_dataset, make_chronological_split
from ml_evaluation import (
    evaluate_predictions,
    get_feature_importance,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_grouped_roc_curves,
    plot_roc_curve,
    predict_scores,
    save_metrics,
    save_model,
)
from ml_models import train_models
from paths import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PROCESSED_FEATURES_DIR, PROJECT_ROOT


MULTIHORIZON_DATASETS = {
    "rainfall_only": PROCESSED_FEATURES_DIR / "flood_features_rainfall_only.csv",
    "hydrology": PROCESSED_FEATURES_DIR / "flood_features_hydrology.csv",
}

FORECAST_HORIZONS = [
    "label_discharge_next_1d_ge_q95",
    "label_discharge_next_7d_ge_q95",
    "label_discharge_next_14d_ge_q95",
]


def _date_range_text(dates: pd.Series) -> str:
    return f"{dates.min().date()} to {dates.max().date()}"


def _class_counts_text(y: pd.Series) -> dict[int, int]:
    counts = y.value_counts().sort_index().to_dict()
    return {int(label): int(count) for label, count in counts.items()}


def _safe_name(*parts: str) -> str:
    return "_".join(part.replace("/", "_").replace(" ", "_").lower() for part in parts)


def train_flood_prediction_pipeline(
    input_path: Path = DEFAULT_FEATURES_PATH,
    target_column: str = DEFAULT_TARGET,
    train_fraction: float = 0.65,
    random_state: int = 42,
) -> dict[str, dict[str, float | None]]:
    print("\nLoading flood feature dataset...")
    df = load_dataset(input_path)
    split = make_chronological_split(
        df,
        target_column=target_column,
        train_fraction=train_fraction,
    )

    print(f"  Rows: {len(df)}")
    print(f"  Features: {len(split.feature_columns)}")
    print(
        "  Train: "
        f"{len(split.y_train)} rows "
        f"({_date_range_text(split.train_dates)}) "
        f"class_counts={_class_counts_text(split.y_train)}"
    )
    print(
        "  Test: "
        f"{len(split.y_test)} rows "
        f"({_date_range_text(split.test_dates)}) "
        f"class_counts={_class_counts_text(split.y_test)}"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\nTraining models...")
    models = train_models(split.X_train, split.y_train, random_state=random_state)

    metrics: dict[str, dict[str, float | None]] = {}
    roc_scores: dict[str, pd.Series] = {}
    predictions = pd.DataFrame({"date": split.test_dates.reset_index(drop=True)})
    predictions["actual"] = split.y_test.reset_index(drop=True)

    print("\nEvaluating models...")
    for model_name, model in models.items():
        y_score = predict_scores(model, split.X_test)
        y_pred = (y_score >= 0.5).astype(int)

        metrics[model_name] = evaluate_predictions(split.y_test, y_pred, y_score)
        roc_scores[model_name] = y_score

        predictions[f"{model_name}_probability"] = y_score
        predictions[f"{model_name}_prediction"] = y_pred

        model_path = MODELS_DIR / f"{model_name}.pkl"
        save_model(model, model_path)

        plot_confusion_matrix(
            split.y_test,
            y_pred,
            model_name,
            FIGURES_DIR / f"confusion_matrix_{model_name}.png",
        )

        importance = get_feature_importance(model, split.feature_columns)
        importance.to_csv(METRICS_DIR / f"feature_importance_{model_name}.csv", index=False)
        plot_feature_importance(
            importance,
            model_name,
            FIGURES_DIR / f"feature_importance_{model_name}.png",
        )

        print(f"  {model_name}: {metrics[model_name]}")

    plot_roc_curve(split.y_test, roc_scores, FIGURES_DIR / "roc_curve_models.png")
    save_metrics(metrics, METRICS_DIR / "model_metrics.json")
    pd.DataFrame(metrics).T.to_csv(METRICS_DIR / "model_metrics.csv")
    predictions.to_csv(METRICS_DIR / "test_predictions.csv", index=False)

    print(f"\nSaved models: {MODELS_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Saved metrics: {METRICS_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Saved plots: {FIGURES_DIR.relative_to(PROJECT_ROOT)}")
    return metrics


def train_multihorizon_comparison_pipeline(
    train_fraction: float = 0.65,
    random_state: int = 42,
) -> pd.DataFrame:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    roc_curves_by_horizon: dict[str, list[dict[str, object]]] = {
        horizon: [] for horizon in FORECAST_HORIZONS
    }

    print("\nStarting multi-horizon flood model comparison.")
    print(f"Datasets: {', '.join(MULTIHORIZON_DATASETS)}")
    print(f"Horizons: {', '.join(FORECAST_HORIZONS)}")

    for dataset_type, input_path in MULTIHORIZON_DATASETS.items():
        print(f"\nDataset: {dataset_type}")
        print(f"  Path: {input_path.relative_to(PROJECT_ROOT)}")
        df = load_dataset(input_path)
        print(f"  Rows: {len(df)}")

        for horizon in FORECAST_HORIZONS:
            print(f"\nTraining horizon: {horizon}")
            split = make_chronological_split(
                df,
                target_column=horizon,
                train_fraction=train_fraction,
            )
            print(f"  Features: {len(split.feature_columns)}")
            print(
                "  Train: "
                f"{len(split.y_train)} rows "
                f"({_date_range_text(split.train_dates)}) "
                f"class_counts={_class_counts_text(split.y_train)}"
            )
            print(
                "  Test: "
                f"{len(split.y_test)} rows "
                f"({_date_range_text(split.test_dates)}) "
                f"class_counts={_class_counts_text(split.y_test)}"
            )

            models = train_models(split.X_train, split.y_train, random_state=random_state)

            for model_name, model in models.items():
                print(f"  Evaluating {model_name}...")
                y_score = predict_scores(model, split.X_test)
                y_pred = (y_score >= 0.5).astype(int)
                metrics = evaluate_predictions(split.y_test, y_pred, y_score)

                metric_row = {
                    "dataset_type": dataset_type,
                    "horizon": horizon,
                    "model": model_name,
                    **metrics,
                }
                all_metrics.append(metric_row)
                print(f"    Metrics: {metrics}")

                output_stem = _safe_name(dataset_type, horizon, model_name)
                save_model(model, MODELS_DIR / f"{output_stem}.pkl")

                plot_confusion_matrix(
                    split.y_test,
                    y_pred,
                    f"{dataset_type} | {horizon} | {model_name}",
                    FIGURES_DIR / f"confusion_matrix_{output_stem}.png",
                )

                importance = get_feature_importance(model, split.feature_columns)
                importance.to_csv(
                    METRICS_DIR / f"feature_importance_{output_stem}.csv",
                    index=False,
                )
                plot_feature_importance(
                    importance,
                    f"{dataset_type} | {horizon} | {model_name}",
                    FIGURES_DIR / f"feature_importance_{output_stem}.png",
                )

                prediction_frame = pd.DataFrame(
                    {
                        "date": split.test_dates.reset_index(drop=True),
                        "dataset_type": dataset_type,
                        "horizon": horizon,
                        "model": model_name,
                        "actual": split.y_test.reset_index(drop=True),
                        "predicted_probability": y_score,
                        "predicted_label": y_pred,
                    }
                )
                all_predictions.append(prediction_frame)

                roc_curves_by_horizon[horizon].append(
                    {
                        "label": f"{dataset_type} | {model_name}",
                        "y_true": split.y_test.reset_index(drop=True),
                        "y_score": np.asarray(y_score),
                    }
                )

    metrics_df = pd.DataFrame(all_metrics)[
        [
            "dataset_type",
            "horizon",
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
        ]
    ]
    metrics_df.to_csv(METRICS_DIR / "multihorizon_model_metrics.csv", index=False)
    metrics_df.to_json(
        METRICS_DIR / "multihorizon_model_metrics.json",
        orient="records",
        indent=2,
    )

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df.to_csv(METRICS_DIR / "multihorizon_test_predictions.csv", index=False)

    for horizon, curves in roc_curves_by_horizon.items():
        horizon_stem = _safe_name(horizon)
        plot_grouped_roc_curves(
            curves,
            FIGURES_DIR / f"roc_curves_{horizon_stem}.png",
            f"ROC Curves: {horizon}",
        )

    print("\nSummary comparison table:")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved metrics: {METRICS_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Saved predictions: {(METRICS_DIR / 'multihorizon_test_predictions.csv').relative_to(PROJECT_ROOT)}")
    print(f"Saved plots: {FIGURES_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Saved models: {MODELS_DIR.relative_to(PROJECT_ROOT)}")
    return metrics_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train flood forecasting models.")
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run the original single-target workflow instead of the multi-horizon comparison.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.single:
        train_flood_prediction_pipeline(
            train_fraction=args.train_fraction,
            random_state=args.random_state,
        )
    else:
        train_multihorizon_comparison_pipeline(
            train_fraction=args.train_fraction,
            random_state=args.random_state,
        )


if __name__ == "__main__":
    main()
