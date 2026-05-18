from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_data import DEFAULT_FEATURES_PATH, DEFAULT_TARGET, load_dataset, make_chronological_split
from ml_evaluation import (
    evaluate_predictions,
    get_feature_importance,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    predict_scores,
    save_metrics,
    save_model,
)
from ml_models import train_models
from paths import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PROJECT_ROOT


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
        f"({split.train_dates.min().date()} to {split.train_dates.max().date()}) "
        f"class_counts={split.y_train.value_counts().sort_index().to_dict()}"
    )
    print(
        "  Test: "
        f"{len(split.y_test)} rows "
        f"({split.test_dates.min().date()} to {split.test_dates.max().date()}) "
        f"class_counts={split.y_test.value_counts().sort_index().to_dict()}"
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


def main() -> None:
    train_flood_prediction_pipeline()


if __name__ == "__main__":
    main()
