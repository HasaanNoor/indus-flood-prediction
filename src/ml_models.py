from __future__ import annotations

from collections import OrderedDict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _positive_class_weight(y_train: pd.Series) -> float:
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    if positives == 0:
        return 1.0
    return negatives / positives


def build_model_candidates(y_train: pd.Series, random_state: int = 42) -> OrderedDict[str, object]:
    models: OrderedDict[str, object] = OrderedDict()

    models["logistic_regression"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    models["random_forest"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        print(f"  Skipping xgboost: {exc}")
        return models

    models["xgboost"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=_positive_class_weight(y_train),
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return models


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> OrderedDict[str, object]:
    models = build_model_candidates(y_train, random_state=random_state)
    trained: OrderedDict[str, object] = OrderedDict()

    for name, model in models.items():
        print(f"  Training {name}...")
        trained[name] = model.fit(X_train, y_train)

    return trained
