from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from paths import PROCESSED_FEATURES_DIR


DEFAULT_FEATURES_PATH = PROCESSED_FEATURES_DIR / "flood_ml_features.csv"
DEFAULT_TARGET = "label_discharge_next_1d_ge_q95"


@dataclass
class ChronologicalSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_dates: pd.Series
    test_dates: pd.Series
    feature_columns: list[str]


def load_dataset(input_path: Path = DEFAULT_FEATURES_PATH) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {input_path}")

    df = pd.read_csv(input_path, parse_dates=["date"])
    if "date" not in df.columns:
        raise ValueError("Dataset must include a 'date' column for chronological splitting.")

    return df.sort_values("date").reset_index(drop=True)


def select_feature_columns(df: pd.DataFrame, target_column: str = DEFAULT_TARGET) -> list[str]:
    excluded = {"date", target_column}
    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
        and not column.startswith("label_")
        and not column.startswith("target_")
        and "future" not in column.lower()
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    if not feature_columns:
        raise ValueError("No numeric feature columns were found after excluding targets and dates.")

    return feature_columns


def make_chronological_split(
    df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET,
    train_fraction: float = 0.65,
) -> ChronologicalSplit:
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    df = df.sort_values("date").reset_index(drop=True)
    feature_columns = select_feature_columns(df, target_column)

    split_index = int(len(df) * train_fraction)
    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Chronological split produced an empty train or test set.")

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    y_train = train_df[target_column].astype(int)
    y_test = test_df[target_column].astype(int)
    if y_train.nunique() < 2:
        raise ValueError(
            "Training split contains only one class. Use a later train_fraction or more data."
        )

    return ChronologicalSplit(
        X_train=train_df[feature_columns],
        X_test=test_df[feature_columns],
        y_train=y_train,
        y_test=y_test,
        train_dates=train_df["date"],
        test_dates=test_df["date"],
        feature_columns=feature_columns,
    )
