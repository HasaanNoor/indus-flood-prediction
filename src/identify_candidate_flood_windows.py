from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paths import METRICS_DIR, PROJECT_ROOT


DEFAULT_INPUT_PATH = METRICS_DIR / "multihorizon_test_predictions.csv"
DEFAULT_OUTPUT_PATH = METRICS_DIR / "candidate_flood_windows.csv"

DATASET_TYPE = "hydrology"
MODEL_NAME = "xgboost"
PROBABILITY_THRESHOLD = 0.80
WINDOW_GAP_DAYS = 7

OUTPUT_COLUMNS = [
    "event_id",
    "start_date",
    "end_date",
    "peak_probability",
    "number_of_actual_flood_days",
    "number_of_predicted_flood_days",
]


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _validate_columns(df: pd.DataFrame, input_path: Path) -> None:
    required_columns = {
        "date",
        "dataset_type",
        "horizon",
        "model",
        "actual",
        "predicted_probability",
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{_relative_path(input_path)} is missing required columns: {missing}")


def _build_daily_candidates(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["actual_flood_flag"] = predictions["actual"].astype(int).eq(1)
    predictions["predicted_flood_flag"] = predictions["predicted_probability"].ge(
        PROBABILITY_THRESHOLD
    )

    daily = (
        predictions.groupby("date", as_index=False)
        .agg(
            peak_probability=("predicted_probability", "max"),
            actual_flood_flag=("actual_flood_flag", "max"),
            predicted_flood_flag=("predicted_flood_flag", "max"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["candidate_flag"] = daily["actual_flood_flag"] | daily["predicted_flood_flag"]
    return daily.loc[daily["candidate_flag"]].reset_index(drop=True)


def _build_event_windows(candidate_dates: pd.DataFrame) -> pd.DataFrame:
    if candidate_dates.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    events: list[dict[str, object]] = []
    current_event: list[dict[str, object]] = []
    previous_date: pd.Timestamp | None = None

    for row in candidate_dates.to_dict("records"):
        row_date = pd.Timestamp(row["date"])
        starts_new_event = (
            previous_date is not None
            and (row_date - previous_date) > pd.Timedelta(days=WINDOW_GAP_DAYS)
        )

        if starts_new_event:
            events.append(_summarize_event(len(events) + 1, current_event))
            current_event = []

        current_event.append(row)
        previous_date = row_date

    if current_event:
        events.append(_summarize_event(len(events) + 1, current_event))

    return pd.DataFrame(events, columns=OUTPUT_COLUMNS)


def _summarize_event(event_id: int, rows: list[dict[str, object]]) -> dict[str, object]:
    event = pd.DataFrame(rows)
    return {
        "event_id": event_id,
        "start_date": pd.Timestamp(event["date"].min()).date().isoformat(),
        "end_date": pd.Timestamp(event["date"].max()).date().isoformat(),
        "peak_probability": float(event["peak_probability"].max()),
        "number_of_actual_flood_days": int(event["actual_flood_flag"].sum()),
        "number_of_predicted_flood_days": int(event["predicted_flood_flag"].sum()),
    }


def identify_candidate_flood_windows(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    print("\nIdentifying candidate flood validation windows...")
    print(f"  Input: {_relative_path(input_path)}")
    print(f"  Dataset filter: {DATASET_TYPE}")
    print(f"  Model filter: {MODEL_NAME}")
    print(f"  Probability threshold: {PROBABILITY_THRESHOLD:.2f}")
    print(f"  Event merge gap: {WINDOW_GAP_DAYS} days")

    predictions = pd.read_csv(input_path, parse_dates=["date"])
    _validate_columns(predictions, input_path)
    print(f"  Loaded rows: {len(predictions):,}")

    filtered = predictions.loc[
        (predictions["dataset_type"] == DATASET_TYPE) & (predictions["model"] == MODEL_NAME)
    ].copy()
    if filtered.empty:
        raise ValueError(
            f"No rows found for dataset_type={DATASET_TYPE!r} and model={MODEL_NAME!r}"
        )

    horizons = ", ".join(sorted(filtered["horizon"].astype(str).unique()))
    print(f"  Filtered rows: {len(filtered):,}")
    print(f"  Horizons included: {horizons}")

    candidate_dates = _build_daily_candidates(filtered)
    print(f"  Candidate dates: {len(candidate_dates):,}")
    print(f"  Actual flood dates: {int(candidate_dates['actual_flood_flag'].sum()):,}")
    print(f"  Predicted flood dates: {int(candidate_dates['predicted_flood_flag'].sum()):,}")

    windows = _build_event_windows(candidate_dates)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(output_path, index=False)

    print(f"  Event windows: {len(windows):,}")
    if not windows.empty:
        print("\nCandidate windows:")
        for row in windows.itertuples(index=False):
            print(
                f"  {row.event_id}: {row.start_date} to {row.end_date} | "
                f"peak_probability={row.peak_probability:.3f} | "
                f"actual_days={row.number_of_actual_flood_days} | "
                f"predicted_days={row.number_of_predicted_flood_days}"
            )

    print(f"\nSaved candidate flood windows: {_relative_path(output_path)}")
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identify candidate flood validation windows from multihorizon predictions."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    identify_candidate_flood_windows(
        input_path=args.input_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
