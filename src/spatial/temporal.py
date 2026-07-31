from __future__ import annotations

import pandas as pd


def normalise_daily_index(values: object, source_name: str) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(values))
    if dates.tz is not None:
        raise ValueError(f"{source_name} uses timezone-aware timestamps; expected UTC-normalized daily dates.")
    if dates.isna().any():
        raise ValueError(f"{source_name} contains missing timestamps.")
    normalized = dates.normalize()
    if normalized.duplicated().any():
        duplicated = normalized[normalized.duplicated()].unique()
        raise ValueError(f"{source_name} contains duplicate daily timestamps: {duplicated[:5].tolist()}")
    return pd.DatetimeIndex(normalized).sort_values()


def date_range(start_date: str, end_date: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start.tzinfo is not None or end.tzinfo is not None:
        raise ValueError("Date range must be timezone-naive daily dates.")
    if end < start:
        raise ValueError("end-date must be on or after start-date.")
    return pd.date_range(start.normalize(), end.normalize(), freq="D")


def validate_available_dates(available: pd.DatetimeIndex, requested: pd.DatetimeIndex, source_name: str) -> None:
    missing = requested.difference(available)
    if len(missing):
        sample = [str(value.date()) for value in missing[:5]]
        raise ValueError(f"{source_name} is missing requested dates: {sample}")


def validate_no_future_looking_join(feature_date: pd.Timestamp, source_dates: list[pd.Timestamp]) -> None:
    future = [value for value in source_dates if value.normalize() > feature_date.normalize()]
    if future:
        raise ValueError(f"Future-looking source dates detected for {feature_date.date()}: {future}")


def partition_name(start: pd.Timestamp, end: pd.Timestamp, partition_by_year: bool = False) -> str:
    if partition_by_year:
        if start.year != end.year:
            raise ValueError("Year partitions must stay within one calendar year.")
        return f"year={start.year}"
    return f"{start.date()}_{end.date()}"

