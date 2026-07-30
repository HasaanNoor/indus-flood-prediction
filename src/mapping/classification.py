from __future__ import annotations

import numpy as np

from src.mapping.configuration import DEFAULT_THRESHOLDS, RISK_CLASSES


def validate_thresholds(thresholds: tuple[float, float, float] = DEFAULT_THRESHOLDS) -> tuple[float, float, float]:
    if len(thresholds) != 3:
        raise ValueError("Exactly three thresholds are required for four risk classes.")
    values = tuple(float(value) for value in thresholds)
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Risk thresholds must be within [0, 1].")
    if not values[0] < values[1] < values[2]:
        raise ValueError("Risk thresholds must be strictly increasing.")
    return values


def validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype="float64")
    finite = np.isfinite(values)
    if np.any(finite & ((values < 0.0) | (values > 1.0))):
        raise ValueError("Flood probabilities must be within [0, 1].")
    return values


def classify_probabilities(
    probabilities: np.ndarray,
    thresholds: tuple[float, float, float] = DEFAULT_THRESHOLDS,
) -> np.ndarray:
    """Classify probabilities using left-closed upper classes.

    For thresholds (t1, t2, t3): p < t1 is Low, t1 <= p < t2 is
    Moderate, t2 <= p < t3 is High, and p >= t3 is Very High.
    NaN values remain 0 for NoData handling.
    """
    t1, t2, t3 = validate_thresholds(thresholds)
    values = validate_probabilities(probabilities)
    classes = np.zeros(values.shape, dtype="uint8")
    finite = np.isfinite(values)
    classes[finite & (values < t1)] = 1
    classes[finite & (values >= t1) & (values < t2)] = 2
    classes[finite & (values >= t2) & (values < t3)] = 3
    classes[finite & (values >= t3)] = 4
    return classes


def labels_for_classes(classes: np.ndarray) -> np.ndarray:
    values = np.asarray(classes)
    return np.vectorize(lambda value: RISK_CLASSES.get(int(value), "NoData"))(values)
