# ADR-0002: Use Chronological Splits for Temporal Flood Forecasting

- Status: Accepted
- Date: 2026-08-04
- Decision owners: Project maintainers
- Related phases: Temporal ML phases
- Related files: [src/ml_data.py](../../src/ml_data.py), [src/train_flood_models.py](../../src/train_flood_models.py), [src/build_features.py](../../src/build_features.py), [src/run_shap_explainability.py](../../src/run_shap_explainability.py), [README.md](../../README.md)

## Context

The temporal models forecast future high-discharge events from historical daily environmental features. A random row split would mix earlier and later monsoon seasons across train and test sets. That would make evaluation less like an operational forecast and could leak temporal regimes, seasonality, and future-derived feature behavior into model selection.

## Decision

Use chronological train/test splits for temporal flood forecasting. `load_dataset` sorts by `date`, `make_chronological_split` divides rows by time order, and target/future columns are excluded from predictors. The q95 threshold is also computed from the first chronological training portion in `add_multi_horizon_labels`.

## Alternatives Considered

- Random train/test split. Rejected because it can overstate forecasting performance by mixing future periods into training.
- Shuffled cross-validation. Rejected for the same leakage reason.
- Rolling-origin backtesting. Deferred because the current pipeline uses a simpler fixed chronological evaluation and already supports multi-horizon comparisons.

## Consequences

### Positive

- Test metrics reflect a forward-looking setting.
- The feature selector avoids label, target, and future columns.
- The evaluation can be explained as training on earlier observations and testing later observations.

### Negative

- A single split may be sensitive to the chosen boundary.
- Rare high-discharge events may be unevenly distributed across train and test.

### Risks

- Future contributors could accidentally add predictors that encode future information unless feature selection and tests remain strict.
- Climate and land-use non-stationarity can still affect results even without leakage.

## Validation

The behavior is implemented in [src/ml_data.py](../../src/ml_data.py) and used by [src/train_flood_models.py](../../src/train_flood_models.py) and [src/run_shap_explainability.py](../../src/run_shap_explainability.py). The README documents the chronological split. Spatial temporal alignment tests also reject future-looking joins in [tests/test_spatial_phase12.py](../../tests/test_spatial_phase12.py).

## Revisit Conditions

Revisit if the project adds enough events and years to support rolling-origin evaluation, blocked time-series cross-validation, or explicit event-based temporal holdouts.

## References

- [README.md](../../README.md)
- [src/ml_data.py](../../src/ml_data.py)
- [src/build_features.py](../../src/build_features.py)

