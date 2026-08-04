# ADR-0003: Maintain Rainfall-Only and Hydrology-Enhanced Model Tracks

- Status: Accepted
- Date: 2026-08-04
- Decision owners: Project maintainers
- Related phases: Temporal ML, SHAP explainability
- Related files: [src/build_features.py](../../src/build_features.py), [src/train_flood_models.py](../../src/train_flood_models.py), [src/run_shap_explainability.py](../../src/run_shap_explainability.py), [src/mapping/configuration.py](../../src/mapping/configuration.py), [tests/test_mapping.py](../../tests/test_mapping.py), [README.md](../../README.md)

## Context

The project evaluates whether river-discharge predictors add value beyond meteorological predictors. This distinction matters scientifically because rainfall-only models answer a different question from models that include GloFAS discharge and hydrology-derived state variables.

## Decision

Maintain separate rainfall-only and hydrology-enhanced datasets and model tracks. `build_feature_csv` writes `flood_features_rainfall_only.csv` and `flood_features_hydrology.csv`; `train_multihorizon_comparison_pipeline` trains both across 1, 7, and 14 day q95 labels.

## Alternatives Considered

- Train only the best-performing hydrology-enhanced model. Rejected because it would remove the controlled comparison.
- Train one dataset with all features and infer hydrological value from feature importances. Rejected because feature importance is not a substitute for an ablation-style dataset comparison.
- Exclude hydrology entirely. Rejected because GloFAS discharge is a central flood-relevant predictor in the repository.

## Consequences

### Positive

- The project can quantify added value from discharge features.
- SHAP findings can be interpreted separately for rainfall-only and hydrology-enhanced experiments.
- Mapping configuration can bind model artifacts to the correct dataset family.

### Negative

- More artifacts are produced and must be named consistently.
- Rainfall-only labels are still discharge-defined, so the task is not an observed-rainfall flood-extent target.

### Risks

- Users may compare model tracks without noticing that the target remains q95 discharge in both cases.
- Hydrology-enhanced superiority may partly reflect target/source proximity because both predictors and labels use GloFAS discharge.

## Validation

Dataset generation is implemented in [src/build_features.py](../../src/build_features.py). Multi-horizon training loops over both tracks in [src/train_flood_models.py](../../src/train_flood_models.py). Mapping tests verify hydrology and rainfall-only configuration behavior in [tests/test_mapping.py](../../tests/test_mapping.py).

## Revisit Conditions

Revisit if observed inundation labels become available for temporal modeling, if a third predictor family is added, or if the project changes from scientific comparison to a single operational model.

## References

- [README.md](../../README.md)
- [src/build_features.py](../../src/build_features.py)
- [src/train_flood_models.py](../../src/train_flood_models.py)

