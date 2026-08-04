# ADR-0011: Defer Robust Spatial Retraining Until Multiple Independent Events Exist

- Status: Accepted
- Date: 2026-08-04
- Decision owners: Project maintainers
- Related phases: Phase 13, Phase 14, Phase 15
- Related files: [docs/research/spatial-model-training.md](../research/spatial-model-training.md), [docs/research/multi-event-sentinel-labels.md](../research/multi-event-sentinel-labels.md), [docs/research/multi-event-processing.md](../research/multi-event-processing.md), [src/spatial/model_training.py](../../src/spatial/model_training.py), [src/spatial/sentinel_pipeline.py](../../src/spatial/sentinel_pipeline.py), [tests/test_spatial_model_training.py](../../tests/test_spatial_model_training.py), [tests/test_sentinel_phase14.py](../../tests/test_sentinel_phase14.py), [README.md](../../README.md)

## Context

The current local processed Sentinel-1 label inventory has one independent event, `2019_sindh_monsoon_event_01`. Phase 13 trains a same-event cell-level classifier using a deterministic spatial block holdout, but this does not test generalization to future flood events.

## Decision

Treat Phase 13 spatial model training as diagnostic and defer robust spatial retraining and generalization claims until multiple independent Sentinel-1 events are available and processed.

## Alternatives Considered

- Retrain robust spatial models from threshold variants of the 2019 event. Rejected because variants are not independent observations.
- Claim spatial generalization from a within-event block holdout. Rejected because event-level variation is untested.
- Avoid all spatial training until many events exist. Rejected because the pilot is useful for validating feature/label plumbing and diagnostic behavior.

## Consequences

### Positive

- The repository can test spatial modeling mechanics without overstating scientific conclusions.
- Event inventory work has a clear purpose: enable event-level validation later.
- README and reports can state limitations precisely.

### Negative

- Current spatial model artifacts are not strong evidence of future-event performance.
- Interview or review discussion must distinguish diagnostic model training from deployable spatial forecasting.

### Risks

- Users may still treat generated spatial probability maps as operational flood hazard products.
- One-event feature importance can reflect event-specific geography and threshold artifacts.

## Validation

[docs/research/spatial-model-training.md](../research/spatial-model-training.md) records the one-event limitation. [src/spatial/model_training.py](../../src/spatial/model_training.py) enforces one de-duplicated event label file and uses a spatial block holdout. [tests/test_spatial_model_training.py](../../tests/test_spatial_model_training.py) verifies block-split leakage prevention and deterministic training.

## Revisit Conditions

Revisit after at least several independent Sentinel-1 events are processed with consistent metadata, enough positive cells, and event-level train/test or cross-event validation.

## References

- [docs/research/spatial-model-training.md](../research/spatial-model-training.md)
- Roberts et al. 2017, spatial/temporal cross-validation: https://doi.org/10.1111/ecog.02881

