# ADR-0004: Use Q95 Discharge Exceedance as the Flood-Event Target

- Status: Accepted
- Date: 2026-08-04
- Decision owners: Project maintainers
- Related phases: Temporal feature engineering and ML
- Related files: [src/build_features.py](../../src/build_features.py), [src/ml_data.py](../../src/ml_data.py), [src/train_flood_models.py](../../src/train_flood_models.py), [README.md](../../README.md)

## Context

The temporal pipeline needs daily binary targets for 1, 7, and 14 day forecasting. The repository has continuous GloFAS discharge over the full study period, while observed inundation labels are sparse Sentinel-1 event products.

## Decision

Define temporal flood-event labels as future discharge maxima meeting or exceeding the 95th percentile of GloFAS discharge from the chronological training portion. The labels are named `label_discharge_next_<horizon>d_ge_q95`.

## Alternatives Considered

- Use Sentinel-1 inundation as the temporal target. Rejected for the temporal pipeline because local Sentinel-1 labels currently cover only one independent processable event.
- Use administrative disaster reports as labels. Not implemented in the repository and would require a separate provenance and alignment workflow.
- Use a fixed physical discharge threshold. Deferred because station-specific flood stages and calibration metadata are not represented in the current processed data.

## Consequences

### Positive

- Produces consistent multi-horizon temporal labels across 2010-2023.
- Avoids computing the q95 threshold from the later test period.
- Supports rainfall-only versus hydrology-enhanced comparisons on a shared target.

### Negative

- Extreme discharge is not the same as observed inundation.
- Q95 is relative to this processed data record, not a calibrated flood-stage threshold.
- Hydrology-enhanced models include predictors from the same source family used to define the target.

### Risks

- Results may be misread as direct flood-extent prediction.
- Changes in GloFAS preprocessing or study period can change the q95 threshold.

## Validation

Target creation is implemented in [src/build_features.py](../../src/build_features.py). Feature selection in [src/ml_data.py](../../src/ml_data.py) excludes `label_`, `target_`, and `future` columns from predictors. README forecasting target text distinguishes high-discharge events from Sentinel-1 validation.

## Revisit Conditions

Revisit if reliable multi-year observed inundation labels, gauge flood-stage thresholds, or calibrated event catalogs are added.

## References

- [README.md](../../README.md)
- [src/build_features.py](../../src/build_features.py)

