# ADR-0005: Keep Temporal and Spatial Flood Models Separate

- Status: Accepted
- Date: 2026-08-04
- Decision owners: Project maintainers
- Related phases: Phase 12, Phase 13, mapping safeguards
- Related files: [docs/research/spatial-feature-grid.md](../research/spatial-feature-grid.md), [docs/research/spatial-model-training.md](../research/spatial-model-training.md), [src/spatial/model_training.py](../../src/spatial/model_training.py), [src/mapping/configuration.py](../../src/mapping/configuration.py), [src/mapping/inference.py](../../src/mapping/inference.py), [tests/test_mapping.py](../../tests/test_mapping.py)

## Context

The early ML datasets are province-level daily aggregate tables. They do not contain a complete grid, affine transform, stable cell ordering, or cell-level labels. Spatial flood inference requires a different unit of analysis: one grid cell on one date.

## Decision

Keep temporal forecasting models and spatial flood models separate. Province-level temporal models predict future q95 discharge events. Phase 13 spatial models are trained as cell-level classifiers using spatial feature rows and Sentinel-1-derived labels.

## Alternatives Considered

- Apply temporal aggregate models to every grid cell. Rejected because the feature semantics and unit of analysis differ.
- Reshape temporal aggregate CSVs into rasters. Rejected because those CSVs lack grid completeness and transform metadata.
- Replace temporal models with spatial models. Rejected because they answer different questions.

## Consequences

### Positive

- Prevents invalid reuse of aggregate temporal features for grid-cell inference.
- Allows mapping code to bind model architecture and dataset type explicitly.
- Keeps spatial labels tied to grid cells and Sentinel-1 events.

### Negative

- Two modeling paths require separate artifacts, documentation, and validation.
- Spatial model maturity is limited by available event labels.

### Risks

- Users may still confuse temporal risk classes with spatial inundation probability if outputs are not labeled carefully.
- Feature names that look similar can have different spatial meanings.

## Validation

Research notes state that Phase 12 does not train a model or reuse temporal models for cell inference. [src/spatial/model_training.py](../../src/spatial/model_training.py) builds a separate spatial training dataset. [tests/test_mapping.py](../../tests/test_mapping.py) verifies spatial model binding and horizon/dataset checks.

## Revisit Conditions

Revisit if the project develops a unified spatiotemporal architecture with explicit cell-level time histories and event-level validation.

## References

- [docs/research/spatial-feature-grid.md](../research/spatial-feature-grid.md)
- [docs/research/spatial-model-training.md](../research/spatial-model-training.md)

