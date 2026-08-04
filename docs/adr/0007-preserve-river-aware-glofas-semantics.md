# ADR-0007: Preserve River-Aware GloFAS Semantics

- Status: Accepted
- Date: 2026-08-04
- Decision owners: Project maintainers
- Related phases: Phase 12, Phase 13
- Related files: [docs/research/spatial-feature-grid.md](../research/spatial-feature-grid.md), [src/spatial/alignment.py](../../src/spatial/alignment.py), [src/spatial/features.py](../../src/spatial/features.py), [src/spatial/model_training.py](../../src/spatial/model_training.py), [tests/test_spatial_phase12.py](../../tests/test_spatial_phase12.py)

## Context

GloFAS represents river discharge. Treating discharge as a smooth land-surface raster would imply meaningful discharge values in cells that are not river cells, which would distort the scientific meaning of the predictor.

## Decision

Represent GloFAS with river-aware features: nearest valid river-cell discharge, distance to the nearest GloFAS river cell, an explicit `has_glofas_river_cell` indicator, and `glofas_river_discharge_m3s_on_river_cell` set to missing outside cells containing a valid river cell.

## Alternatives Considered

- Bilinear interpolation across every land cell. Rejected because discharge is not an areal field like precipitation.
- Drop GloFAS from spatial features. Rejected because river discharge is central to flood risk in the Indus basin.
- Use only a binary river proximity indicator. Rejected because it would discard discharge magnitude.

## Consequences

### Positive

- Preserves the river-network meaning of discharge.
- Makes structural missingness explicit and available to model pipelines.
- Allows spatial models to use both hydrologic magnitude and distance-to-river context.

### Negative

- Nearest-river features are still approximations at ERA5 grid scale.
- Models need imputation for structurally missing on-river discharge values.

### Risks

- Distance-to-river and nearest discharge may still be too coarse for local flood hydraulics.
- Future contributors might remove the explicit river indicator and make missingness harder to interpret.

## Validation

River mapping is implemented in [src/spatial/alignment.py](../../src/spatial/alignment.py) and consumed in [src/spatial/features.py](../../src/spatial/features.py). [tests/test_spatial_phase12.py](../../tests/test_spatial_phase12.py) checks river-cell handling and resampling-mode selection. [src/spatial/model_training.py](../../src/spatial/model_training.py) treats on-river discharge as structural missingness.

## Revisit Conditions

Revisit if river network geometry, gauge locations, hydraulic model outputs, or catchment-routing features are added.

## References

- [docs/research/spatial-feature-grid.md](../research/spatial-feature-grid.md)
- Copernicus CEMS GloFAS User Guide: https://confluence.ecmwf.int/spaces/CEMS/pages/288346314/GloFAS+User+Guide

