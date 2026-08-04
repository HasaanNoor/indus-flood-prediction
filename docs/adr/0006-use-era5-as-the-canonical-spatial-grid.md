# ADR-0006: Use ERA5 as the Canonical Spatial Grid

- Status: Accepted
- Date: 2026-08-04
- Decision owners: Project maintainers
- Related phases: Phase 12
- Related files: [docs/research/spatial-feature-grid.md](../research/spatial-feature-grid.md), [src/spatial/grid.py](../../src/spatial/grid.py), [src/spatial/features.py](../../src/spatial/features.py), [src/spatial/validation.py](../../src/spatial/validation.py), [tests/test_spatial_phase12.py](../../tests/test_spatial_phase12.py), [data_processed/spatial/grid_metadata.json](../../data_processed/spatial/grid_metadata.json)

## Context

Spatial predictors come from sources with different resolutions and meanings: ERA5 is a 0.25-degree atmospheric grid, GloFAS is a finer river-discharge grid, SRTM is much finer terrain, and Sentinel-1 is a raster flood mask. The project needs one stable grid for tabular spatial features and labels.

## Decision

Use the processed ERA5 WGS84 regular latitude/longitude grid as the canonical spatial grid. The grid is EPSG:4326, 0.25 degrees, row-major north-to-south then west-to-east, with stable `sindh_era5_r{row:03d}_c{col:03d}` identifiers and a Sindh boundary mask.

## Alternatives Considered

- Use the GloFAS 0.05-degree grid. Rejected because GloFAS discharge is river-network data, not an areal land-surface predictor for every cell.
- Use a custom finer grid based on SRTM or Sentinel-1. Rejected because it would downscale ERA5 and create false spatial precision.
- Use a coarser custom grid. Rejected because it would discard dynamic information already present in ERA5.

## Consequences

### Positive

- The grid matches the coarsest dynamic spatial predictor.
- Cell ordering, IDs, CRS, bounds, and transform are deterministic.
- SRTM and Sentinel-1 can be aligned using data-type-appropriate resampling.

### Negative

- The grid is coarse for local inundation mapping.
- Fine-scale topographic and Sentinel-1 information is aggregated or resampled to ERA5 scale.

### Risks

- Users may overinterpret cell-level probabilities as fine-scale flood extent.
- A future higher-resolution meteorological product could make the canonical grid obsolete.

## Validation

Grid construction and metadata are implemented in [src/spatial/grid.py](../../src/spatial/grid.py). Validation checks are implemented in [src/spatial/validation.py](../../src/spatial/validation.py) and tested in [tests/test_spatial_phase12.py](../../tests/test_spatial_phase12.py). The research note records the source dimensions and grid tradeoffs.

## Revisit Conditions

Revisit if the project adopts a defensible downscaling method, a higher-resolution dynamic predictor, or a modeling task that requires a different unit of analysis.

## References

- [docs/research/spatial-feature-grid.md](../research/spatial-feature-grid.md)
- ECMWF ERA5 documentation: https://confluence.ecmwf.int/pages/viewpage.action?pageId=78295305

