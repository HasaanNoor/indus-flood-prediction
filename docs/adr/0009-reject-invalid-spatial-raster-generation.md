# ADR-0009: Reject Invalid Spatial Raster Generation

- Status: Accepted
- Date: 2026-08-04
- Decision owners: Project maintainers
- Related phases: Mapping safeguards, Phase 12, Phase 14
- Related files: [src/mapping/raster_export.py](../../src/mapping/raster_export.py), [src/mapping/pipeline.py](../../src/mapping/pipeline.py), [src/spatial/alignment.py](../../src/spatial/alignment.py), [src/spatial/sentinel_alignment.py](../../src/spatial/sentinel_alignment.py), [src/spatial/validation.py](../../src/spatial/validation.py), [tests/test_mapping.py](../../tests/test_mapping.py), [tests/test_spatial_phase12.py](../../tests/test_spatial_phase12.py), [tests/test_sentinel_phase14.py](../../tests/test_sentinel_phase14.py)

## Context

Temporal aggregate prediction CSVs and filtered spatial feature tables can lack the complete coordinate grid needed to write GeoTIFFs. Manufacturing rasters from incomplete, duplicated, irregular, CRS-less, or misaligned data would create visually convincing but invalid maps.

## Decision

Fail loudly when raster generation lacks complete coordinates, explicit CRS, unique coordinate pairs, regular spacing, matching dimensions, valid CRS, north-up transforms, or canonical-grid alignment. Write GeoTIFFs only from validated grid arrays.

## Alternatives Considered

- Infer CRS silently. Rejected because wrong CRS metadata can make maps spatially invalid.
- Fill missing grid cells from original unfiltered rows. Rejected because probabilities would no longer match retained inference rows.
- Export approximate CSV-shaped rasters. Rejected because it would hide alignment defects.

## Consequences

### Positive

- Prevents invalid GeoTIFFs from entering analysis outputs.
- Makes missing grid metadata visible to users.
- Keeps tabular probability export available when raster export is not defensible.

### Negative

- Some runs produce CSV outputs but no raster.
- Users must provide complete single-date spatial grids for raster export.

### Risks

- Strict validation can interrupt exploratory workflows.
- Future legitimate grids with unusual transforms would need explicit support.

## Validation

[src/mapping/raster_export.py](../../src/mapping/raster_export.py) rejects missing CRS, duplicate coordinates, incomplete grids, irregular spacing, and dimension mismatches. [src/spatial/sentinel_alignment.py](../../src/spatial/sentinel_alignment.py) rejects missing CRS, CRS mismatch, rotation/shear, and unsupported label values. Tests in [tests/test_mapping.py](../../tests/test_mapping.py), [tests/test_spatial_phase12.py](../../tests/test_spatial_phase12.py), and [tests/test_sentinel_phase14.py](../../tests/test_sentinel_phase14.py) cover these failures.

## Revisit Conditions

Revisit if the project introduces a formal raster metadata sidecar for prediction tables or supports non-regular grids through vector or cloud-optimized raster outputs.

## References

- [README.md](../../README.md)
- Rasterio documentation: https://rasterio.readthedocs.io/

