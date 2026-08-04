# ADR-0008: Treat Sentinel-1 Masks as Candidate Observed Inundation Labels

- Status: Accepted
- Date: 2026-08-04
- Decision owners: Project maintainers
- Related phases: Sentinel-1 validation, Phase 12, Phase 14, Phase 15
- Related files: [docs/research/spatial-feature-grid.md](../research/spatial-feature-grid.md), [docs/research/multi-event-sentinel-labels.md](../research/multi-event-sentinel-labels.md), [docs/research/multi-event-processing.md](../research/multi-event-processing.md), [src/spatial/sentinel_ingestion.py](../../src/spatial/sentinel_ingestion.py), [src/spatial/sentinel_alignment.py](../../src/spatial/sentinel_alignment.py), [tests/test_sentinel_phase14.py](../../tests/test_sentinel_phase14.py), [gee/sentinel1_flood_validation_sindh.js](../../gee/sentinel1_flood_validation_sindh.js)

## Context

Sentinel-1 SAR can detect flood-related backscatter changes, but masks depend on event windows, orbit filtering, polarization, thresholding, permanent-water masking, slope masking, connected-pixel cleanup, and NoData handling. The repository uses these masks for validation and spatial labels, but they are not field-verified ground truth.

## Decision

Treat Sentinel-1 flood masks as candidate observed inundation labels. Preserve `observed_inundation_label`, `permanent_water_label`, `label_valid`, NoData, threshold metadata, event metadata, and source raster provenance separately.

## Alternatives Considered

- Treat Sentinel-1 masks as perfect ground truth. Rejected because SAR thresholding and masking choices introduce uncertainty.
- Collapse permanent water into flood labels. Rejected because permanent water and temporary inundation have different meanings.
- Convert all NoData to non-flood. Rejected because absent or invalid observations are not evidence of dry land.

## Consequences

### Positive

- Model training and validation keep label uncertainty visible.
- Permanent water does not inflate temporary flood counts.
- Future two-band exports can carry inundation and permanent-water information together.

### Negative

- Nullable labels add complexity to downstream training.
- Current local labels lack a complete permanent-water band for the original 2019 mask.

### Risks

- Users may overstate Sentinel-1 labels as ground truth if limitations are omitted from reports.
- Threshold sensitivity may change flood-cell counts materially.

## Validation

[src/spatial/sentinel_ingestion.py](../../src/spatial/sentinel_ingestion.py) preserves permanent water and NoData separately. [src/spatial/sentinel_alignment.py](../../src/spatial/sentinel_alignment.py) validates label raster values and uses nearest-neighbor alignment. [tests/test_sentinel_phase14.py](../../tests/test_sentinel_phase14.py) checks multiband permanent-water and NoData behavior.

## Revisit Conditions

Revisit if field validation, authoritative flood extent products, or stronger multi-sensor labels are incorporated.

## References

- [docs/research/multi-event-sentinel-labels.md](../research/multi-event-sentinel-labels.md)
- UN-SPIDER Sentinel-1 flood mapping practice: https://un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping/step-by-step
- JRC Global Surface Water: https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_GlobalSurfaceWater
- Sen1Floods11: https://doi.org/10.1109/CVPRW50498.2020.00113

