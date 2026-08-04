# ADR-0010: Use Event-Based Label Inventory and Independence Checks

- Status: Accepted
- Date: 2026-08-04
- Decision owners: Project maintainers
- Related phases: Phase 14, Phase 15
- Related files: [docs/research/multi-event-sentinel-labels.md](../research/multi-event-sentinel-labels.md), [docs/research/multi-event-processing.md](../research/multi-event-processing.md), [data_processed/spatial/labels/inventory/sentinel_event_inventory.json](../../data_processed/spatial/labels/inventory/sentinel_event_inventory.json), [src/spatial/sentinel_inventory.py](../../src/spatial/sentinel_inventory.py), [src/spatial/sentinel_pipeline.py](../../src/spatial/sentinel_pipeline.py), [src/spatial/sentinel_validation.py](../../src/spatial/sentinel_validation.py), [tests/test_sentinel_phase14.py](../../tests/test_sentinel_phase14.py)

## Context

The original local Sentinel-1 files represented threshold/version variants of one 2019 event, not multiple independent flood observations. Counting variants as independent events would overstate label diversity and spatial model validation strength.

## Decision

Use a JSON event inventory as the Sentinel-1 provenance ledger. Track event identity, parent event relationships, independent-event status, processing status, threshold metadata, source raster hashes, label-array hashes, duplicate `event_id/grid_cell_id` checks, and combined-label regeneration from valid local event outputs.

## Alternatives Considered

- Discover every GeoTIFF in a folder and treat it as an event. Rejected because filenames alone cannot encode independence and provenance.
- Keep only a flat label Parquet. Rejected because it loses event lifecycle, threshold variants, and processing status.
- Count threshold variants as extra training events. Rejected because they reuse the same flood episode.

## Consequences

### Positive

- Event independence is auditable.
- Duplicate source rasters and duplicate label arrays are detectable.
- Pending, unavailable, failed, and processed events can coexist in one workflow.

### Negative

- New events require inventory maintenance and manual GEE export metadata.
- JSON schema changes require care because the inventory is both configuration and provenance.

### Risks

- Incorrect event metadata can still misrepresent independence.
- Hash checks detect duplicate files/arrays, not scientific equivalence of two different products.

## Validation

[src/spatial/sentinel_inventory.py](../../src/spatial/sentinel_inventory.py) validates duplicate IDs and parent references. [src/spatial/sentinel_ingestion.py](../../src/spatial/sentinel_ingestion.py) computes raster and label hashes. [tests/test_sentinel_phase14.py](../../tests/test_sentinel_phase14.py) checks threshold variants, duplicate hashes, restart-safe skips, and duplicate combined rows.

## Revisit Conditions

Revisit if the inventory grows enough to require a database, formal schema validation package, or integration with an external catalog.

## References

- [docs/research/multi-event-sentinel-labels.md](../research/multi-event-sentinel-labels.md)
- [docs/research/multi-event-processing.md](../research/multi-event-processing.md)
- Sen1Floods11: https://doi.org/10.1109/CVPRW50498.2020.00113

