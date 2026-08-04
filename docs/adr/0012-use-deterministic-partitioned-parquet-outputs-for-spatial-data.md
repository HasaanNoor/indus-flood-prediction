# ADR-0012: Use Deterministic, Partitioned Parquet Outputs for Spatial Data

- Status: Accepted
- Date: 2026-08-04
- Decision owners: Project maintainers
- Related phases: Phase 12, Phase 14, Phase 15
- Related files: [docs/research/spatial-feature-grid.md](../research/spatial-feature-grid.md), [docs/research/multi-event-processing.md](../research/multi-event-processing.md), [src/spatial/pipeline.py](../../src/spatial/pipeline.py), [src/spatial/alignment.py](../../src/spatial/alignment.py), [src/spatial/sentinel_ingestion.py](../../src/spatial/sentinel_ingestion.py), [src/spatial/sentinel_pipeline.py](../../src/spatial/sentinel_pipeline.py), [tests/test_spatial_phase12.py](../../tests/test_spatial_phase12.py), [tests/test_sentinel_phase14.py](../../tests/test_sentinel_phase14.py)

## Context

Spatial features are one row per grid cell per day. Full 2010-2023 generation can produce many rows and many feature columns. Sentinel-1 labels are also event-scoped and need restart-safe processing as additional exports arrive.

## Decision

Use deterministic Parquet outputs for spatial features and labels, with year partitioning for long feature ranges and event directories for Sentinel-1 labels. Writes go through a temporary file followed by replace, and existing event outputs are reused only when schema and hash validation pass.

## Alternatives Considered

- Large monolithic CSV files. Rejected because they are bulkier, slower to read/write, and weaker for typed nullable labels.
- One all-years Parquet file. Rejected because year partitions make long runs easier to restart and inspect.
- Always overwrite outputs. Rejected because manual Sentinel-1 exports and long feature runs benefit from restart safety.

## Consequences

### Positive

- Spatial feature generation can be restarted by partition.
- Nullable labels and typed columns are preserved better than in CSV.
- Event processing can skip valid outputs and regenerate combined labels from all valid local events.

### Negative

- Parquet requires compatible dependencies.
- Partitioned outputs require consumers to handle multiple files.

### Risks

- Non-deterministic dependency behavior could still affect binary Parquet bytes across environments.
- Schema drift across partitions would need validation before large training runs.

## Validation

[src/spatial/pipeline.py](../../src/spatial/pipeline.py) supports `--partition-by-year`, `--overwrite`, and partition naming. [src/spatial/alignment.py](../../src/spatial/alignment.py) performs temporary Parquet writes. [src/spatial/sentinel_ingestion.py](../../src/spatial/sentinel_ingestion.py) validates existing outputs with source and label hashes. Tests in [tests/test_spatial_phase12.py](../../tests/test_spatial_phase12.py) and [tests/test_sentinel_phase14.py](../../tests/test_sentinel_phase14.py) cover deterministic writes and restart-safe skips.

## Revisit Conditions

Revisit if spatial data volume outgrows local Parquet files, if cloud-native storage is adopted, or if schema versioning across many partitions becomes necessary.

## References

- [docs/research/spatial-feature-grid.md](../research/spatial-feature-grid.md)
- [docs/research/multi-event-processing.md](../research/multi-event-processing.md)

