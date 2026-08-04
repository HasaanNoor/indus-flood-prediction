# Architecture Decision Records

This directory records major engineering, data, modeling, and scientific decisions for the Indus flood prediction project. ADRs are used for decisions that shape how the repository is maintained, reviewed, extended, or explained.

| ADR | Title | Status | Date | Summary |
|---|---|---|---|---|
| [ADR-0001](0001-record-architecture-decisions.md) | Use Architecture Decision Records | Accepted | 2026-08-04 | Introduces ADRs as the durable record for major project decisions. |
| [ADR-0002](0002-use-chronological-splits-for-temporal-flood-forecasting.md) | Use Chronological Splits for Temporal Flood Forecasting | Accepted | 2026-08-04 | Keeps temporal evaluation forward-looking and avoids leakage from future observations. |
| [ADR-0003](0003-maintain-rainfall-only-and-hydrology-enhanced-model-tracks.md) | Maintain Rainfall-Only and Hydrology-Enhanced Model Tracks | Accepted | 2026-08-04 | Preserves paired meteorological-only and discharge-enhanced experiments. |
| [ADR-0004](0004-use-q95-discharge-exceedance-as-the-flood-event-target.md) | Use Q95 Discharge Exceedance as the Flood-Event Target | Accepted | 2026-08-04 | Defines temporal flood-event labels using high GloFAS discharge, not observed inundation. |
| [ADR-0005](0005-keep-temporal-and-spatial-flood-models-separate.md) | Keep Temporal and Spatial Flood Models Separate | Accepted | 2026-08-04 | Separates province-level temporal forecasting from grid-cell spatial classification. |
| [ADR-0006](0006-use-era5-as-the-canonical-spatial-grid.md) | Use ERA5 as the Canonical Spatial Grid | Accepted | 2026-08-04 | Uses the 0.25-degree WGS84 ERA5 grid as the canonical spatial feature grid. |
| [ADR-0007](0007-preserve-river-aware-glofas-semantics.md) | Preserve River-Aware GloFAS Semantics | Accepted | 2026-08-04 | Represents GloFAS discharge as a river-network feature instead of interpolating it across land. |
| [ADR-0008](0008-treat-sentinel-1-masks-as-candidate-observed-inundation-labels.md) | Treat Sentinel-1 Masks as Candidate Observed Inundation Labels | Accepted | 2026-08-04 | Keeps Sentinel-1 labels separate from permanent water, NoData, and ground-truth claims. |
| [ADR-0009](0009-reject-invalid-spatial-raster-generation.md) | Reject Invalid Spatial Raster Generation | Accepted | 2026-08-04 | Fails raster generation when grid metadata or alignment is invalid. |
| [ADR-0010](0010-use-event-based-label-inventory-and-independence-checks.md) | Use Event-Based Label Inventory and Independence Checks | Accepted | 2026-08-04 | Tracks independent Sentinel-1 events, variants, hashes, and duplicate labels explicitly. |
| [ADR-0011](0011-defer-robust-spatial-retraining-until-multiple-independent-events-exist.md) | Defer Robust Spatial Retraining Until Multiple Independent Events Exist | Accepted | 2026-08-04 | Treats the Phase 13 spatial model as diagnostic until multiple events are processed. |
| [ADR-0012](0012-use-deterministic-partitioned-parquet-outputs-for-spatial-data.md) | Use Deterministic, Partitioned Parquet Outputs for Spatial Data | Accepted | 2026-08-04 | Uses restart-safe Parquet partitions for spatial features and labels. |

## Process

Create a new ADR when a decision changes the project architecture, scientific method, data semantics, validation approach, or interpretation of outputs. Use the next sequential number, keep the status current, and link source files, tests, research notes, and external references that support the decision.

