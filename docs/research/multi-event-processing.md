# Phase 15 Research Notes: Multi-Event Sentinel-1 Export, Processing, And Label Expansion

## Repository Inspection

Phase 14 already provided the core architecture for multi-event Sentinel-1 labels:

- Inventory: `data_processed/spatial/labels/inventory/sentinel_event_inventory.json`
- GEE configuration: `gee/sentinel1_event_config.js`
- Shared GEE processing script: `gee/sentinel1_flood_validation_sindh.js`
- Python ingestion pipeline: `src/spatial/sentinel_pipeline.py`, `src/spatial/sentinel_ingestion.py`, `src/spatial/sentinel_alignment.py`
- Event outputs: `data_processed/spatial/labels/events/<event_id>/`
- Combined labels: `data_processed/spatial/labels/combined/sentinel_labels_all_events.parquet`
- Validation reports: `outputs/validation/sentinel_label_inventory_report.*`

The canonical grid remains the Phase 12 ERA5-derived Sindh grid in EPSG:4326 with 240 in-province cells. Binary Sentinel-1 labels are nearest-neighbor aligned to that grid. The current local GeoTIFF inventory contains one independent processable event, `2019_sindh_monsoon_event_01`; the 2015, 2020, and 2022 targets are retained as Sentinel-1-era candidates pending manual Earth Engine verification and export.

## Literature And Data Constraints Reviewed

- Google Earth Engine Sentinel-1 guidance and the `COPERNICUS/S1_GRD` catalog describe the collection as heterogeneous and require filtering by metadata such as polarization, instrument mode, resolution, and orbit pass before comparing images. Earth Engine GRD images are preprocessed to calibrated, terrain-corrected dB backscatter. Sources: https://developers.google.cn/earth-engine/guides/sentinel1 and https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
- UN-SPIDER's Sentinel-1 flood-mapping practice supports pre-event/during-event change detection, same-orbit filtering, permanent-water masking, terrain masking, connected-pixel cleanup, flood-area reporting, and export from GEE. Source: https://un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping/step-by-step
- JRC Global Surface Water is appropriate as a long-term permanent-water reference, but permanent water must remain separate from temporary flood labels. Source: https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_GlobalSurfaceWater
- Sentinel-1 flood-mapping literature continues to show threshold sensitivity across regions and events. This phase records thresholds and sensitivity diagnostics but does not infer event thresholds automatically. Sources: https://doi.org/10.3390/rs13071342 and https://doi.org/10.1111/jfr3.70201
- Multi-event validation literature emphasizes independent flood observations, permanent-water exclusion, terrain masking, and reporting performance/event balance across many events rather than reusing threshold variants as independent evidence. Sources: https://doi.org/10.1109/CVPRW50498.2020.00113 and https://www.sciencedirect.com/science/article/pii/S2666017225000161

## Phase 15 Decisions

The event lifecycle is explicit: candidate, verified export available, processed, unavailable, failed, or pending GEE export. Python processing never fabricates an event when a GeoTIFF is absent.

The GEE workflow remains a single configurable script. Event selection, threshold, polarization, orbit filtering, diagnostic printing, image counts, flood-area estimates, permanent-water-area estimates, and two-band export are driven by the event configuration. The expected export name is `sentinel1_flood_mask_sindh_<event_id>.tif`.

The Python pipeline accepts `--event`, `--all-events`, `--overwrite`, and `--skip-existing`. Existing outputs are reused only when labels, metadata, validation JSON, source raster hash, and label-array hash are valid. Combined labels are regenerated from every valid local event output, so processing a single event cannot silently drop other processed events.

The inventory JSON is now updated as the provenance ledger with processed status, validation status, raster hash, label hash, flood-cell count, non-flood count, permanent-water count, image counts when available, export timestamp when available, and processing version.

The combined dataset rejects duplicate `event_id/grid_cell_id` combinations and requires `event_id`, `grid_cell_id`, `observed_inundation_label`, `permanent_water_label`, `event_date`, `threshold`, and `processing_version`.

## Event Target Status

- `2015_sindh_monsoon_candidate`: Sentinel-1-era candidate; pending GEE image-count verification and export.
- `2020_sindh_monsoon_candidate`: Sentinel-1-era candidate; pending GEE image-count verification and export.
- `2022_pakistan_sindh_event_01`: Sentinel-1-era candidate; pending GEE image-count verification and export.
- `2019_sindh_monsoon_event_01`: locally available and processable from the existing repository export.

## Scientific Guardrails

Thresholds are event metadata, not automatically invented. Threshold variants are sensitivity runs, not new independent events. Permanent water is preserved separately and excluded from valid temporary-inundation labels. The outputs are candidate observed inundation labels, not ground truth. Phase 15 does not retrain spatial models or claim cross-event generalization until additional independent exported events are processed.
