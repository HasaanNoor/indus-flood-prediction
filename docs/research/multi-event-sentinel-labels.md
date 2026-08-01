# Phase 14 Research Notes: Multi-Event Sentinel-1 Label Expansion

## Repository Findings

Phase 12 created a canonical ERA5-derived spatial grid at `data_processed/spatial/grid_metadata.json`: EPSG:4326, 0.25 degree cells, row-major ordering, and 240 Sindh cells. Phase 13 training consumes one event label table keyed by `grid_cell_id` and rejects duplicate `grid_cell_id/event_id` pairs.

The existing Sentinel-1 workflow was a single-event GEE script for 2019-08-01 through 2019-08-15, with a 2019-06-15 through 2019-06-30 baseline. It used VH backscatter difference, JRC Global Surface Water masking, SRTM slope masking, and connected-component cleanup. The existing exported masks in `outputs/validation/sentinel1/` are two files for the same 2019 event and are not independent events.

The existing labels under `data_processed/spatial/labels/` flatten that one event onto the canonical grid, but permanent water is not supplied as a separate label. Phase 14 therefore adds a new inventory-driven pipeline and keeps the Phase 13 training defaults unchanged.

## Research And Documentation Reviewed

- Google Earth Engine Sentinel-1 guide and `COPERNICUS/S1_GRD` data catalog: Earth Engine GRD data are calibrated, terrain-corrected, log-scaled dB backscatter. The collection is heterogeneous, so filtering by polarization, instrument mode, resolution, and orbit direction is required. The catalog reports Sentinel-1 GRD availability beginning in October 2014. Sources: https://developers.google.cn/earth-engine/guides/sentinel1 and https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
- UN-SPIDER Sentinel-1 flood mapping recommended practice: use pre-event and during-event windows, select the same pass direction for change detection, support VH/VV choice, remove permanent water with JRC Global Surface Water, mask steep terrain, and use connected-pixel filtering to reduce speckle-like noise. Source: https://un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping/step-by-step
- JRC Global Surface Water documentation and Pekel et al. 2016: JRC maps long-term surface-water occurrence from Landsat and is suitable as a permanent-water mask, but it should be tracked separately from flood inundation. Source: https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_Metadata
- Sentinel-1 flood-mapping literature supports event-specific thresholding and sensitivity analysis rather than assuming one universal threshold. Otsu/adaptive thresholding can be useful when histogram sampling is well constrained, while fixed thresholds require documented sensitivity. Sources: https://doi.org/10.3390/rs13071342 and https://google-earth-engine.com/Aquatic-and-Hydrological-Applications/Surface-Water-Mapping/
- Multi-event validation studies emphasize testing across independent flood events and avoiding sensor/time-window leakage. Sen1Floods11 separates multiple flood events and distinguishes flood water from permanent water; recent multi-event validation work selects independent Sentinel-1 flood observations and masks permanent water during evaluation. Sources: https://doi.org/10.1109/CVPRW50498.2020.00113 and https://www.sciencedirect.com/science/article/pii/S2666017225000161
- Pakistan 2022 flood research shows Sentinel-1 is directly relevant for Sindh/Punjab inundation mapping, but those products are separate studies and are not copied into this repository. Source: https://www.sciencedirect.com/science/article/abs/pii/S003442572400066X
- Open-source implementation review: UN-SPIDER's GEE practice and community Sentinel-1 flood scripts commonly use image-count diagnostics, same-orbit filtering, permanent-water/slope masks, connected-pixel cleanup, and threshold sensitivity. These ideas were adopted as workflow structure only; no external implementation was copied.

## Adopted Decisions

The event inventory is JSON at `data_processed/spatial/labels/inventory/sentinel_event_inventory.json` because the repository already uses JSON metadata and has no YAML dependency.

Events before October 2014 are retained as historical flood candidates but marked unavailable for Sentinel-1 label generation. This prevents fabrication of 2010/2011 Sentinel-1 labels while keeping the event-selection audit explicit.

The GEE workflow uses one shared `processSentinelEvent` function and a separate `sentinel1_event_config.js` event list. Each event records windows, polarization, orbit pass, selected threshold, alternatives, permanent-water occurrence threshold, slope threshold, and connected-pixel threshold.

The Python pipeline only processes inventory events with a real exported mask. Missing candidate exports are reported as pending or unavailable rather than causing the entire run to fail.

Binary flood labels are aligned to the Phase 12 canonical grid using nearest-neighbor resampling. Bilinear interpolation is intentionally not used for binary labels.

Permanent water is preserved separately. If an exported GeoTIFF has band 2, it is read as permanent water. If a separate permanent-water raster path is supplied, that raster is used. If neither is supplied, the pipeline records `permanent_water_mode=not_supplied` and does not claim permanent-water completeness for that event.

`observed_inundation_label` is nullable. Permanent-water and NoData cells are not treated as valid flood/non-flood training labels in the Phase 14 outputs.

Duplicate event IDs, threshold variants, repeated raster hashes, repeated label-array hashes, and duplicate `event_id/grid_cell_id` rows are validated. Threshold variants can be represented, but they must point to a parent event and are not counted as independent events.

## Current Event Status

- `2010_indus_superflood_unavailable_s1`: unavailable because Sentinel-1 GRD did not exist yet.
- `2011_sindh_monsoon_unavailable_s1`: unavailable because Sentinel-1 GRD did not exist yet.
- `2015_sindh_monsoon_candidate`: Sentinel-1-era candidate, pending GEE image-count verification and export.
- `2019_sindh_monsoon_event_01`: existing repository mask available and processable.
- `2020_sindh_monsoon_candidate`: Sentinel-1-era candidate, pending GEE export.
- `2022_pakistan_sindh_event_01`: high-priority Sentinel-1-era candidate, pending GEE export.

## Limitations

The current local repository still has only one processable exported Sentinel-1 event. Phase 14 creates the deterministic multi-event inventory and ingestion workflow, but it does not by itself establish robust spatial generalization.

The existing 2019 mask does not include a separate permanent-water band. Future GEE exports should use the Phase 14 two-band export so candidate inundation and permanent water remain separable.

Sentinel-1 threshold masks are candidate observed inundation, not perfect ground truth. Thresholds are event-specific and must be documented with image counts, percentiles, and sensitivity area estimates.

No spatial model retraining is performed in this phase.
