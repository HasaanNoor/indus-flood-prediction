# Phase 12 Research Notes: Spatial Feature Grid Generation

## Repository Findings

The existing ML datasets in `data_processed/features/` are province-level temporal aggregate tables. They contain daily aggregate columns such as `era5_tp_daily_total_mean`, `glofas_discharge_max`, rolling windows, terrain summary statistics, and future discharge labels, but they do not contain a complete grid, CRS, affine transform, or stable cell ordering. Phase 11 correctly avoids reshaping those tables into rasters.

Available spatial inputs are:

- ERA5 daily multiyear processed NetCDF: `data_processed/era5/era5_sindh_multiyear_combined.nc`
  - WGS84 regular latitude/longitude grid, 37 x 33 cells, 0.25 degree spacing, daily 2010-01-01 through 2023-12-29.
  - Present variables: `tp`, `sro`, `e`, `pev`, `t2m`, `swvl1`, `swvl2`, `sp`, `u10`, `v10`.
- GloFAS daily multiyear processed NetCDF: `data_processed/glofas/glofas_sindh_multiyear_clean.nc`
  - WGS84 regular latitude/longitude grid, 280 x 270 cells, 0.05 degree spacing, daily 2010-01-02 through 2024-01-01.
  - Present discharge variable: `dis24`.
- SRTM clipped raster: `data_processed/clipped/srtm_sindh_clipped.tif`
  - EPSG:4326, 1 arc-second resolution, clipped to Sindh.
- Sindh boundary: `data_processed/boundaries/sindh_boundary.geojson`
  - EPSG:4326, one ADM1 Sindh polygon.
- Sentinel-1 validation masks: `outputs/validation/sentinel1/*.tif`
  - EPSG:4326 binary flood masks for a 2019 event.

## Literature And Technical Sources Reviewed

- ECMWF/Copernicus ERA5 documentation states that ERA5 HRES atmospheric data are about 31 km and that CDS NetCDF data are supplied on regular latitude/longitude grids after interpolation. It also notes bilinear interpolation for continuous fields and nearest neighbor for discrete fields in ERA5 retrieval contexts. This supports using the processed ERA5 0.25 degree grid as the canonical dynamic predictor grid rather than upscaling meteorology to finer SRTM/Sentinel-1 resolution. Source: https://confluence-stage.ecmwf.int/spaces/CKB/pages/76414402/ERA5%2Bdata%2Bdocumentation and https://confluence.ecmwf.int/pages/viewpage.action?pageId=78295305
- Copernicus CEMS GloFAS documentation describes GloFAS as a global flood forecasting system centered on river discharge products. This supports preserving discharge as a river-network feature instead of interpolating it across every land cell as if it were an areal land variable. Source: https://confluence.ecmwf.int/spaces/CEMS/pages/288346314/GloFAS%2BUser%2BGuide
- Sen1Floods11 distinguishes permanent water and flood water classes for Sentinel-1/Sentinel-2 flood mapping. This supports separate label columns for observed inundation, permanent water, and model-estimated probability. Source: https://doi.org/10.1109/CVPRW50498.2020.00113
- Martinis et al. (2022) emphasize that reference water masks improve flood mapping and reduce over-estimation of inundation extent, especially where seasonal water is dynamic. This supports treating permanent water as a separate exclusion/metadata concept, not as flood. Source: https://www.sciencedirect.com/science/article/pii/S0034425722001912
- Recent Sentinel-1 flood mapping work continues to stress terrain masking, permanent-water masking, threshold sensitivity, and field/independent validation limitations. This supports using Sentinel-1 masks as candidate observed inundation labels with explicit limitations. Sources include https://doi.org/10.1111/jfr3.70201 and https://journals.ametsoc.org/view/journals/bams/102/5/BAMS-D-19-0319.1.xml
- Rasterio/GDAL reprojection conventions support choosing resampling by data type: nearest for categorical masks, bilinear for continuous rasters, and explicit CRS/transform validation before any reshape or export. Source: https://rasterio.readthedocs.io/

## Open-Source References Reviewed

- `google-research/arco-era5` demonstrates ERA5 analysis-ready organization with explicit grid and time metadata. Adopted idea: make chunking/partitioning explicit and keep source dates in the feature table. Source: https://github.com/google-research/arco-era5
- `STURM-WEO/STURM-Flood` separates imagery, masks, probabilities, and evaluation outputs for flood extent mapping. Adopted idea: keep observed labels and probability products separate. Source: https://github.com/STURM-WEO/STURM-Flood
- A configurable Sentinel-1 flood-mapping GEE workflow documents slope, HAND/proximity, permanent water, and threshold guardrails. Adopted idea: record threshold metadata and avoid treating permanent water as flood. Source: https://gist.github.com/bennyistanto/934f3ab9b92fdc2b1aaa6436c480d80e

No external implementation code was copied.

## Unit Of Analysis

One row represents one canonical Sindh grid cell on one daily date. The output is a spatial-temporal feature grid, not a trained spatial model and not a flood-risk map.

## Canonical Grid Decision

The canonical grid is the processed ERA5 WGS84 regular latitude/longitude grid:

- CRS: EPSG:4326
- Resolution: 0.25 degrees
- Origin: upper-left outer corner derived from ERA5 cell centers
- Cell order: row-major, north-to-south then west-to-east
- Extent: ERA5 processed bounds covering Sindh and buffer cells
- Mask: Sindh boundary rasterized to the canonical grid
- Identifier: `sindh_era5_r{row:03d}_c{col:03d}`
- Temporal frequency: daily

Tradeoffs:

- ERA5 is coarser than GloFAS, SRTM, and Sentinel-1, but it is the coarsest spatially varying dynamic predictor. Choosing ERA5 avoids presenting meteorological detail finer than the source information supports.
- GloFAS at 0.05 degrees contains river discharge and is not an areal land-surface variable. It is linked to ERA5 cells through nearest valid river-cell discharge, distance to nearest river cell, and an explicit river-cell indicator.
- A separate coarser grid would reduce row count but would discard dynamic information already present in ERA5. A finer grid would require downscaling ERA5 and would create false precision.

## Resampling And Alignment Decisions

- ERA5: native canonical grid lookup; no spatial resampling.
- Continuous terrain elevation: bilinear reprojection to canonical grid.
- Terrain slope: computed from the aligned elevation surface.
- Categorical/binary masks: nearest-neighbor reprojection.
- Sentinel-1 flood masks: nearest-neighbor reprojection; output as candidate observed inundation labels.
- GloFAS discharge: river-aware nearest valid GloFAS river cell, plus `distance_to_glofas_river_km`, `has_glofas_river_cell`, and `glofas_river_discharge_m3s_on_river_cell` with `NaN` outside canonical cells that do not contain a valid GloFAS river cell.

## Temporal Alignment Decisions

Timestamps are normalized to timezone-naive daily dates. Duplicate daily dates are rejected. Requested dates must be present in each source. Lag and rolling features use only current or earlier dates; leading-edge lag windows are `NaN` when a run starts at the first available source date. Source date columns are preserved.

The current real-data ERA5/GloFAS overlap is 2010-01-02 through 2023-12-29.

## Spatial Feature Schema

The pipeline creates:

- Metadata: `grid_cell_id`, `date`, `row`, `col`, `latitude`, `longitude`, `source_era5_date`, `source_glofas_date`
- ERA5 current, lagged, and rolling features for variables present in the repository.
- Wind speed derived from `u10` and `v10`.
- GloFAS nearest river discharge, lagged/rolling discharge, distance to nearest river cell, and river-cell indicator.
- Terrain elevation, slope, and relative elevation within Sindh.
- Temporal features: month, day of year, sine/cosine seasonality, monsoon indicator.

Feature names intentionally differ from province-level aggregate feature names where the spatial meaning is different.

## Label Schema And Limitations

Sentinel-1 masks are aligned into `data_processed/spatial/labels/` with:

- `observed_inundation_label`
- `permanent_water_label`
- `model_estimated_flood_probability`
- event id, threshold, and source raster metadata

Current repository masks do not include a separate permanent-water raster or probability raster, so those fields are preserved as missing values. The current label inventory appears to cover one event and is insufficient for robust spatial model training.

## Scalability Decision

Full 2010-2023 generation is feasible only with partitioning. Approximate size is:

- ERA5 grid cells: 1,221 total; Sindh mask is smaller.
- Daily rows for 2010-2023: roughly Sindh cells x 5,113 dates.
- With dozens of feature columns, Parquet year partitions are preferred.

The CLI supports year partitions and restart-safe skipping of valid existing partitions unless `--overwrite` is passed. The default CLI range is a small 2019 pilot.

## Scientific Guardrails

This phase does not train a spatial model. It does not reuse temporal aggregate models for cell-level prediction. It does not claim Sentinel-1 masks are perfect ground truth. It does not treat permanent water as flood. It does not interpolate river discharge indiscriminately across land cells. It does not generate risk maps.
