# Indus River Flood Prediction and Risk Assessment

A machine learning and remote sensing framework for multi-horizon flood forecasting and validation in the Lower Indus River Basin using hydrological, meteorological, topographic, and satellite datasets.

---

## Overview

This project develops a machine learning-based flood forecasting framework for Sindh Province, Pakistan, using hydrological, meteorological, topographic, and satellite-derived datasets. The objective is to evaluate the ability of environmental predictors to forecast high-discharge flood events across multiple forecasting horizons while maintaining temporal integrity and model explainability.

The project combines:

- ERA5 precipitation and atmospheric variables
- GloFAS river discharge data
- SRTM elevation data
- Sentinel-1 SAR imagery for flood validation
- Machine learning forecasting models
- Explainable AI techniques (SHAP)

---

## Current Status

### Completed

- Data ingestion and preprocessing
- Feature engineering
- Multi-horizon forecasting (1-day, 7-day, and 14-day horizons)
- Logistic Regression baseline
- Random Forest
- XGBoost
- SHAP explainability
- Sentinel-1 SAR validation

### In Progress

- Flood-risk mapping
- Additional Sentinel-1 validation events
- Automated testing suite
- Spatial validation and susceptibility mapping

---

## Key Results

- XGBoost achieved the strongest overall performance across forecasting horizons.
- Hydrology-enhanced models consistently outperformed rainfall-only models.
- SHAP analysis identified GloFAS discharge variables as the most influential predictors.
- Sentinel-1 SAR validation detected flood extents ranging from 15,943 km² to 20,872 km² depending on threshold selection.
- River discharge variables provided substantial predictive value beyond precipitation-only inputs.

---

## Study Area

- **Region:** Sindh Province, Pakistan
- **Basin:** Lower Indus River Basin
- **Time Period:** 2010–2023
- **Spatial Reference:** EPSG:4326

---

## Data Sources

### ERA5 Reanalysis

**Source:** ECMWF Copernicus Climate Data Store

Variables include:

- Total precipitation
- Accumulated rainfall metrics
- Temporal rainfall aggregations

### GloFAS

**Source:** Copernicus Emergency Management Service

Variables include:

- River discharge
- Maximum discharge
- Rolling discharge statistics

### SRTM

**Source:** NASA Shuttle Radar Topography Mission

Variables include:

- Elevation
- Terrain-derived features

### Sentinel-1 SAR

**Source:** ESA Sentinel-1 Ground Range Detected (GRD)

Used for independent validation of detected flood events through radar backscatter change analysis.

---

## Feature Engineering

More than 150 predictive features were generated from environmental datasets.

### Rainfall Features

- Rainfall accumulation over 1, 3, 7, 14, and 30-day windows
- Rolling precipitation statistics
- Lagged rainfall variables
- Rainfall anomaly indicators

### Hydrological Features

- River discharge lags
- Maximum discharge values
- Rolling discharge averages
- Discharge anomaly indicators
- Multi-window discharge summaries

### Terrain Features

- Elevation-derived variables
- Topographic context
- Terrain-based flood susceptibility indicators

---

## Forecasting Targets

Flood events are defined as extreme discharge occurrences exceeding the 95th percentile of observed discharge values.

Forecast horizons evaluated:

- 1-day flood prediction
- 7-day flood prediction
- 14-day flood prediction

---

## Machine Learning Models

### Logistic Regression

Baseline probabilistic classifier using a linear decision boundary.

### Random Forest

Ensemble learning model that aggregates predictions from multiple decision trees trained on bootstrapped samples.

### XGBoost

Gradient-boosted decision tree model that sequentially learns residual errors from previous trees to improve predictive performance.

---

## Experimental Design

### Temporal Train/Test Split

**Training Period:** 2010–2019

**Testing Period:** 2019–2023

The split is strictly chronological to prevent temporal leakage and ensure realistic forecasting evaluation.

### Dataset Comparisons

Two forecasting pipelines were evaluated:

1. Rainfall-only features
2. Hydrology-enhanced features (rainfall + discharge)

---

## Explainability

SHAP (SHapley Additive exPlanations) was applied to trained XGBoost models to identify the environmental variables contributing most strongly to flood predictions.

### Key Findings

- Hydrology-enhanced models were primarily driven by GloFAS discharge variables.
- Rainfall-only models relied more heavily on accumulated precipitation features.
- River discharge metrics consistently ranked among the most important predictors.
- Hydrological information substantially improved predictive performance across all forecast horizons.

---

## Sentinel-1 Flood Validation

Independent validation was conducted using Sentinel-1 SAR imagery.

### Validation Workflow

1. Construct pre-flood and during-flood composites.
2. Compute VH backscatter differences.
3. Detect significant reductions in radar backscatter associated with surface water expansion.
4. Remove permanent water bodies using JRC Global Surface Water.
5. Remove steep terrain using SRTM slope filtering.
6. Apply connected-component filtering to reduce noise.

### Threshold Sensitivity Analysis

| VH Threshold | Detected Flood Area |
|-------------|-------------------:|
| -2.4 dB | 20,872 km² |
| -3.0 dB | 15,943 km² |

The stricter **-3.0 dB** threshold was selected as the preferred validation mask due to reduced false positives and improved spatial coherence along the Indus floodplain.

---

## Multi-Horizon Spatial Flood-Risk Inference

The repository includes a deterministic inference workflow for the trained multi-horizon XGBoost models. The workflow loads saved model artifacts and processed feature datasets; it does not retrain models.

Supported forecast horizons:

- 1-day
- 7-day
- 14-day

Default model and dataset family:

- Hydrology-enhanced XGBoost models
- `data_processed/features/flood_features_hydrology.csv`
- `outputs/models/hydrology_label_discharge_next_{1d,7d,14d}_ge_q95_xgboost.pkl`

Run all horizons:

```bash
python3 -m src.mapping.pipeline --all-horizons
```

Run one horizon:

```bash
python3 -m src.mapping.pipeline --horizon 7day
```

The default risk thresholds are `0.25`, `0.50`, and `0.75`. They can be overridden with three strictly increasing values in `[0, 1]`:

```bash
python3 -m src.mapping.pipeline --all-horizons --thresholds 0.2 0.5 0.8
```

The current processed prediction CSVs are daily aggregate feature tables and do not contain latitude/longitude coordinates, CRS, affine transform, or a complete raster grid. Because of that, the workflow currently produces scientifically valid tabular probability outputs and metadata, and it deliberately skips GeoTIFF and map PNG products rather than fabricating spatial rasters. If early rolling-window rows contain non-finite features, a smoke run on the valid subset can be performed explicitly:

```bash
python3 -m src.mapping.pipeline --all-horizons --drop-invalid-rows
```

Expected output structure for the current aggregate datasets:

```text
outputs/
  flood_risk_maps/
    1day/
      predictions.csv
      metadata.json
    7day/
      predictions.csv
      metadata.json
    14day/
      predictions.csv
      metadata.json
```

`predictions.csv` contains the model-estimated flood-event probability, numeric risk class, risk label, forecast horizon, and timestamp where available. Risk classes are probability-derived categories only:

| Class | Label |
|------:|-------|
| 1 | Low |
| 2 | Moderate |
| 3 | High |
| 4 | Very High |

Boundary behavior is deterministic: `Low` is below the first threshold, `Moderate` starts at the first threshold, `High` starts at the second threshold, and `Very High` starts at the third threshold.

These outputs should be interpreted as model-estimated flood-event probability categories under the project target definition. They are not calibrated hazard, exposure, vulnerability, or comprehensive disaster-risk products. Sentinel-1 SAR validation remains an independent observed flood-extent reference and is not used as a model input in this workflow.

GeoTIFF and map PNG export is implemented for future prediction datasets that provide a complete regular latitude/longitude grid plus explicit CRS metadata. Categorical risk rasters are written as integer classes without interpolation when such inputs are available.

When `--drop-invalid-rows` is used with a spatial prediction grid, raster reconstruction uses only the exact
rows retained for inference. If dropping invalid feature rows makes a formerly complete regular grid incomplete,
the workflow rejects raster generation with a clear grid-completeness error instead of combining probabilities
from filtered rows with coordinates from the original unfiltered grid.

---

## Spatial Feature Grid Generation

Phase 12 adds a deterministic spatial data foundation for future spatial flood-event classification. It does not train a spatial model and it does not use the existing province-level temporal models for grid-cell inference.

Unit of analysis:

- One row = one canonical grid cell on one daily date
- Canonical grid = processed ERA5 WGS84 0.25 degree grid
- Cell ordering = row-major, north-to-south then west-to-east
- Mask = Sindh boundary rasterized to the canonical grid
- Stable ID = `grid_cell_id`

The ERA5 grid is used because it is the coarsest spatially varying dynamic predictor already present in the repository. GloFAS discharge is handled as a river-aware feature using nearest valid river-cell discharge, distance to nearest GloFAS river cell, and an explicit river-cell indicator. It is not interpolated across all land cells as though every cell were a river.

Spatial feature outputs:

```text
data_processed/
  spatial/
    grid_metadata.json
    grid_cells.csv
    features/
      spatial_features_<date_or_partition>.parquet
    labels/
      sentinel1_labels_<event_id>.parquet
      label_availability_report.json
    validation/
      spatial_grid_report.json
      spatial_grid_report.md
```

Run only the canonical grid build:

```bash
python3 -m src.spatial.pipeline --build-grid
```

Run a small real-data pilot with Sentinel-1 label alignment:

```bash
python3 -m src.spatial.pipeline \
  --start-date 2019-08-01 \
  --end-date 2019-08-15 \
  --include-labels
```

Run a partitioned full-period build:

```bash
python3 -m src.spatial.pipeline \
  --start-year 2010 \
  --end-year 2023 \
  --partition-by-year
```

The default run intentionally uses a small pilot range. Full 2010-2023 generation should be year-partitioned to avoid unnecessary memory and disk pressure. Sentinel-1 labels are candidate observed inundation labels, not perfect ground truth; permanent water and model-estimated probability are preserved as separate concepts.

See `docs/research/spatial-feature-grid.md` for the canonical-grid decision, resampling rules, label limitations, and scientific guardrails.

---

## Results

Major findings include:

- Hydrology-enhanced models consistently outperformed rainfall-only models.
- XGBoost achieved the strongest overall performance across forecasting horizons.
- Forecast skill decreased as prediction horizons increased.
- River discharge information provided substantial predictive value beyond precipitation alone.
- Sentinel-1 validation demonstrated spatial correspondence between modeled flood-risk areas and observed inundation patterns.

---

## Project Structure

```text
src/
gee/
tests/
data_raw/
data_processed/
outputs/
validation/
```

---

## Future Work

- Flood susceptibility mapping
- Spatial flood-risk products
- Additional Sentinel-1 validation events
- Hyperparameter optimization
- Automated testing and validation framework
- Streamlit dashboard deployment
- Integration of land-cover and soil-moisture predictors
- Multi-basin generalization experiments
- Real-time forecasting workflows
