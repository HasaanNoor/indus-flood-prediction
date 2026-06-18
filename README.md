Indus River Flood Prediction and Risk Assessment

Overview

This project develops a machine learning-based flood forecasting framework for Sindh Province, Pakistan, using hydrological, meteorological, topographic, and satellite-derived datasets. The objective is to evaluate the ability of environmental predictors to forecast high-discharge flood events across multiple forecasting horizons while maintaining temporal integrity and model explainability.

The project combines:

* ERA5 precipitation and atmospheric variables
* GloFAS river discharge data
* SRTM elevation data
* Sentinel-1 SAR imagery for flood validation
* Machine learning forecasting models
* Explainable AI techniques (SHAP)

⸻

Current Status

Completed

* Data ingestion and preprocessing
* Feature engineering
* Multi-horizon forecasting (1-day, 7-day, and 14-day horizons)
* Logistic Regression baseline
* Random Forest
* XGBoost
* SHAP explainability
* Sentinel-1 SAR validation

In Progress

* Flood-risk mapping
* Additional Sentinel-1 validation events
* Automated testing suite
* Spatial validation and susceptibility mapping

⸻

Key Results

* XGBoost achieved the strongest overall performance across forecasting horizons.
* Hydrology-enhanced models consistently outperformed rainfall-only models.
* SHAP analysis identified GloFAS discharge variables as the most influential predictors.
* Sentinel-1 SAR validation detected flood extents ranging from 15,943 km² to 20,872 km² depending on threshold selection.
* River discharge variables provided substantial predictive value beyond precipitation-only inputs.

⸻

Study Area

* Region: Sindh Province, Pakistan
* Basin: Lower Indus River Basin
* Time Period: 2010–2023
* Spatial Reference: EPSG:4326

⸻

Data Sources

ERA5 Reanalysis

Source: ECMWF Copernicus Climate Data Store

Variables include:

* Total precipitation
* Accumulated rainfall metrics
* Temporal rainfall aggregations

GloFAS

Source: Copernicus Emergency Management Service

Variables include:

* River discharge
* Maximum discharge
* Rolling discharge statistics

SRTM

Source: NASA Shuttle Radar Topography Mission

Variables include:

* Elevation
* Terrain-derived features

Sentinel-1 SAR

Source: ESA Sentinel-1 Ground Range Detected (GRD)

Used for independent validation of detected flood events through radar backscatter change analysis.

⸻

Feature Engineering

More than 150 predictive features were generated from environmental datasets.

Rainfall Features

* Rainfall accumulation over 1, 3, 7, 14, and 30-day windows
* Rolling precipitation statistics
* Lagged rainfall variables
* Rainfall anomaly indicators

Hydrological Features

* River discharge lags
* Maximum discharge values
* Rolling discharge averages
* Discharge anomaly indicators
* Multi-window discharge summaries

Terrain Features

* Elevation-derived variables
* Topographic context
* Terrain-based flood susceptibility indicators

⸻

Forecasting Targets

Flood events are defined as extreme discharge occurrences exceeding the 95th percentile of observed discharge values.

Forecast horizons evaluated:

* 1-day flood prediction
* 7-day flood prediction
* 14-day flood prediction

⸻

Machine Learning Models

Logistic Regression

Baseline probabilistic classifier using a linear decision boundary.

Random Forest

Ensemble learning model that aggregates predictions from multiple decision trees trained on bootstrapped samples.

XGBoost

Gradient-boosted decision tree model that sequentially learns residual errors from previous trees to improve predictive performance.

⸻

Experimental Design

Temporal Train/Test Split

Training Period: 2010–2019

Testing Period: 2019–2023

The split is strictly chronological to prevent temporal leakage and ensure realistic forecasting evaluation.

Dataset Comparisons

Two forecasting pipelines were evaluated:

1. Rainfall-only features
2. Hydrology-enhanced features (rainfall + discharge)

⸻

Explainability

SHAP (SHapley Additive exPlanations) was applied to trained XGBoost models to identify the environmental variables contributing most strongly to flood predictions.

Key Findings

* Hydrology-enhanced models were primarily driven by GloFAS discharge variables.
* Rainfall-only models relied more heavily on accumulated precipitation features.
* River discharge metrics consistently ranked among the most important predictors.
* Hydrological information substantially improved predictive performance across all forecast horizons.

⸻

Sentinel-1 Flood Validation

Independent validation was conducted using Sentinel-1 SAR imagery.

Validation Workflow

1. Construct pre-flood and during-flood composites.
2. Compute VH backscatter differences.
3. Detect significant reductions in radar backscatter associated with surface water expansion.
4. Remove permanent water bodies using JRC Global Surface Water.
5. Remove steep terrain using SRTM slope filtering.
6. Apply connected-component filtering to reduce noise.

Threshold Sensitivity Analysis

VH Threshold	Detected Flood Area
-2.4 dB	20,872 km²
-3.0 dB	15,943 km²

The stricter -3.0 dB threshold was selected as the preferred validation mask due to reduced false positives and improved spatial coherence along the Indus floodplain.

⸻

Results

Major findings include:

* Hydrology-enhanced models consistently outperformed rainfall-only models.
* XGBoost achieved the strongest overall performance across forecasting horizons.
* Forecast skill decreased as prediction horizons increased.
* River discharge information provided substantial predictive value beyond precipitation alone.
* Sentinel-1 validation demonstrated spatial correspondence between modeled flood-risk areas and observed inundation patterns.

⸻

Project Structure

src/
gee/
tests/
data_raw/
data_processed/
outputs/
validation/

⸻

Future Work

* Flood susceptibility mapping
* Spatial flood-risk products
* Additional Sentinel-1 validation events
* Hyperparameter optimization
* Automated testing and validation framework
* Streamlit dashboard deployment
* Integration of land-cover and soil-moisture predictors
* Multi-basin generalization experiments
* Real-time forecasting workflows