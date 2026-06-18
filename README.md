# Indus Flood Prediction using Machine Learning

## Overview

This project develops a machine learning-based flood prediction framework for the Lower Indus River Basin in Sindh, Pakistan.

The system combines hydrological, meteorological, and topographical datasets to predict flood-related discharge events and analyze flood risk patterns over time.

The long-term goal is to create a scalable flood forecasting pipeline capable of supporting early warning systems in regions with limited ground-based monitoring infrastructure.

---

## Objectives

- Predict flood-related discharge events
- Build multi-horizon flood forecasting models
- Compare rainfall-only and hydrology-informed models
- Generate spatial flood-risk visualizations
- Validate predictions using Sentinel-1 satellite imagery
- Improve interpretability using explainable AI techniques

---

## Datasets Used

### ERA5 Reanalysis
Used for:
- rainfall
- temperature
- runoff
- soil moisture

Source:
Copernicus Climate Data Store

### GloFAS
Used for:
- river discharge measurements
- hydrological forecasting variables

Source:
Copernicus Emergency Management Service

### SRTM DEM
Used for:
- elevation
- terrain analysis
- floodplain characterization

Source:
USGS / NASA

### Administrative Boundaries
Used for:
- Sindh boundary clipping
- spatial masking

Source:
geoBoundaries

---

## Current Pipeline

### Data Acquisition
- Multi-year ERA5 download pipeline (2015–2023)
- Multi-year GloFAS download pipeline
- SRTM DEM tile mosaicking

### Preprocessing
- Spatial clipping to Sindh boundary
- Temporal aggregation
- Feature engineering
- DEM mosaicking
- NetCDF processing

### Feature Engineering
Current features include:
- rainfall totals
- rolling rainfall averages
- lagged rainfall variables
- discharge statistics
- lagged discharge variables
- terrain elevation metrics

### Machine Learning Models
Implemented:
- Logistic Regression
- Random Forest
- XGBoost

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## Current Outputs

Generated outputs include:

- precipitation maps
- discharge maps
- elevation maps
- ROC curves
- confusion matrices
- feature importance plots
- temporal rainfall/discharge visualizations

---

## Repository Structure

```text
data_raw/              # downloaded datasets
data_processed/        # cleaned and clipped datasets
outputs/               # figures, metrics, trained models
src/                   # preprocessing and ML scripts
