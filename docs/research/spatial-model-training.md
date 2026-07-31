# Phase 13 Research Notes: Spatial Flood Model Training

## Repository Findings

The Phase 12 spatial feature grid is available at `data_processed/spatial/features/spatial_features_2019-08-01_2019-08-15.parquet`. It contains 3,600 rows: 240 Sindh grid cells for 15 daily dates. The canonical grid metadata reports EPSG:4326, 0.25 degree resolution, row-major ordering, and 240 in-province cells.

The available label inventory is limited. `data_processed/spatial/labels/label_availability_report.json` lists two Sentinel-1 masks, but both describe the same 2019 event: `2019_event1_threshold_24` and a versioned duplicate. The de-duplicated non-versioned label file has 240 cells, with 223 non-flood and 17 flood labels. The versioned file has similar but not identical counts and is not an independent event, so it is excluded from supervised training to avoid duplicated observations.

The Sentinel-1 validation script defines the event window as 2019-08-01 through 2019-08-15. Phase 13 binds the labels to the 2019-08-15 spatial feature slice as a same-event spatial inundation classifier. This is not a temporal forecast and must not be interpreted as cross-event generalization.

Rows with missing terrain elevation, slope, or relative elevation are rejected because those are required spatial predictors. `glofas_river_discharge_m3s_on_river_cell` is structurally missing outside cells containing a valid GloFAS river cell; it is retained and median-imputed inside each model pipeline alongside the explicit `has_glofas_river_cell` and `distance_to_glofas_river_km` features.

## Literature And Technical Sources Reviewed

- Spatial cross-validation literature warns that random cross-validation on spatially dependent data can underestimate predictive error because neighboring observations and residuals are not independent. Phase 13 therefore avoids random pixel splits and uses a contiguous spatial block holdout. Source: Roberts et al. (2017), "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure", https://doi.org/10.1111/ecog.02881
- Recent flood susceptibility studies commonly frame spatial flood mapping as binary classification over raster cells or sampled points using conditioning factors and flood-inventory labels; Random Forest and XGBoost are recurring baselines, and Sentinel-1-derived flood inventories require validation caveats. Source example: Remote Sensing 2025 Sentinel-1/geospatial ML flood susceptibility workflow, https://www.mdpi.com/2072-4292/17/20/3471
- XGBoost documentation recommends `scale_pos_weight = negative / positive` as a typical class-weighting value for imbalanced binary classification. Source: https://xgboost.readthedocs.io/en/latest/parameter.html
- scikit-learn documents `class_weight="balanced"` for Logistic Regression and `class_weight="balanced_subsample"` for Random Forest as inverse-frequency class weighting strategies. Sources: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html and https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- scikit-learn classification metrics include precision, recall, F1, ROC-AUC, confusion matrix, and average precision/PR-AUC. Average precision is appropriate for rare positives because it summarizes precision-recall behavior over thresholds. Sources: https://scikit-learn.org/stable/api/sklearn.metrics.html and https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.average_precision_score.html
- SHAP TreeExplainer supports XGBoost/tree ensemble explainability, and SHAP summary, bar, and dependence plots provide global feature ranking and feature-effect diagnostics. Sources: https://shap-community.readthedocs.io/en/latest/generated/shap.explainers.Tree.html and https://shap-community.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Fitting%20a%20Linear%20Simulation%20with%20XGBoost.html
- Recent XGBoost-SHAP flood susceptibility work uses SHAP to interpret spatial flood probabilities and conditioning factors, but interpretation remains associational. Source: https://ideas.repec.org/a/spr/nathaz/v122y2026i6d10.1007_s11069-025-07908-7.html

No external implementation code was copied.

## Training Objective

One training row represents one canonical grid cell on one date. The target is `observed_inundation_label` from the de-duplicated Sentinel-1 event mask. The models are new cell-level spatial classifiers and do not reuse province-level temporal models or temporal feature schemas.

## Split Strategy

The current labels do not support a temporal split or event-based split because only one de-duplicated Sentinel-1 event exists. Phase 13 therefore uses a deterministic spatial block holdout. Candidate blocks are evaluated and the selected block must keep both classes and at least two positive cells in train and test. Entire grid cells are assigned to either train or test, preventing identical cells from appearing in both sets.

This is the largest defensible supervised subset available, but it is still a pilot evaluation of spatial interpolation within one event, not a robust estimate of performance on future flood events.

## Class Imbalance

Flood cells are rare in the de-duplicated label file. The selected handling is:

- Logistic Regression: `class_weight="balanced"`
- Random Forest: `class_weight="balanced_subsample"`
- XGBoost: `scale_pos_weight = n_negative / n_positive`

No synthetic labels or oversampled evaluation rows are created.

## Models And Hyperparameters

Hyperparameters are centralized in `src/spatial/model_training.py` under `SPATIAL_MODEL_HYPERPARAMETERS`. All models use deterministic random seeds. Random Forest and XGBoost use `n_jobs=1` to improve reproducibility of serialized artifacts and predictions.

## Evaluation

Each model is evaluated on the spatial block holdout with precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrix. Accuracy is intentionally not used as a primary metric because class imbalance can make it misleading.

## Explainability

SHAP is generated for the trained XGBoost spatial model using the held-out spatial block. Outputs include summary, bar, and dependence plots. The dominant spatial-model features are compared with previous province-level SHAP findings in the implementation report:

- Prior temporal hydrology models were dominated by province-level GloFAS discharge features.
- The spatial model is expected to elevate cell-level terrain, distance-to-river, and local GloFAS/ERA5 predictors because the target varies by cell within one event.

## Limitations

- One Sentinel-1 event is not sufficient for generalizable spatial flood model claims.
- The Sentinel-1 threshold mask is candidate observed inundation, not perfect ground truth.
- Permanent water is not supplied as a separate raster in the repository labels.
- The split tests spatial holdout behavior inside one event only; it does not test future events.
- Spatial autocorrelation remains possible across block boundaries.
- SHAP explains model associations, not causality.
