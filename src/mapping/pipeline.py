from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.mapping.classification import validate_thresholds
from src.mapping.configuration import (
    DEFAULT_DATASET_TYPE,
    DEFAULT_SPATIAL_FEATURE_PATH,
    DEFAULT_SPATIAL_MODEL_NAME,
    DEFAULT_THRESHOLDS,
    SUPPORTED_HORIZONS,
    SUPPORTED_MODEL_ARCHITECTURES,
    get_horizon_config,
    get_spatial_config,
    output_dir_for_horizon,
)
from src.mapping.inference import atomic_write_csv, atomic_write_json, run_tabular_inference
from src.mapping.raster_export import (
    PROBABILITY_NODATA,
    RISK_NODATA,
    reconstruct_regular_grid,
    spatial_metadata_for_grid,
    values_to_grid,
    write_geotiff,
)
from src.mapping.visualization import plot_probability_map, plot_risk_map

try:
    from src.paths import PROCESSED_BOUNDARIES_DIR
except ModuleNotFoundError:  # pragma: no cover
    from paths import PROCESSED_BOUNDARIES_DIR


DEFAULT_BOUNDARY_PATH = PROCESSED_BOUNDARIES_DIR / "sindh_boundary.geojson"
MATERIAL_METADATA_KEYS = (
    "forecast_horizon",
    "dataset_type",
    "model_name",
    "model_artifact",
    "input_dataset",
    "model_sha256",
    "input_dataset_sha256",
    "risk_thresholds",
    "feature_count",
    "prediction_row_count",
    "dropped_non_finite_rows",
    "spatial_output_status",
    "crs",
    "affine_transform",
    "raster_dimensions",
    "resolution",
    "nodata_value",
)


def _guard_material_overwrite(metadata_path: Path, new_metadata: dict[str, object], overwrite: bool) -> None:
    if overwrite or not metadata_path.exists():
        return
    with metadata_path.open() as handle:
        previous = json.load(handle)
    changed = [
        key
        for key in MATERIAL_METADATA_KEYS
        if previous.get(key) != new_metadata.get(key)
    ]
    if changed:
        raise FileExistsError(
            f"Existing metadata at {metadata_path} differs for {changed}. "
            "Use --overwrite to replace outputs from a materially different configuration."
        )


def _parse_thresholds(values: list[float] | None) -> tuple[float, float, float]:
    if values is None:
        return validate_thresholds(DEFAULT_THRESHOLDS)
    return validate_thresholds(tuple(values))  # type: ignore[arg-type]


def run_horizon(
    horizon: str,
    thresholds: tuple[float, float, float],
    dataset_type: str = DEFAULT_DATASET_TYPE,
    output_root: Path | None = None,
    raster_crs: str | None = None,
    boundary_path: Path | None = DEFAULT_BOUNDARY_PATH,
    drop_invalid_rows: bool = False,
    overwrite: bool = False,
    model_architecture: str = "temporal",
    spatial_model_name: str = DEFAULT_SPATIAL_MODEL_NAME,
    spatial_dataset_path: Path = DEFAULT_SPATIAL_FEATURE_PATH,
) -> dict[str, object]:
    if model_architecture not in SUPPORTED_MODEL_ARCHITECTURES:
        raise ValueError(f"Unsupported model architecture: {model_architecture}")
    config = (
        get_spatial_config(spatial_model_name, spatial_dataset_path)
        if model_architecture == "spatial"
        else get_horizon_config(horizon, dataset_type=dataset_type)
    )
    output_dir = (
        (output_root or config.output_dir.parent) / "spatial_event" / spatial_model_name
        if model_architecture == "spatial"
        else output_dir_for_horizon(horizon, output_root or config.output_dir.parent)
    )

    result = run_tabular_inference(
        config,
        thresholds,
        {},
        drop_invalid_rows=drop_invalid_rows,
        allow_non_finite_features=model_architecture == "spatial",
    )

    retained_rows = result.retained_rows
    has_multiple_spatial_dates = (
        model_architecture == "spatial"
        and "date" in retained_rows.columns
        and retained_rows["date"].nunique() > 1
    )
    if has_multiple_spatial_dates:
        grid = None
        spatial_metadata = spatial_metadata_for_grid(None)
        spatial_metadata["spatial_output_reason"] = (
            "Spatial feature-grid contains multiple dates, so coordinate pairs repeat. "
            "Tabular predictions are written; provide a single-date spatial parquet for GeoTIFF export."
        )
    else:
        grid = (
            reconstruct_regular_grid(retained_rows, crs=raster_crs)
            if raster_crs
            else reconstruct_regular_grid(retained_rows)
        )
        spatial_metadata = spatial_metadata_for_grid(grid)
    result.metadata.update(spatial_metadata)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    metadata_path = output_dir / "metadata.json"
    _guard_material_overwrite(metadata_path, result.metadata, overwrite)
    atomic_write_csv(result.predictions, predictions_path)

    generated = {
        "predictions_csv": str(predictions_path),
        "metadata_json": str(metadata_path),
        "probability_geotiff": None,
        "risk_geotiff": None,
        "probability_png": None,
        "risk_png": None,
    }

    if grid is not None:
        probability_grid = values_to_grid(
            retained_rows,
            result.probabilities.astype("float32"),
            grid,
        )
        risk_grid = values_to_grid(retained_rows, result.risk_classes.astype("uint8"), grid)
        probability_path = write_geotiff(
            probability_grid,
            grid,
            output_dir / "probability_map.tif",
            dtype="float32",
            nodata=PROBABILITY_NODATA,
        )
        risk_path = write_geotiff(
            risk_grid,
            grid,
            output_dir / "risk_map.tif",
            dtype="uint8",
            nodata=RISK_NODATA,
        )
        probability_png = plot_probability_map(
            probability_grid, grid, output_dir / "probability_map.png", horizon, boundary_path
        )
        risk_png = plot_risk_map(risk_grid, grid, output_dir / "risk_map.png", horizon, boundary_path)
        generated.update(
            {
                "probability_geotiff": str(probability_path),
                "risk_geotiff": str(risk_path),
                "probability_png": str(probability_png),
                "risk_png": str(risk_png),
            }
        )

    result.metadata["generated_outputs"] = generated
    atomic_write_json(result.metadata, metadata_path)
    return result.metadata


def run_pipeline(
    horizons: list[str],
    thresholds: tuple[float, float, float],
    dataset_type: str = DEFAULT_DATASET_TYPE,
    output_root: Path | None = None,
    raster_crs: str | None = None,
    drop_invalid_rows: bool = False,
    overwrite: bool = False,
    model_architecture: str = "temporal",
    spatial_model_name: str = DEFAULT_SPATIAL_MODEL_NAME,
    spatial_dataset_path: Path = DEFAULT_SPATIAL_FEATURE_PATH,
) -> list[dict[str, object]]:
    return [
        run_horizon(
            horizon=horizon,
            thresholds=thresholds,
            dataset_type=dataset_type,
            output_root=output_root,
            raster_crs=raster_crs,
            drop_invalid_rows=drop_invalid_rows,
            overwrite=overwrite,
            model_architecture=model_architecture,
            spatial_model_name=spatial_model_name,
            spatial_dataset_path=spatial_dataset_path,
        )
        for horizon in horizons
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic multi-horizon flood probability inference and mapping."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all-horizons", action="store_true", help="Run 1-day, 7-day, and 14-day horizons.")
    group.add_argument("--horizon", choices=SUPPORTED_HORIZONS, help="Run one forecast horizon.")
    parser.add_argument(
        "--thresholds",
        nargs=3,
        type=float,
        metavar=("LOW_MODERATE", "MODERATE_HIGH", "HIGH_VERY_HIGH"),
        help="Three strictly increasing probability thresholds in [0, 1].",
    )
    parser.add_argument(
        "--dataset-type",
        choices=("hydrology", "rainfall_only"),
        default=DEFAULT_DATASET_TYPE,
        help="Model/dataset family to use. Defaults to hydrology XGBoost.",
    )
    parser.add_argument(
        "--model-architecture",
        choices=SUPPORTED_MODEL_ARCHITECTURES,
        default="temporal",
        help="Use existing temporal horizon models or Phase 13 spatial cell-level models.",
    )
    parser.add_argument(
        "--spatial-model-name",
        choices=("logistic_regression", "random_forest", "xgboost"),
        default=DEFAULT_SPATIAL_MODEL_NAME,
        help="Spatial model artifact to load when --model-architecture spatial is selected.",
    )
    parser.add_argument(
        "--spatial-dataset-path",
        type=Path,
        default=DEFAULT_SPATIAL_FEATURE_PATH,
        help="Spatial feature-grid parquet used when --model-architecture spatial is selected.",
    )
    parser.add_argument("--output-root", type=Path, help="Override output/flood_risk_maps root.")
    parser.add_argument(
        "--raster-crs",
        help="CRS for a future complete point-grid prediction CSV. Omitted for current aggregate datasets.",
    )
    parser.add_argument(
        "--drop-invalid-rows",
        action="store_true",
        help="Drop rows with non-finite required model features and record the count in metadata.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing outputs from a materially different recorded configuration.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        thresholds = _parse_thresholds(args.thresholds)
        horizons = list(SUPPORTED_HORIZONS) if args.all_horizons else [args.horizon]
        if args.model_architecture == "spatial":
            horizons = ["spatial_event"]

        print("Running spatial flood-risk inference without model retraining.")
        print(f"Model architecture: {args.model_architecture}")
        print(f"Dataset type: {args.dataset_type}")
        print(f"Horizons: {', '.join(horizons)}")
        print(f"Risk thresholds: {thresholds}")
        metadata = run_pipeline(
            horizons=horizons,
            thresholds=thresholds,
            dataset_type=args.dataset_type,
            output_root=args.output_root,
            raster_crs=args.raster_crs,
            drop_invalid_rows=args.drop_invalid_rows,
            overwrite=args.overwrite,
            model_architecture=args.model_architecture,
            spatial_model_name=args.spatial_model_name,
            spatial_dataset_path=args.spatial_dataset_path,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    for item in metadata:
        status = item.get("spatial_output_status")
        print(f"  {item['forecast_horizon']}: wrote predictions and metadata; spatial outputs={status}")


if __name__ == "__main__":
    main()
