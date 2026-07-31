from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    from src.paths import MODELS_DIR, OUTPUTS, PROCESSED_FEATURES_DIR
    from src.spatial.configuration import SPATIAL_FEATURES_DIR
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    from paths import MODELS_DIR, OUTPUTS, PROCESSED_FEATURES_DIR
    from spatial.configuration import SPATIAL_FEATURES_DIR


FLOOD_RISK_MAPS_DIR = OUTPUTS / "flood_risk_maps"

SUPPORTED_HORIZONS = ("1day", "7day", "14day")
SUPPORTED_MODEL_ARCHITECTURES = ("temporal", "spatial")
DEFAULT_DATASET_TYPE = "hydrology"
DEFAULT_THRESHOLDS = (0.25, 0.50, 0.75)
DEFAULT_SPATIAL_MODEL_NAME = "xgboost"
DEFAULT_SPATIAL_FEATURE_PATH = SPATIAL_FEATURES_DIR / "spatial_features_2019-08-01_2019-08-15.parquet"

RISK_CLASSES: dict[int, str] = {
    1: "Low",
    2: "Moderate",
    3: "High",
    4: "Very High",
}


@dataclass(frozen=True)
class HorizonConfig:
    horizon: str
    label_column: str
    model_path: Path
    dataset_path: Path
    dataset_type: str = DEFAULT_DATASET_TYPE
    model_name: str = "xgboost"
    model_architecture: str = "temporal"

    @property
    def output_dir(self) -> Path:
        return FLOOD_RISK_MAPS_DIR / self.horizon


def label_for_horizon(horizon: str) -> str:
    days = horizon.replace("day", "d")
    return f"label_discharge_next_{days}_ge_q95"


def model_path_for_horizon(horizon: str, dataset_type: str = DEFAULT_DATASET_TYPE) -> Path:
    return MODELS_DIR / f"{dataset_type}_{label_for_horizon(horizon)}_xgboost.pkl"


def dataset_path_for_type(dataset_type: str = DEFAULT_DATASET_TYPE) -> Path:
    if dataset_type == "hydrology":
        return PROCESSED_FEATURES_DIR / "flood_features_hydrology.csv"
    if dataset_type == "rainfall_only":
        return PROCESSED_FEATURES_DIR / "flood_features_rainfall_only.csv"
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def spatial_model_path(model_name: str = DEFAULT_SPATIAL_MODEL_NAME) -> Path:
    return OUTPUTS / "spatial_models" / f"{model_name}.pkl"


def get_spatial_config(
    model_name: str = DEFAULT_SPATIAL_MODEL_NAME,
    dataset_path: Path = DEFAULT_SPATIAL_FEATURE_PATH,
) -> HorizonConfig:
    if model_name not in {"logistic_regression", "random_forest", "xgboost"}:
        raise ValueError(f"Unsupported spatial model: {model_name}")
    return HorizonConfig(
        horizon="spatial_event",
        label_column="observed_inundation_label",
        model_path=spatial_model_path(model_name),
        dataset_path=dataset_path,
        dataset_type="spatial",
        model_name=model_name,
        model_architecture="spatial",
    )


def get_horizon_config(
    horizon: str,
    dataset_type: str = DEFAULT_DATASET_TYPE,
    output_root: Path = FLOOD_RISK_MAPS_DIR,
) -> HorizonConfig:
    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(
            f"Unsupported horizon {horizon!r}. Expected one of: {', '.join(SUPPORTED_HORIZONS)}"
        )
    config = HorizonConfig(
        horizon=horizon,
        label_column=label_for_horizon(horizon),
        model_path=model_path_for_horizon(horizon, dataset_type),
        dataset_path=dataset_path_for_type(dataset_type),
        dataset_type=dataset_type,
        model_architecture="temporal",
    )
    if output_root != FLOOD_RISK_MAPS_DIR:
        return HorizonConfig(
            horizon=config.horizon,
            label_column=config.label_column,
            model_path=config.model_path,
            dataset_path=config.dataset_path,
            dataset_type=config.dataset_type,
            model_name=config.model_name,
            model_architecture=config.model_architecture,
        )
    return config


def output_dir_for_horizon(horizon: str, output_root: Path = FLOOD_RISK_MAPS_DIR) -> Path:
    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(f"Unsupported horizon: {horizon}")
    return output_root / horizon


def all_horizon_configs(dataset_type: str = DEFAULT_DATASET_TYPE) -> list[HorizonConfig]:
    return [get_horizon_config(horizon, dataset_type=dataset_type) for horizon in SUPPORTED_HORIZONS]
