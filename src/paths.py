from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_PROCESSED = PROJECT_ROOT / "data_processed"
OUTPUTS = PROJECT_ROOT / "outputs"

RAW_ERA5_DIR = DATA_RAW / "era5"
RAW_GLOFAS_DIR = DATA_RAW / "glofas"
RAW_SRTM_DIR = DATA_RAW / "srtm"
RAW_BOUNDARIES_DIR = DATA_RAW / "boundaries"

PROCESSED_ERA5_DIR = DATA_PROCESSED / "era5"
PROCESSED_GLOFAS_DIR = DATA_PROCESSED / "glofas"
PROCESSED_SRTM_DIR = DATA_PROCESSED / "srtm"
PROCESSED_BOUNDARIES_DIR = DATA_PROCESSED / "boundaries"
PROCESSED_FEATURES_DIR = DATA_PROCESSED / "features"
CLIPPED_DIR = DATA_PROCESSED / "clipped"
FIGURES_DIR = OUTPUTS / "figures"
MODELS_DIR = OUTPUTS / "models"
METRICS_DIR = OUTPUTS / "metrics"
VALIDATION_DIR = OUTPUTS / "validation"


def create_output_folders() -> None:
    """Create the standard preprocessing output folders."""
    output_dirs = [
        PROCESSED_ERA5_DIR,
        PROCESSED_GLOFAS_DIR,
        PROCESSED_SRTM_DIR,
        PROCESSED_BOUNDARIES_DIR,
        PROCESSED_FEATURES_DIR,
        CLIPPED_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        METRICS_DIR,
        VALIDATION_DIR,
    ]

    print("\nCreating preprocessing output folders...")
    for folder in output_dirs:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"  OK: {folder.relative_to(PROJECT_ROOT)}")
