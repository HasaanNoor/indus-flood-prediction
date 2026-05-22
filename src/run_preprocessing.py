from inspect_raw_data import inspect_all_raw_data
from paths import create_output_folders
from process_boundary import process_boundary
from process_era5 import process_era5
from process_glofas import process_glofas
from process_srtm import process_srtm
from preprocessing_validation import write_preprocessing_report


def main() -> None:
    print("Starting Lower Indus / Sindh preprocessing pipeline.")

    inspect_all_raw_data()
    create_output_folders()
    process_boundary()
    era5_path = process_era5()
    glofas_path = process_glofas()
    process_srtm()
    write_preprocessing_report(
        {
            "ERA5 daily multiyear": era5_path,
            "GloFAS multiyear": glofas_path,
        }
    )

    print("\nPreprocessing pipeline finished.")


if __name__ == "__main__":
    main()
