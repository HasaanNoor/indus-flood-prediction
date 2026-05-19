import os
import shutil
from pathlib import Path

import cdsapi


START_YEAR = int(os.getenv("START_YEAR", "2010"))
END_YEAR = int(os.getenv("END_YEAR", "2023"))

DATASET = "reanalysis-era5-single-levels"
OUTPUT_DIR = Path("data_raw/era5")
AREA = [31.0, 65.0, 22.0, 73.0]
VARIABLES = [
    "total_precipitation",
    "2m_temperature",
    "surface_runoff",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "evaporation",
    "potential_evaporation",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]


def year_range() -> range:
    if START_YEAR > END_YEAR:
        raise ValueError(f"START_YEAR ({START_YEAR}) must be <= END_YEAR ({END_YEAR})")
    return range(START_YEAR, END_YEAR + 1)


def build_request(year: int, month: int) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": AREA,
        "data_format": "netcdf",
    }


def main() -> None:
    client = cdsapi.Client()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    years = list(year_range())
    total_files = len(years) * 12
    print(
        f"Downloading ERA5 monthly files for {START_YEAR}-{END_YEAR} "
        f"({total_files} expected files)..."
    )

    for year in years:
        year_dir = OUTPUT_DIR / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nERA5 year {year}: writing to {year_dir}")

        for month in range(1, 13):
            month_str = f"{month:02d}"
            target = year_dir / f"era5_sindh_{year}_{month_str}.nc"
            legacy_target = OUTPUT_DIR / target.name

            if target.exists():
                print(f"  Skip {year}-{month_str}: already exists at {target}")
                continue

            if legacy_target.exists():
                shutil.copy2(legacy_target, target)
                print(
                    f"  Skip download {year}-{month_str}: copied existing legacy file "
                    f"from {legacy_target} to {target}"
                )
                continue

            print(f"  Downloading ERA5 for {year}-{month_str} -> {target}")
            try:
                client.retrieve(DATASET, build_request(year, month), str(target))
            except Exception as e:
                print(f"Failed download for {year}-{month_str}: {e}")
            print(f"  Saved: {target}")

    print("\nERA5 download pipeline complete.")


if __name__ == "__main__":
    main()
