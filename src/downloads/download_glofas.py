import os
from pathlib import Path

import cdsapi


START_YEAR = int(os.getenv("START_YEAR", "2010"))
END_YEAR = int(os.getenv("END_YEAR", "2023"))

DATASET = "cems-glofas-historical"
OUTPUT_DIR = Path("data_raw/glofas")
AREA = [31.0, 65.0, 22.0, 73.0]


def year_range() -> range:
    if START_YEAR > END_YEAR:
        raise ValueError(f"START_YEAR ({START_YEAR}) must be <= END_YEAR ({END_YEAR})")
    return range(START_YEAR, END_YEAR + 1)


def build_request(year: int) -> dict:
    return {
        "system_version": ["version_4_0"],
        "hydrological_model": ["lisflood"],
        "product_type": ["consolidated"],
        "variable": ["river_discharge_in_the_last_24_hours"],
        "hyear": [str(year)],
        "hmonth": [f"{m:02d}" for m in range(1, 13)],
        "hday": [f"{d:02d}" for d in range(1, 32)],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def main() -> None:
    # Use EWDS, not CDS.
    client = cdsapi.Client(
        url="https://ewds.climate.copernicus.eu/api",
        key=os.getenv("EWDS_API_KEY", "72ef36c1-d3e2-42d9-b554-240250d79c84"),
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    years = list(year_range())
    print(
        f"Downloading GloFAS yearly files for {START_YEAR}-{END_YEAR} "
        f"({len(years)} expected files)..."
    )

    for year in years:
        target = OUTPUT_DIR / f"glofas_sindh_{year}.nc"

        if target.exists():
            print(f"  Skip {year}: already exists at {target}")
            continue

        print(f"  Downloading GloFAS for {year} -> {target}")
        client.retrieve(DATASET, build_request(year), str(target))
        print(f"  Saved: {target}")

    print("\nGloFAS download pipeline complete.")


if __name__ == "__main__":
    main()
