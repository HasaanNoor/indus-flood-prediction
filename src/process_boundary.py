from pathlib import Path

import geopandas as gpd

from paths import PROJECT_ROOT, RAW_BOUNDARIES_DIR, PROCESSED_BOUNDARIES_DIR


def _find_boundary_file(raw_dir: Path) -> Path:
    candidates = sorted(raw_dir.glob("*ADM1*.shp")) + sorted(raw_dir.glob("*.shp"))
    if not candidates:
        raise FileNotFoundError(f"No shapefile found in {raw_dir}")
    return candidates[0]


def process_boundary(
    raw_dir: Path = RAW_BOUNDARIES_DIR,
    output_path: Path = PROCESSED_BOUNDARIES_DIR / "sindh_boundary.geojson",
) -> Path:
    print("\nProcessing Sindh boundary...")
    boundary_path = _find_boundary_file(raw_dir)
    print(f"  Reading: {boundary_path.relative_to(PROJECT_ROOT)}")

    gdf = gpd.read_file(boundary_path)
    if gdf.empty:
        raise ValueError("Boundary file is empty.")

    name_columns = ["shapeName", "NAME_1", "name", "Name", "province", "Province"]
    name_column = next((column for column in name_columns if column in gdf.columns), None)
    if name_column is None:
        raise ValueError(f"Could not find a province name column. Columns: {list(gdf.columns)}")

    sindh = gdf[gdf[name_column].astype(str).str.casefold() == "sindh"].copy()
    if sindh.empty:
        raise ValueError(f"Could not find Sindh in column '{name_column}'.")

    # EPSG:4326 is latitude/longitude WGS84, the common CRS for NetCDF grids
    # and GeoJSON web maps. Reprojecting keeps later clipping/alignment simpler.
    if sindh.crs is None:
        print("  Boundary CRS is missing. Assuming EPSG:4326.")
        sindh = sindh.set_crs("EPSG:4326")
    elif sindh.crs.to_epsg() != 4326:
        print(f"  Reprojecting from {sindh.crs} to EPSG:4326.")
        sindh = sindh.to_crs("EPSG:4326")
    else:
        print("  Boundary already in EPSG:4326.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sindh.to_file(output_path, driver="GeoJSON")
    print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")
    return output_path
