from pathlib import Path

import rasterio
from rasterio.merge import merge

from paths import PROJECT_ROOT, RAW_SRTM_DIR, PROCESSED_SRTM_DIR


def process_srtm(
    raw_dir: Path = RAW_SRTM_DIR,
    output_path: Path = PROCESSED_SRTM_DIR / "srtm_sindh_mosaic.tif",
) -> Path:
    print("\nProcessing SRTM elevation tiles...")
    files = sorted(raw_dir.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No SRTM .tif files found in {raw_dir}")

    print(f"  Found {len(files)} GeoTIFF tiles.")
    datasets = []
    try:
        for path in files:
            print(f"  Reading: {path.relative_to(PROJECT_ROOT)}")
            datasets.append(rasterio.open(path))

        # rasterio.merge builds one continuous raster from adjacent or
        # overlapping tiles. We keep the native CRS/resolution and do not clip
        # yet, because clipping should be checked carefully against boundaries.
        mosaic, transform = merge(datasets)
        metadata = datasets[0].meta.copy()
        metadata.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "count": mosaic.shape[0],
                "compress": "lzw",
            }
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **metadata) as dst:
            dst.write(mosaic)
    finally:
        for dataset in datasets:
            dataset.close()

    print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")
    return output_path
