from pathlib import Path

import rasterio
from rasterio.mask import mask

from clip_utils import DEFAULT_BOUNDARY_PATH, read_boundary
from paths import CLIPPED_DIR, PROJECT_ROOT, PROCESSED_SRTM_DIR


def clip_srtm_to_sindh(
    input_path: Path = PROCESSED_SRTM_DIR / "srtm_sindh_mosaic.tif",
    output_path: Path = CLIPPED_DIR / "srtm_sindh_clipped.tif",
    boundary_path: Path = DEFAULT_BOUNDARY_PATH,
) -> Path:
    print("\nClipping SRTM mosaic to Sindh boundary...")
    if not input_path.exists():
        raise FileNotFoundError(f"Input SRTM mosaic not found: {input_path}")

    boundary = read_boundary(boundary_path)
    with rasterio.open(input_path) as src:
        raster_crs = src.crs or "EPSG:4326"
        boundary_for_raster = boundary.to_crs(raster_crs)
        minx, miny, maxx, maxy = boundary_for_raster.total_bounds
        if (
            minx < src.bounds.left
            or maxx > src.bounds.right
            or miny < src.bounds.bottom
            or maxy > src.bounds.top
        ):
            print(
                "  Warning: SRTM mosaic does not fully cover the Sindh boundary; "
                "clipping the available raster extent only."
            )

        shapes = boundary_for_raster.geometry
        clipped, transform = mask(src, shapes, crop=True, all_touched=True, filled=True)
        metadata = src.meta.copy()
        metadata.update(
            {
                "driver": "GTiff",
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": transform,
                "compress": "lzw",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **metadata) as dst:
        dst.write(clipped)

    print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")
    return output_path


if __name__ == "__main__":
    clip_srtm_to_sindh()
