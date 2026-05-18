import rasterio
from pathlib import Path

srtm_dir = Path("data_raw/srtm")

first_file = list(srtm_dir.glob("*.tif"))[0]

print("Testing:", first_file)

with rasterio.open(first_file) as src:
    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Shape:", src.shape)
