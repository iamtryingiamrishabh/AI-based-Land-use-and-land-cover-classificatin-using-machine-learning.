
import rasterio
import numpy as np

files = {
    "Blue": "T43RGM_20260910T052641_B02_20m.jp2",
    "Green": "T43RGM_20260910T052641_B03_20m.jp2",
    "Red": "T43RGM_20260910T052641_B04_20m.jp2",
    "NIR": "T43RGM_20260910T052641_B8A_20m.jp2",
    "NDVI": "NDVI.tif",
    "NDWI": "NDWI.tif"
}

for name, filename in files.items():

    with rasterio.open(filename) as src:
        data = src.read(1)

        print(f"\n--- {name} ---")
        print("Minimum:", np.min(data))
        print("Maximum:", np.max(data))
        print("Zero pixels:", np.count_nonzero(data == 0))
        print("Non-finite pixels:", np.count_nonzero(~np.isfinite(data)))

        if name in ["NDVI", "NDWI"]:
            print("-9999 pixels:", np.count_nonzero(data == -9999))
