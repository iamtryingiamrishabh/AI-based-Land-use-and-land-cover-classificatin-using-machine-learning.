
import rasterio
import numpy as np

files = ["NDVI.tif", "NDWI.tif"]

for file in files:
    with rasterio.open(file) as src:
        data = src.read(1)

        print("\nFile:", file)
        print("Minimum:", np.min(data))
        print("Maximum:", np.max(data))
        print("Mean:", np.mean(data))
        print("Standard deviation:", np.std(data))
