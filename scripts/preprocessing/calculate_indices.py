
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Read Red, Green and NIR bands
with rasterio.open("T43RGM_20260910T052641_B04_20m.jp2") as src:
    red = src.read(1).astype("float32")
    profile = src.profile.copy()

with rasterio.open("T43RGM_20260910T052641_B03_20m.jp2") as src:
    green = src.read(1).astype("float32")

with rasterio.open("T43RGM_20260910T052641_B8A_20m.jp2") as src:
    nir = src.read(1).astype("float32")

# Calculate NDVI
ndvi = np.divide(
    nir - red,
    nir + red,
    out=np.zeros_like(nir),
    where=(nir + red) != 0
)

# Calculate NDWI
ndwi = np.divide(
    green - nir,
    green + nir,
    out=np.zeros_like(nir),
    where=(green + nir) != 0
)

# Save NDVI
profile.update(
    driver="GTiff",
    count=1,
    dtype="float32",
    nodata=-9999
)

with rasterio.open("NDVI.tif", "w", **profile) as dst:
    dst.write(ndvi.astype("float32"), 1)

# Save NDWI
with rasterio.open("NDWI.tif", "w", **profile) as dst:
    dst.write(ndwi.astype("float32"), 1)

print("NDVI and NDWI calculated successfully!")

# Display both indices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
axes[0].set_title("NDVI - Vegetation Index")
axes[0].axis("off")

axes[1].imshow(ndwi, cmap="Blues", vmin=-1, vmax=1)
axes[1].set_title("NDWI - Water Index")
axes[1].axis("off")

plt.tight_layout()
plt.show()
