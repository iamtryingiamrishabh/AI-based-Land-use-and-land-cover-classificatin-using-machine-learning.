
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# File paths
nir_path = "T43RGM_20260910T052641_B8A_20m.jp2"
red_path = "T43RGM_20260910T052641_B04_20m.jp2"

# Read NIR band
with rasterio.open(nir_path) as src:
    nir = src.read(1).astype("float32")
    nir_nodata = src.nodata

# Read Red band
with rasterio.open(red_path) as src:
    red = src.read(1).astype("float32")
    red_nodata = src.nodata

# Check dimensions
if nir.shape != red.shape:
    raise ValueError("Both satellite bands must have the same dimensions")

# Calculate NDVI
denominator = nir + red

ndvi = np.full(nir.shape, np.nan, dtype="float32")

valid = denominator != 0

if nir_nodata is not None:
    valid &= nir != nir_nodata

if red_nodata is not None:
    valid &= red != red_nodata

ndvi[valid] = (nir[valid] - red[valid]) / denominator[valid]

# Print results
print("NDVI calculation completed")
print("Minimum NDVI:", np.nanmin(ndvi))
print("Maximum NDVI:", np.nanmax(ndvi))
print("Mean NDVI:", np.nanmean(ndvi))

# Display NDVI map
plt.figure(figsize=(12, 9))

plt.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)

plt.colorbar(label="NDVI Value")
plt.title("NDVI Vegetation Map - Greater Noida")
plt.axis("off")

plt.savefig("ndvi_map.png", dpi=300, bbox_inches="tight")
plt.show()
