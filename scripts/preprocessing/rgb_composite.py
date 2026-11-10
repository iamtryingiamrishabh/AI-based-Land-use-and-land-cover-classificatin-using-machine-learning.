
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Satellite band file paths
red_path = "T43RGM_20260910T052641_B04_20m.jp2"
green_path = "T43RGM_20260910T052641_B03_20m.jp2"
blue_path = "T43RGM_20260910T052641_B02_20m.jp2"

# Read the three bands
with rasterio.open(red_path) as src:
    red = src.read(1)

with rasterio.open(green_path) as src:
    green = src.read(1)

with rasterio.open(blue_path) as src:
    blue = src.read(1)

# Normalize pixel values for display
def normalize(band):
    low = np.percentile(band, 2)
    high = np.percentile(band, 98)

    result = (band - low) / (high - low)
    return np.clip(result, 0, 1)

red = normalize(red)
green = normalize(green)
blue = normalize(blue)

# Combine bands into RGB image
rgb = np.dstack((red, green, blue))

# Display the composite
plt.figure(figsize=(12, 10))
plt.imshow(rgb)
plt.title("Sentinel-2 RGB Composite - Greater Noida")
plt.axis("off")
plt.show()
