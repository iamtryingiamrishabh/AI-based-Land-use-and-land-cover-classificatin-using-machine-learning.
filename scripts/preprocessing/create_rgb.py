
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Open satellite bands
with rasterio.open("T43RGM_20260910T052641_B04_20m.jp2") as red_src:
    red = red_src.read(1)
    profile = red_src.profile.copy()

with rasterio.open("T43RGM_20260910T052641_B03_20m.jp2") as green_src:
    green = green_src.read(1)

with rasterio.open("T43RGM_20260910T052641_B02_20m.jp2") as blue_src:
    blue = blue_src.read(1)

# Combine bands in RGB order
rgb = np.stack([red, green, blue])

# Save combined satellite image
profile.update(
    driver="GTiff",
    count=3,
    compress="deflate"
)

with rasterio.open("true_colour.tif", "w", **profile) as dst:
    dst.write(rgb)

print("RGB satellite image created successfully!")

# Improve brightness for visualisation
def stretch(band):
    low, high = np.percentile(band, (2, 98))
    return np.clip((band - low) / (high - low), 0, 1)

# Convert into RGB display image
rgb_display = np.dstack([
    stretch(red),
    stretch(green),
    stretch(blue)
])

# Display image
plt.figure(figsize=(10, 10))
plt.imshow(rgb_display)
plt.title("Greater Noida - Sentinel-2 True Colour Image")
plt.axis("off")
plt.show()
