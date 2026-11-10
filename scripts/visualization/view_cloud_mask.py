import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.enums import Resampling

# Read masked satellite image
with rasterio.open("outputs/rasters/masked_feature_stack.tif") as src:
    rgb = src.read(
        [3, 2, 1],
        out_shape=(3, 1200, 1200),
        resampling=Resampling.average,
        masked=True
    )

# Read cloud mask
with rasterio.open("outputs/rasters/masked_feature_stack.tif") as src:
    mask = src.read(
        1,
        out_shape=(1200, 1200),
        resampling=Resampling.nearest
    )

# Convert RGB bands into displayable values
rgb_display = np.zeros((1200, 1200, 3))

for i in range(3):
    band = rgb[i]
    valid_values = band.compressed()

    low, high = np.percentile(valid_values, [2, 98])

    stretched = np.clip(
        (band.filled(low) - low) / (high - low),
        0, 1
    )

    rgb_display[:, :, i] = stretched
    rgb_display[:, :, i][np.ma.getmaskarray(band)] = 0

# Display both images
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].imshow(rgb_display)
axes[0].set_title("Satellite Image After Cloud Masking")
axes[0].axis("off")

axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Cloud Mask (White = Usable, Black = Excluded)")
axes[1].axis("off")

plt.tight_layout()
plt.show()