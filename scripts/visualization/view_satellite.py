
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.enums import Resampling

with rasterio.open("outputs/rasters/masked_feature_stack.tif") as src:

    # Read a smaller version of the image to save memory
    rgb = src.read(
        [3, 2, 1],
        out_shape=(3, 1200, 1200),
        resampling=Resampling.average
    )

    # False colour: NIR, Red, Green
    false_colour = src.read(
        [4, 3, 2],
        out_shape=(3, 1200, 1200),
        resampling=Resampling.average
    )


def stretch(image):
    result = np.zeros_like(image, dtype="float32")

    for i in range(3):
        low = np.percentile(image[i], 2)
        high = np.percentile(image[i], 98)

        result[i] = np.clip(
            (image[i] - low) / (high - low),
            0, 1
        )

    return np.moveaxis(result, 0, -1)


fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].imshow(stretch(rgb))
axes[0].set_title("True Colour (RGB)")
axes[0].axis("off")

axes[1].imshow(stretch(false_colour))
axes[1].set_title("False Colour (NIR-Red-Green)")
axes[1].axis("off")

plt.tight_layout()
plt.show()
