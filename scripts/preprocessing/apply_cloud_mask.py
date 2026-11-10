import rasterio
import numpy as np

feature_file = "feature_stack.tif"
mask_file = "cloud_mask.tif"

with rasterio.open(feature_file) as src:
    features = src.read()
    profile = src.profile.copy()

with rasterio.open(mask_file) as src:
    mask = src.read(1)

# Set excluded pixels to -9999 in all six bands
features[:, mask == 0] = -9999

profile.update(
    dtype="float32",
    nodata=-9999,
    compress="DEFLATE"
)

with rasterio.open("masked_feature_stack.tif", "w", **profile) as dst:
    dst.write(features)

print("Masked feature stack created successfully!")
print("Output: masked_feature_stack.tif")
print("Number of bands:", features.shape[0])
print("Excluded pixels:", np.count_nonzero(mask == 0))