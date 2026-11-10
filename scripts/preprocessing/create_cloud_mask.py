import rasterio
import numpy as np

scl_file = "T43RGM_20260910T052641_SCL_20m.jp2"

with rasterio.open(scl_file) as src:
    scl = src.read(1)
    profile = src.profile.copy()

# Keep only pixels belonging to known classes
valid_classes = [2, 4, 5, 6]

# 1 = usable pixel, 0 = excluded pixel
mask = np.isin(scl, valid_classes).astype("uint8")

# Save the mask
profile.update(
    driver="GTiff",
    dtype="uint8",
    count=1,
    nodata=None,
    compress="LZW"
)

with rasterio.open("cloud_mask.tif", "w", **profile) as dst:
    dst.write(mask, 1)

total = mask.size
valid = np.count_nonzero(mask == 1)
excluded = total - valid

print("Cloud mask created successfully!")
print("Usable pixels:", f"{valid:,}")
print("Excluded pixels:", f"{excluded:,}")
print("Usable percentage:", round(valid / total * 100, 2), "%")
print("Excluded percentage:", round(excluded / total * 100, 2), "%")