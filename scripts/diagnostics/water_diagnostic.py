import rasterio
import numpy as np

with rasterio.open("lulc_classification.tif") as p:
    prediction = p.read(1)

with rasterio.open(
    "T43RGM_20260910T052641_SCL_20m.jp2"
) as s:
    scl = s.read(1)

# Check that both rasters are aligned
print("Same dimensions:", prediction.shape == scl.shape)

# Find SCL classes underneath predicted water
water_pixels = prediction == 4
scl_under_water = scl[water_pixels]

classes, counts = np.unique(
    scl_under_water,
    return_counts=True
)

print("\nSCL CLASSES UNDER MODEL-PREDICTED WATER")
print("---------------------------------------")

for class_id, count in zip(classes, counts):
    print(
        f"SCL class {class_id}: {count:,} pixels "
        f"({count / len(scl_under_water) * 100:.2f}%)"
    )

print("\nMODEL WATER / SCL WATER OVERLAP")

overlap = np.count_nonzero(
    (prediction == 4) & (scl == 6)
)

print("Correctly overlapping water pixels:", overlap)
print("Total model-predicted water:", np.count_nonzero(water_pixels))