import rasterio
import numpy as np

scl_file = "T43RGM_20260910T052641_SCL_20m.jp2"

with rasterio.open(scl_file) as src:
    scl = src.read(1)

    print("Image dimensions:", src.width, "x", src.height)
    print("CRS:", src.crs)
    print("\nSCL class counts:")

    classes, counts = np.unique(scl, return_counts=True)

    for class_value, count in zip(classes, counts):
        percentage = (count / scl.size) * 100

        print(
            f"Class {class_value}: "
            f"{count:,} pixels ({percentage:.2f}%)"
        )