
import rasterio

feature_file = "outputs/rasters/masked_feature_stack.tif"

scl_file = (
    "data/satellite/"
    "T43RGM_20260910T052641_SCL_20m.jp2"
)

with rasterio.open(feature_file) as feature, \
     rasterio.open(scl_file) as scl:

    print("\n--- FEATURE STACK ---")
    print("Width:", feature.width)
    print("Height:", feature.height)
    print("CRS:", feature.crs)
    print("Transform:", feature.transform)
    print("Bounds:", feature.bounds)

    print("\n--- SCL RASTER ---")
    print("Width:", scl.width)
    print("Height:", scl.height)
    print("CRS:", scl.crs)
    print("Transform:", scl.transform)
    print("Bounds:", scl.bounds)

    print("\n--- ALIGNMENT CHECK ---")
    print("Width match:", feature.width == scl.width)
    print("Height match:", feature.height == scl.height)
    print("CRS match:", feature.crs == scl.crs)
    print("Transform match:", feature.transform == scl.transform)
