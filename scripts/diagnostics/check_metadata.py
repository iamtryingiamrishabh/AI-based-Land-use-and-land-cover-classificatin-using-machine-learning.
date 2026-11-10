
import rasterio

file_path = "T43RGM_20260910T052641_B04_20m.jp2"

with rasterio.open(file_path) as src:
    print("Driver:", src.driver)
    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Transform:", src.transform)
    print("Width:", src.width)
    print("Height:", src.height)
    print("NoData:", src.nodata)
    print("Metadata tags:", src.tags())
