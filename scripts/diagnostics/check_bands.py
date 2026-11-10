import rasterio

files = [
    "T43RGM_20260910T052641_B02_20m.jp2",
    "T43RGM_20260910T052641_B03_20m.jp2",
    "T43RGM_20260910T052641_B04_20m.jp2",
    "T43RGM_20260910T052641_B8A_20m.jp2"
]

for file in files:
    with rasterio.open(file) as src:
        print("\nFile:", file)
        print("Dimensions:", src.width, src.height)
        print("CRS:", src.crs)
        print("Resolution:", src.res)
        print("Data type:", src.dtypes)