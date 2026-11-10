
import rasterio

files = {
    "Blue": "T43RGM_20260910T052641_B02_20m.jp2",
    "Green": "T43RGM_20260910T052641_B03_20m.jp2",
    "Red": "T43RGM_20260910T052641_B04_20m.jp2",
    "NIR": "T43RGM_20260910T052641_B8A_20m.jp2",
    "NDVI": "NDVI.tif",
    "NDWI": "NDWI.tif"
}

reference = None

for name, filename in files.items():

    with rasterio.open(filename) as src:

        print(f"\n--- {name} ---")
        print("Dimensions:", src.width, "x", src.height)
        print("CRS:", src.crs)
        print("Resolution:", src.res)
        print("NoData:", src.nodata)

        valid_pixels = (src.read_masks(1) > 0).sum()
        print("Valid pixels:", valid_pixels)

        current = (
            src.width,
            src.height,
            src.crs,
            src.transform
        )

        if reference is None:
            reference = current
            print("Reference raster")

        elif current == reference:
            print("Alignment: MATCH")

        else:
            print("Alignment: DOES NOT MATCH")
