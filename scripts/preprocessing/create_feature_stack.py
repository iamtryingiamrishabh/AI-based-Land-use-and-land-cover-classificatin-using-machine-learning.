
import rasterio

files = [
    "T43RGM_20260910T052641_B02_20m.jp2",
    "T43RGM_20260910T052641_B03_20m.jp2",
    "T43RGM_20260910T052641_B04_20m.jp2",
    "T43RGM_20260910T052641_B8A_20m.jp2",
    "NDVI.tif",
    "NDWI.tif"
]

band_names = [
    "Blue",
    "Green",
    "Red",
    "NIR",
    "NDVI",
    "NDWI"
]

with rasterio.open(files[0]) as blue, \
     rasterio.open(files[1]) as green, \
     rasterio.open(files[2]) as red, \
     rasterio.open(files[3]) as nir, \
     rasterio.open(files[4]) as ndvi, \
     rasterio.open(files[5]) as ndwi:

    profile = blue.profile.copy()

    profile.update(
        driver="GTiff",
        count=6,
        dtype="float32",
        compress="DEFLATE",
        predictor=3,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER"
    )

    with rasterio.open("feature_stack.tif", "w", **profile) as dst:

        datasets = [blue, green, red, nir, ndvi, ndwi]

        for _, window in blue.block_windows(1):

            for band_number, dataset in enumerate(datasets, start=1):

                data = dataset.read(1, window=window)

                dst.write(
                    data.astype("float32"),
                    band_number,
                    window=window
                )

        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)

print("Feature stack created successfully!")
