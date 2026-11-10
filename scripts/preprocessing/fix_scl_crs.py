
import rasterio
from rasterio.crs import CRS

input_file = (
    "data/satellite/"
    "T43RGM_20260910T052641_SCL_20m.jp2"
)

output_file = "outputs/rasters/SCL_corrected.tif"

correct_crs = CRS.from_epsg(32643)

with rasterio.open(input_file) as src:

    profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        crs=correct_crs,
        compress="LZW"
    )

    with rasterio.open(output_file, "w", **profile) as dst:

        for _, window in src.block_windows(1):
            data = src.read(1, window=window)
            dst.write(data, 1, window=window)

print("SCL CRS corrected successfully!")
print("Output:", output_file)
