import csv
import numpy as np
import rasterio
from rasterio.warp import transform

POINTS = r"outputs\validation_points_swir.csv"
MAP = r"outputs\landcover_map_swir.tif"
INDICES = {"NDVI": r"outputs\rasters\NDVI.tif",
           "NDWI": r"outputs\rasters\NDWI.tif"}
WATER_CODE = 3

rows = list(csv.DictReader(open(POINTS, newline="")))
lons = [float(r["lon"]) for r in rows]
lats = [float(r["lat"]) for r in rows]

def sample(path):
    with rasterio.open(path) as src:
        xs, ys = transform("EPSG:4326", src.crs, lons, lats)
        out = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype=float)
        if src.nodata is not None:
            out[out == src.nodata] = np.nan
        return out

code = sample(MAP)
labels = np.array([r["true_class"] for r in rows])
wrong = (code == WATER_CODE) & (labels == "Agriculture")
right = (code == WATER_CODE) & (labels == "Water")
print("wrong:", wrong.sum(), " right:", right.sum())

for name, path in INDICES.items():
    v = sample(path)
    print(f"\n{name}")
    for tag, m in [("mapped Water, really Agriculture", wrong),
                   ("mapped Water, really Water", right)]:
        x = v[m]
        print(f"  {tag}: median {np.nanmedian(x):.3f}  "
              f"min {np.nanmin(x):.3f}  max {np.nanmax(x):.3f}")