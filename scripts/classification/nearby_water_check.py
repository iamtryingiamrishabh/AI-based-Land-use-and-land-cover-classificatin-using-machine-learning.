import csv
import numpy as np
import rasterio
from rasterio.warp import transform

MAP = r"outputs\landcover_map_swir.tif"
NDWI = r"outputs\rasters\NDWI.tif"
POINTS = r"outputs\validation_points_swir.csv"
WATER_CODE = 3
R = 5   # window half-size in pixels (5 px = about 100 m at 20 m pixels)

rows = list(csv.DictReader(open(POINTS, newline="")))
lons = [float(r["lon"]) for r in rows]
lats = [float(r["lat"]) for r in rows]

with rasterio.open(MAP) as m:
    xs, ys = transform("EPSG:4326", m.crs, lons, lats)
    code = np.array([v[0] for v in m.sample(zip(xs, ys))])

with rasterio.open(NDWI) as n:
    arr = n.read(1).astype(float)
    maxes = []
    for x, y in zip(xs, ys):
        r, c = n.index(x, y)
        w = arr[max(r-R, 0):r+R+1, max(c-R, 0):c+R+1]
        maxes.append(np.nanmax(w) if np.isfinite(w).any() else np.nan)
maxes = np.array(maxes)

labels = np.array([r["true_class"] for r in rows])
wrong = (code == WATER_CODE) & (labels == "Agriculture")
right = (code == WATER_CODE) & (labels == "Water")

for tag, mask in [("wrong (really Agriculture)", wrong), ("right (really Water)", right)]:
    x = maxes[mask]
    print(f"{tag}: n={mask.sum()}  median max-NDWI nearby={np.nanmedian(x):.3f}  "
          f"share with water nearby (>0): {np.nanmean(x > 0):.2f}")