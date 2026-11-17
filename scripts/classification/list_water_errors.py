import csv
import numpy as np
import rasterio
from rasterio.warp import transform

RASTER = r"outputs\landcover_map_swir.tif"
POINTS = r"outputs\validation_points_swir.csv"
WATER_CODE = 3   # set after the first run, e.g. 4

rows = list(csv.DictReader(open(POINTS, newline="")))
lons = [float(r["lon"]) for r in rows]
lats = [float(r["lat"]) for r in rows]

with rasterio.open(RASTER) as src:
    print("Raster CRS:", src.crs)
    u, c = np.unique(src.read(1), return_counts=True)
    print("Class values and pixel counts:", dict(zip(u.tolist(), c.tolist())))
    xs, ys = transform("EPSG:4326", src.crs, lons, lats)
    vals = [int(v[0]) for v in src.sample(zip(xs, ys))]

# Map class at each point vs your label
from collections import Counter
print("\nMap code vs your label (counts):")
for k, n in sorted(Counter((v, r["true_class"]) for r, v in zip(rows, vals)).items()):
    print(k, n)

if WATER_CODE is not None:
    hits = [(r, v) for r, v in zip(rows, vals)
            if v == WATER_CODE and r["true_class"] == "Agriculture"]
    print(f"\nMapped Water but labelled Agriculture: {len(hits)}")
    for r, v in hits:
        print(r["id"], r["lat"], r["lon"])
    with open(r"outputs\water_vs_agri_review.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "lon", "lat", "true_class", "notes"])
        for r, v in hits:
            w.writerow([r["id"], r["lon"], r["lat"], "", ""])