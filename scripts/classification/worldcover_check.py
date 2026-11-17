import csv
from collections import Counter
import numpy as np
import rasterio
from rasterio.warp import transform

MAP = r"outputs\landcover_map_swir.tif"
WC = r"ESA_WorldCover_10m_2021_v200_N27E075_Map.tif"
POINTS = r"outputs\validation_points_swir.csv"
WATER_CODE = 3

rows = list(csv.DictReader(open(POINTS, newline="")))
lons = [float(r["lon"]) for r in rows]
lats = [float(r["lat"]) for r in rows]

with rasterio.open(MAP) as m:
    xs, ys = transform("EPSG:4326", m.crs, lons, lats)
    code = np.array([v[0] for v in m.sample(zip(xs, ys))])

with rasterio.open(WC) as w:
    b = w.bounds
    inside = [(b.left <= x <= b.right and b.bottom <= y <= b.top)
              for x, y in zip(lons, lats)]
    wc = np.array([v[0] for v in w.sample(zip(lons, lats))])

labels = np.array([r["true_class"] for r in rows])
inside = np.array(inside)
for tag, mask in [("wrong (really Agriculture)", (code == WATER_CODE) & (labels == "Agriculture")),
                  ("right (really Water)", (code == WATER_CODE) & (labels == "Water"))]:
    print(f"{tag}: n={mask.sum()}, inside this WorldCover tile={int((mask & inside).sum())}")
    print("  WorldCover codes:", dict(Counter(wc[mask & inside].tolist())))