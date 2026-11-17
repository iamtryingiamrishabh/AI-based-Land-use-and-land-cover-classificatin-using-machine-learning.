"""Create independent validation points from the predicted map.

Outputs (in outputs/):
  validation_points[_swir].csv        <- you fill in true_class
  validation_points_key[_swir].csv    <- map class + confidence (don't peek!)
  validation_points[_swir].kml        <- open in Google Earth Pro

Usage:
    $env:LULC_SWIR = "1"
    python .\\scripts\\classification\\make_validation_points.py
"""

import csv

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

from lulc_common import OUTPUT_DIR, CLASS_NAMES, SUFFIX

MAP_FILE = OUTPUT_DIR / f"landcover_map{SUFFIX}.tif"
BLANK_CSV = OUTPUT_DIR / f"validation_points{SUFFIX}.csv"
KEY_CSV = OUTPUT_DIR / f"validation_points_key{SUFFIX}.csv"
KML_FILE = OUTPUT_DIR / f"validation_points{SUFFIX}.kml"

N_PER_CLASS = 50
MIN_SPACING_M = 200
SEED = 7
NODATA = 255


def main():
    rng = np.random.default_rng(SEED)

    with rasterio.open(MAP_FILE) as src:
        cls = src.read(1)
        conf = src.read(2)
        transform, crs = src.transform, src.crs
        res = abs(transform.a)

    spacing_px = MIN_SPACING_M / res
    points = []  # (id, map_class, row, col)

    for c, name in enumerate(CLASS_NAMES):
        idx = np.flatnonzero(cls == c)
        if len(idx) == 0:
            print(f"WARNING: no pixels mapped as {name}")
            continue
        idx = rng.permutation(idx)[:30000]
        rr, cc = np.divmod(idx, cls.shape[1])

        chosen = []
        for r, col in zip(rr, cc):
            if all((r - r2) ** 2 + (col - c2) ** 2 >= spacing_px ** 2
                   for r2, c2 in chosen):
                chosen.append((r, col))
                if len(chosen) == N_PER_CLASS:
                    break
        if len(chosen) < N_PER_CLASS:
            print(f"NOTE: only {len(chosen)} points found for {name}")
        for r, col in chosen:
            points.append((c, int(r), int(col)))

    points = [points[i] for i in rng.permutation(len(points))]  # blind order

    xs = [transform.c + (col + 0.5) * transform.a for _, _, col in points]
    ys = [transform.f + (r + 0.5) * transform.e for _, r, _ in points]
    lons, lats = warp_transform(crs, "EPSG:4326", xs, ys)

    with open(BLANK_CSV, "w", newline="", encoding="utf-8") as fb, \
         open(KEY_CSV, "w", newline="", encoding="utf-8") as fk:
        wb, wk = csv.writer(fb), csv.writer(fk)
        wb.writerow(["id", "lon", "lat", "true_class", "notes"])
        wk.writerow(["id", "map_class", "confidence"])
        for i, ((c, r, col), lon, lat) in enumerate(zip(points, lons, lats), 1):
            wb.writerow([i, f"{lon:.6f}", f"{lat:.6f}", "", ""])
            wk.writerow([i, c, int(conf[r, col])])

    with open(KML_FILE, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n')
        for i, (lon, lat) in enumerate(zip(lons, lats), 1):
            f.write(f"<Placemark><name>{i}</name><Point><coordinates>"
                    f"{lon:.6f},{lat:.6f},0</coordinates></Point></Placemark>\n")
        f.write("</Document></kml>\n")

    print(f"{len(points)} points written.")
    print("Label this file :", BLANK_CSV)
    print("Google Earth    :", KML_FILE)


if __name__ == "__main__":
    main()