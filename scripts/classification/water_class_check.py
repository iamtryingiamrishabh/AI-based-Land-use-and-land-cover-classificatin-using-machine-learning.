import numpy as np
import rasterio

with rasterio.open(r"outputs\landcover_map_swir.tif") as m, \
     rasterio.open(r"outputs\rasters\NDWI.tif") as n:
    print("map shape:", m.shape, " ndwi shape:", n.shape)
    if m.shape != n.shape:
        raise SystemExit("Grids differ; tell me and I'll resample.")
    cls, ndwi = m.read(1), n.read(1).astype(float)

w = ndwi[(cls == 3) & np.isfinite(ndwi)]
print("Water pixels:", w.size)
print("NDWI percentiles 10/50/90:", np.percentile(w, [10, 50, 90]).round(3))
print("Share with NDWI > 0:", round(float((w > 0).mean()), 3))