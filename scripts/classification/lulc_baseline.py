"""
LULC baseline: Sentinel-2 (T43RGM) + ESA WorldCover labels -> Random Forest.

Classes: 0 = forest, 1 = agricultural, 2 = built-up
Install: pip install rasterio geopandas numpy scikit-learn joblib

Run from your project folder after editing the CONFIG block:
    python lulc_baseline.py
"""
import os
import fnmatch
from pathlib import Path
import numpy as np
import rasterio
import joblib
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------------------------------ CONFIG
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # folder above scripts/
SEARCH_ROOT = PROJECT_ROOT    # or narrow it: Path(r"D:\major_project\data")


def find(pattern, required=True):
    """Find a file by name pattern anywhere under SEARCH_ROOT (skips venvs/hidden dirs)."""
    skip = {"site-packages", "venv", "env", "node_modules", "__pycache__"}
    for root, dirs, files in os.walk(SEARCH_ROOT):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d.lower() not in skip]
        for f in sorted(files):
            if fnmatch.fnmatch(f, pattern):
                return str(Path(root) / f)
    if required:
        raise FileNotFoundError(f"No file matching {pattern!r} under {SEARCH_ROOT}. "
                                "Move the file there or edit SEARCH_ROOT.")
    return None


S2 = {
    "B02": find("*B02_10m*.jp2"),
    "B03": find("*B03_10m*.jp2"),
    "B04": find("*B04_10m*.jp2"),
    "B08": find("*B08_10m*.jp2"),
}
# Optional 20 m L2A bands: used automatically if found, otherwise skipped.
B11 = find("*B11_20m*.jp2", required=False)
B12 = find("*B12_20m*.jp2", required=False)
SCL = find("*SCL_20m*.jp2", required=False)   # cloud mask

WORLDCOVER = find("*WorldCover*N27E075*Map*.tif")
AOI = None   # e.g. str(PROJECT_ROOT / "data" / "greater_noida.geojson"). None = whole tile.
print("Using:", S2, WORLDCOVER, "B11:", B11, "SCL:", SCL, sep="\n  ")

BOA_OFFSET = 1000      # L2A processing baseline >= 04.00 (your Sept 2026 scene)
CLASS_MAP = {10: 0, 40: 1, 50: 2}          # WorldCover code -> our class
CLASS_NAMES = ["forest", "agricultural", "built-up"]

BLOCK = 500            # px; 500 px = 5 km blocks for the spatial split
N_TRAIN = 30000        # max samples per class for training
N_TEST = 10000         # max samples per class for testing
TEST_FRACTION = 0.25   # share of blocks held out for testing
SEED = 42
PREDICT_MAP = True
OUT_MODEL = "lulc_rf.joblib"
OUT_MAP = "lulc_prediction.tif"
# -------------------------------------------------------------------------

rng = np.random.default_rng(SEED)

# ---- reference grid (the 10 m Sentinel-2 grid, optionally clipped to AOI)
with rasterio.open(S2["B04"]) as ref:
    crs = ref.crs
    full = Window(0, 0, ref.width, ref.height)
    aoi_geoms = None
    if AOI:
        import geopandas as gpd
        gdf = gpd.read_file(AOI).to_crs(crs)
        aoi_geoms = list(gdf.geometry)
        minx, miny, maxx, maxy = gdf.total_bounds
        r0, c0 = ref.index(minx, maxy)
        r1, c1 = ref.index(maxx, miny)
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1 + 1, ref.height), min(c1 + 1, ref.width)
        win = Window(c0, r0, c1 - c0, r1 - r0)
    else:
        win = full
    transform = ref.window_transform(win)
H, W = int(win.height), int(win.width)
print(f"Working grid: {H} x {W} px, CRS {crs}")


def read_10m(path):
    with rasterio.open(path) as src:
        return src.read(1, window=win)


def to_reflectance(dn):
    r = (dn.astype("float32") - BOA_OFFSET) / 10000.0
    r[dn == 0] = np.nan
    return np.clip(r, 0, 1.5)


def onto_grid(path, resampling, dtype):
    """Reproject any raster (other CRS / resolution) onto the working grid."""
    out = np.zeros((H, W), dtype=dtype)
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src, 1), destination=out,
                  dst_transform=transform, dst_crs=crs,
                  resampling=resampling, src_nodata=src.nodata, dst_nodata=0)
    return out


# ---- bands and indices
b02, b03, b04, b08 = (to_reflectance(read_10m(S2[k])) for k in ("B02", "B03", "B04", "B08"))
arrays = {"B02": b02, "B03": b03, "B04": b04, "B08": b08,
          "NDVI": (b08 - b04) / (b08 + b04 + 1e-6),
          "NDWI": (b03 - b08) / (b03 + b08 + 1e-6)}
if B11:
    b11 = to_reflectance(onto_grid(B11, Resampling.bilinear, "float32"))
    arrays["B11"] = b11
    arrays["NDBI"] = (b11 - b08) / (b11 + b08 + 1e-6)
    arrays["MNDWI"] = (b03 - b11) / (b03 + b11 + 1e-6)
if B12:
    arrays["B12"] = to_reflectance(onto_grid(B12, Resampling.bilinear, "float32"))
feat_names = list(arrays)
feats = [arrays[k].ravel() for k in feat_names]
print("Features:", feat_names)

# ---- cloud / validity mask
if SCL:
    scl = onto_grid(SCL, Resampling.nearest, "uint8")
    bad = np.isin(scl, [0, 1, 3, 8, 9, 10])          # nodata, defective, shadow, cloud, cirrus
else:
    print("WARNING: no SCL given, using a crude brightness cloud mask.")
    bad = np.isnan(b02) | (b02 > 0.20)
valid = ~bad
if aoi_geoms:
    valid &= geometry_mask(aoi_geoms, out_shape=(H, W), transform=transform, invert=True)

# ---- labels from WorldCover (nearest neighbour, EPSG:4326 -> UTM)
wc = onto_grid(WORLDCOVER, Resampling.nearest, "uint8")
label = np.full((H, W), -1, dtype="int8")
for code, k in CLASS_MAP.items():
    label[wc == code] = k
valid &= label >= 0
label = label.ravel()
valid = valid.ravel()
print("Valid labelled pixels per class:",
      {CLASS_NAMES[k]: int((valid & (label == k)).sum()) for k in range(3)})

# ---- spatial block split: whole 5 km blocks go to train OR test
nbc = -(-W // BLOCK)
nbr = -(-H // BLOCK)
n_test_blocks = max(1, int(TEST_FRACTION * nbr * nbc))
test_blocks = rng.choice(nbr * nbc, size=n_test_blocks, replace=False)

train_idx, test_idx = [], []
for k in range(3):
    idx = np.flatnonzero(valid & (label == k))
    blk = (idx // W // BLOCK) * nbc + (idx % W) // BLOCK
    is_test = np.isin(blk, test_blocks)
    tr, te = idx[~is_test], idx[is_test]
    print(f"{CLASS_NAMES[k]:13s} train pool {len(tr):>9d}   test pool {len(te):>9d}")
    train_idx.append(rng.choice(tr, min(N_TRAIN, len(tr)), replace=False))
    test_idx.append(rng.choice(te, min(N_TEST, len(te)), replace=False))
train_idx, test_idx = np.concatenate(train_idx), np.concatenate(test_idx)


def matrix(idx):
    return np.column_stack([f[idx] for f in feats])


Xtr, ytr = matrix(train_idx), label[train_idx]
Xte, yte = matrix(test_idx), label[test_idx]
ok_tr, ok_te = np.isfinite(Xtr).all(1), np.isfinite(Xte).all(1)
Xtr, ytr, Xte, yte = Xtr[ok_tr], ytr[ok_tr], Xte[ok_te], yte[ok_te]

# ---- train + evaluate
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                            n_jobs=-1, random_state=SEED)
rf.fit(Xtr, ytr)
pred = rf.predict(Xte)
print("\n=== Held-out spatial blocks (balanced test set) ===")
print(classification_report(yte, pred, target_names=CLASS_NAMES, digits=3))
print("Confusion matrix (rows = truth):\n", confusion_matrix(yte, pred))
print("\nFeature importance:")
for n, v in sorted(zip(feat_names, rf.feature_importances_), key=lambda t: -t[1]):
    print(f"  {n:6s} {v:.3f}")
joblib.dump({"model": rf, "features": feat_names, "classes": CLASS_NAMES}, OUT_MODEL)

# ---- optional: predict the whole working area and save a GeoTIFF
if PREDICT_MAP:
    out = np.full(H * W, 255, dtype="uint8")
    step = 2_000_000
    for s in range(0, H * W, step):
        sel = np.arange(s, min(s + step, H * W))
        X = np.column_stack([f[sel] for f in feats])
        good = np.isfinite(X).all(1) & ~bad.ravel()[sel]
        if good.any():
            out[sel[good]] = rf.predict(X[good]).astype("uint8")
    with rasterio.open(OUT_MAP, "w", driver="GTiff", height=H, width=W, count=1,
                       dtype="uint8", crs=crs, transform=transform, nodata=255,
                       compress="deflate") as dst:
        dst.write(out.reshape(H, W), 1)
    print(f"Saved {OUT_MAP} (0=forest, 1=agricultural, 2=built-up, 255=masked)")