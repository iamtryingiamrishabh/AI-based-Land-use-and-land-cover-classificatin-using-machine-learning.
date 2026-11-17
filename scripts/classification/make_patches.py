"""Generate labelled image patches for the LULC CNN.

6 features by default (Blue, Green, Red, NIR, NDVI, NDWI).
With LULC_SWIR=1, SWIR1 (B11) and SWIR2 (B12) are appended (8 features).
"""

import math
import numpy as np

from lulc_common import (
    ROOT,
    OUTPUT_DIR,
    PATCH_FILE,
    CLASS_MAP,
    CLASS_NAMES,
    FEATURE_NAMES,
    BOA_OFFSET,
    BOA_SCALE,
    SWIR,
    find,
    find_band,
    open_grid,
    read_to_grid,
    window_fraction,
)

# ============================================================
# CONFIGURATION
# ============================================================

PATCH = 32
STRIDE = 16

PURITY = 0.80          # purity for Built-up / Agriculture / Vegetation
WATER_PURITY = 0.30    # water bodies are narrow, so a lower threshold
WATER_ID = CLASS_NAMES.index("Water")
MAX_BAD = 0.0

N_PER_CLASS = 2500
ROWS_PER_CHUNK = 32

TARGET_RESOLUTION = 20
SEED = 42

# Bands to read. B11/B12 are native 20 m, matching the reference grid.
BANDS = ["B02", "B03", "B04", "B8A"] + (["B11", "B12"] if SWIR else [])

# SCL classes always excluded.
# 0 no data, 1 saturated/defective, 8/9 cloud, 10 cirrus, 11 snow/ice
SCL_BAD = [0, 1, 8, 9, 10, 11]
# SCL class 3 (cloud shadow) is excluded EXCEPT where WorldCover says
# water, because SCL often mislabels dark water as shadow.
SCL_SHADOW = 3


# ============================================================
# WORLDCOVER FILE DISCOVERY
# ============================================================

def find_worldcover_files():
    folder = ROOT / "training_data" / "worldcover_2021"

    if not folder.exists():
        raise FileNotFoundError(f"WorldCover folder not found: {folder}")

    files = sorted(folder.glob("*WorldCover*Map*.tif"))

    if not files:
        raise FileNotFoundError("No WorldCover Map TIFF files found.")

    return files


# ============================================================
# WORLDCOVER MOSAIC
# ============================================================

def read_worldcover(paths, transform, crs, height, width, row, chunk_height):
    merged = np.zeros((chunk_height, width), dtype=np.uint8)

    for path in paths:
        arr = read_to_grid(
            path, transform, crs, height, width,
            row, 0, chunk_height, width,
            resampling="nearest", dtype="float32",
        )

        valid = np.isfinite(arr) & (arr > 0) & (merged == 0)
        merged[valid] = arr[valid].astype(np.uint8)

    return merged


# ============================================================
# REFLECTANCE (nodata-safe)
# ============================================================

def dn_to_reflectance(dn):
    """DN -> reflectance. DN 0 (nodata) and NaN become NaN."""
    result = (dn.astype(np.float32) - BOA_OFFSET) / BOA_SCALE
    result[~np.isfinite(dn) | (dn == 0)] = np.nan
    return np.clip(result, 0.0, 1.5)


# ============================================================
# FEATURE GENERATION
# ============================================================

def make_feature_stack(band_arrays):
    blue = band_arrays["B02"]
    green = band_arrays["B03"]
    red = band_arrays["B04"]
    nir = band_arrays["B8A"]  # matches the RF baseline

    ndvi_den = nir + red
    ndvi = np.divide(
        nir - red, ndvi_den,
        out=np.full_like(nir, np.nan, dtype=np.float32),
        where=np.abs(ndvi_den) > 1e-6,
    )

    ndwi_den = green + nir
    ndwi = np.divide(
        green - nir, ndwi_den,
        out=np.full_like(green, np.nan, dtype=np.float32),
        where=np.abs(ndwi_den) > 1e-6,
    )

    # Order must match FEATURE_NAMES in lulc_common.py
    layers = [blue, green, red, nir, ndvi, ndwi]
    if SWIR:
        layers += [band_arrays["B11"], band_arrays["B12"]]

    return np.stack(layers, axis=0).astype(np.float32)


# ============================================================
# MAIN PATCH GENERATION
# ============================================================

def main():
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nMODE:", "6 + SWIR (8 channels)" if SWIR else "6 channels")

    # 1. Reference grid ---------------------------------------
    red_path, actual_resolution = find_band("B04", TARGET_RESOLUTION)
    transform, crs, height, width = open_grid(red_path)

    print("\nREFERENCE GRID")
    print("------------------------------")
    print("Reference:", red_path)
    print("Dimensions:", height, "x", width)
    print("Resolution:", actual_resolution, "m")
    print("CRS:", crs)
    print("Transform:", transform)

    # 2. Sentinel-2 bands -------------------------------------
    band_paths = {}

    for band in BANDS:
        path, source_resolution = find_band(band, actual_resolution)
        band_paths[band] = path
        print(f"{band}: {path.name} (source resolution {source_resolution} m)")

    # 3. SCL cloud mask ---------------------------------------
    scl_path = find(f"*SCL_{actual_resolution}m*.jp2", required=False)

    if scl_path is None:
        for resolution in (10, 20, 60):
            scl_path = find(f"*SCL_{resolution}m*.jp2", required=False)
            if scl_path:
                break

    if scl_path:
        print("\nSCL cloud mask:", scl_path)
    else:
        print("\nWARNING: SCL not found. Only invalid feature values "
              "will be excluded.")

    # 4. WorldCover tiles -------------------------------------
    worldcover_paths = find_worldcover_files()

    print("\nWorldCover tiles:")
    for path in worldcover_paths:
        print(" -", path)

    # 5. Prepare ----------------------------------------------
    n_patch_rows = (height - PATCH) // STRIDE + 1
    n_chunks = math.ceil(n_patch_rows / ROWS_PER_CHUNK)
    cap_per_chunk = max(1, math.ceil(1.5 * N_PER_CLASS / n_chunks))

    n_classes = len(CLASS_NAMES)
    X_by_class = [[] for _ in range(n_classes)]
    rows_by_class = [[] for _ in range(n_classes)]
    cols_by_class = [[] for _ in range(n_classes)]

    candidate_counts = np.zeros(n_classes, dtype=np.int64)
    label_pixel_counts = np.zeros(n_classes, dtype=np.int64)
    usable_label_pixel_counts = np.zeros(n_classes, dtype=np.int64)

    print("\nPATCH GENERATION")
    print("------------------------------")
    print("Patch size:", PATCH)
    print("Stride:", STRIDE)
    print("Purity:", PURITY, "| Water purity:", WATER_PURITY)
    print("Maximum patches per class:", N_PER_CLASS)

    # 6. Process in chunks ------------------------------------
    for patch_row_start in range(0, n_patch_rows, ROWS_PER_CHUNK):

        chunk_patch_rows = min(ROWS_PER_CHUNK, n_patch_rows - patch_row_start)
        top = patch_row_start * STRIDE
        chunk_height = (chunk_patch_rows - 1) * STRIDE + PATCH

        # Spectral bands
        band_arrays = {}
        for band, path in band_paths.items():
            raw = read_to_grid(
                path, transform, crs, height, width,
                top, 0, chunk_height, width,
                resampling="bilinear", dtype="float32",
            )
            band_arrays[band] = dn_to_reflectance(raw)

        stack = make_feature_stack(band_arrays)
        bad = ~np.isfinite(stack).all(axis=0)

        # WorldCover -> CNN classes
        wc = read_worldcover(
            worldcover_paths, transform, crs,
            height, width, top, chunk_height,
        )

        mapped = np.full(wc.shape, -1, dtype=np.int16)
        for worldcover_code, class_id in CLASS_MAP.items():
            mapped[wc == worldcover_code] = class_id

        # SCL mask (shadow ignored over water)
        if scl_path:
            scl = read_to_grid(
                scl_path, transform, crs, height, width,
                top, 0, chunk_height, width,
                resampling="nearest", dtype="float32",
            )

            bad |= np.isin(scl, SCL_BAD)
            bad |= (scl == SCL_SHADOW) & (mapped != WATER_ID)

        # Diagnostic: labelled pixels
        for class_id in range(n_classes):
            class_pixels = mapped == class_id
            label_pixel_counts[class_id] += np.count_nonzero(class_pixels)
            usable_label_pixel_counts[class_id] += np.count_nonzero(
                class_pixels & ~bad
            )

        # Class fractions per patch
        fractions = np.stack(
            [window_fraction(mapped == c, PATCH, STRIDE)
             for c in range(n_classes)],
            axis=0,
        )

        bad_fraction = window_fraction(bad, PATCH, STRIDE)

        dominant_class = fractions.argmax(axis=0)
        dominant_fraction = fractions.max(axis=0)

        # Water uses a lower purity threshold
        water_ok = fractions[WATER_ID] >= WATER_PURITY
        dominant_class = np.where(water_ok, WATER_ID, dominant_class)

        valid_patches = (
            ((dominant_fraction >= PURITY) | water_ok)
            & (bad_fraction <= MAX_BAD)
        )

        # Sample patches per class
        for class_id in range(n_classes):

            locations = np.argwhere(valid_patches & (dominant_class == class_id))
            candidate_counts[class_id] += len(locations)

            if len(locations) == 0:
                continue

            if len(locations) > cap_per_chunk:
                selected = rng.choice(len(locations), size=cap_per_chunk,
                                      replace=False)
                locations = locations[selected]

            for patch_r, patch_c in locations:
                r0 = int(patch_r * STRIDE)
                c0 = int(patch_c * STRIDE)

                patch_data = stack[:, r0:r0 + PATCH, c0:c0 + PATCH]

                X_by_class[class_id].append(patch_data.astype(np.float16))
                rows_by_class[class_id].append(top + r0)
                cols_by_class[class_id].append(c0)

        completed = patch_row_start // ROWS_PER_CHUNK + 1
        print(f"Chunk {completed}/{n_chunks} completed")

    # 7. Combine and balance ----------------------------------
    X_final, y_final, rows_final, cols_final = [], [], [], []

    for class_id in range(n_classes):

        if not X_by_class[class_id]:
            continue

        class_X = np.stack(X_by_class[class_id], axis=0)
        class_rows = np.asarray(rows_by_class[class_id], dtype=np.int32)
        class_cols = np.asarray(cols_by_class[class_id], dtype=np.int32)

        if len(class_X) > N_PER_CLASS:
            selected = rng.choice(len(class_X), size=N_PER_CLASS, replace=False)
            class_X = class_X[selected]
            class_rows = class_rows[selected]
            class_cols = class_cols[selected]

        X_final.append(class_X)
        y_final.append(np.full(len(class_X), class_id, dtype=np.int64))
        rows_final.append(class_rows)
        cols_final.append(class_cols)

    if not X_final:
        raise RuntimeError(
            "No valid patches were generated. Check the WorldCover "
            "alignment, feature values, and cloud mask."
        )

    X = np.concatenate(X_final, axis=0).astype(np.float16)
    y = np.concatenate(y_final, axis=0)
    rows = np.concatenate(rows_final, axis=0)
    cols = np.concatenate(cols_final, axis=0)

    order = rng.permutation(len(y))
    X, y, rows, cols = X[order], y[order], rows[order], cols[order]

    # 8. Save -------------------------------------------------
    np.savez_compressed(
        PATCH_FILE,
        X=X,
        y=y,
        rows=rows,
        cols=cols,
        W=np.int32(width),
        H=np.int32(height),
        res_m=np.int32(actual_resolution),
        bands=np.asarray(FEATURE_NAMES),
        patch=np.int32(PATCH),
        stride=np.int32(STRIDE),
        purity=np.float32(PURITY),
        water_purity=np.float32(WATER_PURITY),
        cand_total=candidate_counts,
    )

    # 9. Report -----------------------------------------------
    print("\nCLASS PATCH COUNTS")
    print("------------------------------")

    for class_id, name in enumerate(CLASS_NAMES):
        saved = int(np.count_nonzero(y == class_id))
        candidates = int(candidate_counts[class_id])
        print(f"{name:12s}: {saved:,} saved / {candidates:,} candidates")

    print("\nLABEL COVERAGE DIAGNOSTIC")
    print("------------------------------")

    for class_id, name in enumerate(CLASS_NAMES):
        labelled = int(label_pixel_counts[class_id])
        usable = int(usable_label_pixel_counts[class_id])
        percent = (usable / labelled * 100) if labelled > 0 else 0.0
        print(f"{name:12s}: labelled={labelled:,}, usable={usable:,}, "
              f"usable={percent:.2f}%")

    print("\nPATCH DATASET CREATED")
    print("------------------------------")
    print("Saved to:", PATCH_FILE)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Data type:", X.dtype)
    print("Features:", FEATURE_NAMES)
    print("Classes:", CLASS_NAMES)
    print("Class distribution:", np.bincount(y, minlength=n_classes))


if __name__ == "__main__":
    main()