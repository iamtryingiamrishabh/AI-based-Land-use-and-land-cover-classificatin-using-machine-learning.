"""Classify the full Sentinel-2 tile with the trained CNN.

Output: outputs/landcover_map[_swir].tif
  band 1 = class id (0 Built-up, 1 Agriculture, 2 Vegetation, 3 Water, 255 nodata)
  band 2 = confidence (max softmax probability, 0-100)

Usage (from D:\\major_project):
    $env:LULC_SWIR = "1"
    python .\\scripts\\classification\\predict_map.py
"""

import numpy as np
import rasterio
import torch

from lulc_common import (
    OUTPUT_DIR, PATCH_FILE, MODEL_FILE, CLASS_NAMES, FEATURE_NAMES,
    SWIR, SUFFIX, find, find_band, open_grid, read_to_grid, block_split,
)
from make_patches import (
    BANDS, TARGET_RESOLUTION, SCL_BAD, PATCH, STRIDE,
    dn_to_reflectance, make_feature_stack,
)
import train_cnn
from train_cnn import SmallCNN, get_normalisation_stats

MAP_FILE = OUTPUT_DIR / f"landcover_map{SUFFIX}.tif"
PREVIEW_FILE = OUTPUT_DIR / f"landcover_map{SUFFIX}.png"
ROWS_PER_CHUNK = 32
BATCH = 512
NODATA = 255

COLOURS = {
    0: (220, 50, 50),     # Built-up
    1: (240, 220, 80),    # Agriculture
    2: (40, 150, 60),     # Vegetation
    3: (40, 100, 220),    # Water
}


def to_np(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    return np.asarray(v, dtype=np.float32).reshape(-1)


def load_model_and_stats(device):
    n_ch = len(FEATURE_NAMES)
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"{MODEL_FILE} not found. Run train_cnn.py in the same "
            f"LULC_SWIR mode first.")

    ck = torch.load(MODEL_FILE, map_location="cpu", weights_only=False)
    mean = std = None

    if isinstance(ck, torch.nn.Module):
        model = ck
    else:
        state = ck
        if isinstance(ck, dict):
            for k in ("model_state_dict", "state_dict", "model_state", "model"):
                if k in ck and isinstance(ck[k], dict):
                    state = ck[k]
                    break
            if "mean" in ck and "std" in ck:
                mean, std = to_np(ck["mean"]), to_np(ck["std"])
        model = SmallCNN(n_ch, len(CLASS_NAMES))
        model.load_state_dict(state)

    model.to(device).eval()

    if mean is None:
        print("Normalisation stats not in checkpoint; recomputing from the "
              "training blocks.")
        with np.load(PATCH_FILE, allow_pickle=False) as d:
            X, rows, cols = d["X"], d["rows"], d["cols"]
            width, res = int(d["W"]), int(d["res_m"])
        seed = getattr(train_cnn, "SEED", 42)
        train_mask, _, _ = block_split(
            rows, cols, width, res, patch=PATCH, block_m=10000, seed=seed)
        mean, std = get_normalisation_stats(X, train_mask)
        mean, std = to_np(mean), to_np(std)

    if mean.size != n_ch:
        raise ValueError(f"Stats have {mean.size} values, expected {n_ch}.")
    return model, mean.reshape(-1, 1, 1), np.maximum(std.reshape(-1, 1, 1), 1e-6)


@torch.no_grad()
def predict_probs(model, batch, device):
    out = []
    for i in range(0, len(batch), BATCH):
        xb = torch.from_numpy(batch[i:i + BATCH]).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    print("MODE:", "6 + SWIR (8 channels)" if SWIR else "6 channels")
    model, mean, std = load_model_and_stats(device)

    ref_path, res = find_band("B04", TARGET_RESOLUTION)
    transform, crs, height, width = open_grid(ref_path)
    print("Grid:", height, "x", width, "|", res, "m |", crs)

    band_paths = {b: find_band(b, res)[0] for b in BANDS}

    scl_path = find(f"*SCL_{res}m*.jp2", required=False)
    if scl_path is None:
        for r in (10, 20, 60):
            scl_path = find(f"*SCL_{r}m*.jp2", required=False)
            if scl_path:
                break
    print("SCL:", scl_path)

    out = np.full((height, width), NODATA, dtype=np.uint8)
    conf = np.zeros((height, width), dtype=np.uint8)

    n_rows = (height - PATCH) // STRIDE + 1
    n_cols = (width - PATCH) // STRIDE + 1
    half = (PATCH - STRIDE) // 2
    n_chunks = -(-n_rows // ROWS_PER_CHUNK)

    for start in range(0, n_rows, ROWS_PER_CHUNK):
        k = min(ROWS_PER_CHUNK, n_rows - start)
        top = start * STRIDE
        ch = (k - 1) * STRIDE + PATCH

        arrays = {}
        for band, path in band_paths.items():
            raw = read_to_grid(path, transform, crs, height, width,
                               top, 0, ch, width,
                               resampling="bilinear", dtype="float32")
            arrays[band] = dn_to_reflectance(raw)

        stack = make_feature_stack(arrays)
        bad = ~np.isfinite(stack).all(axis=0)

        if scl_path:
            scl = read_to_grid(scl_path, transform, crs, height, width,
                               top, 0, ch, width,
                               resampling="nearest", dtype="float32")
            bad |= np.isin(scl, SCL_BAD)

        norm = np.nan_to_num((stack - mean) / std, nan=0.0).astype(np.float32)
        wins = np.lib.stride_tricks.sliding_window_view(
            norm, (PATCH, PATCH), axis=(1, 2))[:, ::STRIDE, ::STRIDE]

        for r in range(k):
            batch = np.ascontiguousarray(wins[:, r].transpose(1, 0, 2, 3))
            probs = predict_probs(model, batch, device)
            cls = probs.argmax(axis=1).astype(np.uint8)
            cf = (probs.max(axis=1) * 100).astype(np.uint8)

            blk = np.repeat(np.repeat(cls[None, :], STRIDE, axis=0), STRIDE, axis=1)
            cblk = np.repeat(np.repeat(cf[None, :], STRIDE, axis=0), STRIDE, axis=1)

            lr = r * STRIDE + half                      # local row in chunk
            sub_bad = bad[lr:lr + STRIDE, half:half + n_cols * STRIDE]
            blk = np.where(sub_bad, NODATA, blk).astype(np.uint8)
            cblk = np.where(sub_bad, 0, cblk).astype(np.uint8)

            y0 = top + lr
            out[y0:y0 + STRIDE, half:half + n_cols * STRIDE] = blk
            conf[y0:y0 + STRIDE, half:half + n_cols * STRIDE] = cblk

        print(f"Chunk {start // ROWS_PER_CHUNK + 1}/{n_chunks} done")

    profile = dict(driver="GTiff", height=height, width=width, count=2,
                   dtype="uint8", crs=crs, transform=transform,
                   nodata=NODATA, compress="lzw", tiled=True)
    with rasterio.open(MAP_FILE, "w", **profile) as dst:
        dst.write(out, 1)
        dst.write(conf, 2)
        dst.write_colormap(1, {k: v + (255,) for k, v in COLOURS.items()})
        dst.set_band_description(1, "class")
        dst.set_band_description(2, "confidence_percent")

    print("\nSaved:", MAP_FILE)
    valid = out != NODATA
    print(f"Classified pixels: {valid.sum():,} of {out.size:,} "
          f"({100 * valid.mean():.1f}%)")
    for i, name in enumerate(CLASS_NAMES):
        n = int((out == i).sum())
        print(f"{name:12s}: {n:,} px ({100 * n / max(valid.sum(), 1):.1f}%)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rgb = np.full((height, width, 3), 255, dtype=np.uint8)
        for k, v in COLOURS.items():
            rgb[out == k] = v
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.axis("off")
        plt.savefig(PREVIEW_FILE, dpi=150, bbox_inches="tight")
        print("Preview:", PREVIEW_FILE)
    except Exception as e:
        print("Preview skipped:", e)


if __name__ == "__main__":
    main()