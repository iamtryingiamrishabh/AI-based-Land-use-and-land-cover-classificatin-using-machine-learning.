"""Shared utilities for the separate LULC CNN pipeline.

The Random Forest pipeline is not modified by this module.

Set the environment variable LULC_SWIR=1 to add SWIR1 (B11) and SWIR2 (B12)
as two extra channels (8 channels instead of 6). Outputs then get a "_swir"
suffix so the 6-channel files are never overwritten.
"""
from pathlib import Path
import fnmatch
import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

# SWIR switch: PowerShell -> $env:LULC_SWIR = "1"
SWIR = os.environ.get("LULC_SWIR", "0") == "1"

# CNN class IDs are zero-based. RF class IDs remain independent.
CLASS_NAMES = [
    "Built-up",
    "Agriculture",
    "Vegetation",
    "Water",
]

# ESA WorldCover class code -> CNN class ID
CLASS_MAP = {
    50: 0,                  # Built-up
    40: 1,                  # Cropland
    10: 2, 20: 2, 30: 2,   # Tree cover, shrubland, grassland
    90: 2, 95: 2, 100: 2,  # Wetland, mangroves, moss/lichen
    80: 3,                  # Permanent water
}

FEATURE_NAMES = ["Blue", "Green", "Red", "NIR", "NDVI", "NDWI"]
if SWIR:
    FEATURE_NAMES = FEATURE_NAMES + ["SWIR1", "SWIR2"]

BOA_OFFSET = 1000
BOA_SCALE = 10000.0


def get_root():
    """Find the project root containing the data directory."""
    here = Path(__file__).resolve().parent
    for folder in (here, *here.parents):
        if (folder / "data").is_dir():
            return folder
    raise FileNotFoundError(
        "Could not find project root. Expected a 'data' folder "
        "above scripts/lulc_common.py."
    )


ROOT = get_root()
OUTPUT_DIR = ROOT / "outputs"
SUFFIX = "_swir" if SWIR else ""
PATCH_FILE = OUTPUT_DIR / f"patches_cnn{SUFFIX}.npz"
MODEL_FILE = OUTPUT_DIR / f"lulc_cnn{SUFFIX}.pt"


def find(pattern, required=True):
    """Find a file recursively under the project root."""
    skip = {
        ".git", ".venv", "venv", "env", "site-packages",
        "node_modules", "__pycache__"
    }

    matches = []
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [
            d for d in dirs
            if d.lower() not in skip and not d.startswith(".")
        ]
        for filename in files:
            if fnmatch.fnmatch(filename, pattern):
                matches.append(Path(folder) / filename)

    matches.sort()

    if matches:
        return matches[0]

    if required:
        raise FileNotFoundError(
            f"No file matching {pattern!r} found under {ROOT}"
        )
    return None


def find_band(name, target_resolution):
    """Find a Sentinel-2 band, preferring the target resolution."""
    satellite_dir = ROOT / "data" / "satellite"
    search_roots = [satellite_dir, ROOT]

    resolutions = [target_resolution]
    for r in (10, 20, 60):
        if r not in resolutions:
            resolutions.append(r)

    skip = {
        ".git", ".venv", "venv", "env", "site-packages",
        "node_modules", "__pycache__"
    }

    for resolution in resolutions:
        pattern = f"*{name}_{resolution}m*.jp2"

        for search_root in search_roots:
            if not search_root.exists():
                continue

            matches = []
            for folder, dirs, files in os.walk(search_root):
                dirs[:] = [
                    d for d in dirs
                    if d.lower() not in skip and not d.startswith(".")
                ]
                for filename in files:
                    if fnmatch.fnmatch(filename, pattern):
                        matches.append(Path(folder) / filename)

            if matches:
                matches.sort()
                return matches[0], resolution

    raise FileNotFoundError(
        f"Could not find Sentinel-2 band {name} at 10, 20 or 60 m."
    )


def open_grid(path):
    """Return reference raster grid information.

    Sentinel-2 T43RGM uses WGS 84 / UTM Zone 43N (EPSG:32643).
    Some JP2 files may lack CRS metadata, so use the known tile CRS.
    """
    with rasterio.open(path) as src:
        transform = src.transform
        crs = src.crs

        if crs is None:
            if Path(path).suffix.lower() == ".jp2":
                crs = rasterio.crs.CRS.from_epsg(32643)
                print(
                    "Warning: Reference JP2 has no CRS metadata. "
                    "Using verified tile CRS EPSG:32643."
                )
            else:
                raise ValueError(
                    f"Reference raster has no CRS: {path}"
                )

        height = src.height
        width = src.width

    return transform, crs, height, width


def read_to_grid(path, transform, crs, height, width,
                 row, col, chunk_height, chunk_width,
                 resampling="bilinear", dtype="float32"):
    """Read and align a raster window to the reference grid."""

    method = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
    }[resampling]

    window = Window(
        col_off=col,
        row_off=row,
        width=chunk_width,
        height=chunk_height,
    )

    with rasterio.open(path) as src:

        source_crs = src.crs

        # Sentinel-2 T43RGM JP2 files without CRS metadata
        if source_crs is None:
            if Path(path).suffix.lower() == ".jp2":
                source_crs = rasterio.crs.CRS.from_epsg(32643)
            else:
                raise ValueError(
                    f"Source raster has no CRS metadata: {path}"
                )

        with WarpedVRT(
            src,
            src_crs=source_crs,
            crs=crs,
            transform=transform,
            width=width,
            height=height,
            resampling=method,
        ) as vrt:

            arr = vrt.read(
                1,
                window=window,
                out_dtype=dtype,
                masked=True,
            )

            return arr.filled(np.nan)


def to_reflectance(dn):
    """Convert Sentinel-2 L2A DN to approximate surface reflectance.

    This assumes the 1000 DN offset used by processing baselines
    that apply the post-04.00 BOA offset. Verify your product metadata.
    """
    result = (dn.astype(np.float32) - BOA_OFFSET) / BOA_SCALE
    result[~np.isfinite(dn)] = np.nan
    return result


def window_fraction(mask, patch, stride):
    """Calculate the fraction of True pixels in every sliding window."""
    mask = np.asarray(mask, dtype=np.uint8)

    integral = np.pad(
        mask.cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0)),
        mode="constant",
    )

    total = (
        integral[patch::stride, patch::stride]
        - integral[:-patch:stride, patch::stride]
        - integral[patch::stride, :-patch:stride]
        + integral[:-patch:stride, :-patch:stride]
    )

    return total.astype(np.float32) / float(patch * patch)


def block_split(rows, cols, width, resolution,
                patch=32, block_m=10000, seed=42):
    """Split patches by geographic blocks, not by random patch.

    A half-patch buffer around block boundaries is excluded to reduce
    leakage caused by overlapping image patches.
    """
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    center_x = (cols + patch / 2.0) * resolution
    center_y = (rows + patch / 2.0) * resolution

    block_x = np.floor(center_x / block_m).astype(np.int64)
    block_y = np.floor(center_y / block_m).astype(np.int64)

    block_ids = block_y * 100000 + block_x
    unique_blocks = np.unique(block_ids)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_blocks)

    n = len(shuffled)
    n_train = max(1, int(n * 0.70))
    n_val = max(1, int(n * 0.15))

    train_blocks = set(shuffled[:n_train].tolist())
    val_blocks = set(shuffled[n_train:n_train + n_val].tolist())
    test_blocks = set(shuffled[n_train + n_val:].tolist())

    # Exclude patch centres too close to a 10 km block boundary.
    block_pixels = block_m / resolution
    margin = patch / 2.0

    x_inside = np.minimum(
        center_x % block_m,
        block_m - (center_x % block_m)
    ) >= margin * resolution

    y_inside = np.minimum(
        center_y % block_m,
        block_m - (center_y % block_m)
    ) >= margin * resolution

    safe = x_inside & y_inside

    train = np.array(
        [(b in train_blocks) for b in block_ids], dtype=bool
    ) & safe
    val = np.array(
        [(b in val_blocks) for b in block_ids], dtype=bool
    ) & safe
    test = np.array(
        [(b in test_blocks) for b in block_ids], dtype=bool
    ) & safe

    if not np.any(train) or not np.any(val) or not np.any(test):
        raise ValueError(
            "Geographical split produced an empty set. "
            "You may need more geographic coverage or fewer blocks."
        )

    return train, val, test