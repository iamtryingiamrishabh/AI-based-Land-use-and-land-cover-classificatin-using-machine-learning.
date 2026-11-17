"""Paired comparison: CNN with 6 channels vs CNN with 6 + SWIR (8 channels).

Reads outputs/cnn_multi_seed.csv and outputs/cnn_multi_seed_swir.csv,
both produced by run_multi_seed.py (the second with LULC_SWIR=1).

Usage (from D:\\major_project):
    python .\\scripts\\classification\\compare_swir.py
"""

import csv

import numpy as np

# Importing lulc_common with LULC_SWIR unset is fine: only OUTPUT_DIR and
# CLASS_NAMES are used here.
from lulc_common import OUTPUT_DIR, CLASS_NAMES

BASE_CSV = OUTPUT_DIR / "cnn_multi_seed.csv"
SWIR_CSV = OUTPUT_DIR / "cnn_multi_seed_swir.csv"


def load(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[int(r["seed"])] = {
                "acc": float(r["test_acc"]),
                "macro_f1": float(r["test_macro_f1"]),
                **{n: (float(r[f"f1_{n}"]) if r[f"f1_{n}"] != "" else np.nan)
                   for n in CLASS_NAMES},
            }
    return out


def ms(v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():.3f} +/- {sd:.3f}"


def main():
    base, swir = load(BASE_CSV), load(SWIR_CSV)
    seeds = sorted(set(base) & set(swir))
    if not seeds:
        raise RuntimeError("No common seeds in the two CSV files.")

    L = ["CNN: 6 channels vs 8 channels (+SWIR1, SWIR2)", "=" * 48,
         f"Seeds: {seeds}\n",
         f"{'metric':14s}{'6-channel':>18}{'8-channel':>18}{'mean diff':>12}{'SWIR wins':>11}"]

    for key, label in [("acc", "Accuracy"), ("macro_f1", "Macro F1")] + \
                      [(n, f"F1 {n}") for n in CLASS_NAMES]:
        b = np.array([base[s][key] for s in seeds])
        w = np.array([swir[s][key] for s in seeds])
        d = w - b
        ok = ~np.isnan(d)
        L.append(f"{label:14s}{ms(b):>18}{ms(w):>18}"
                 f"{d[ok].mean():>+12.3f}{int((d[ok] > 0).sum()):>8}/{int(ok.sum())}")

    L.append("\nPer-seed macro F1")
    L.append(f"{'seed':>5}{'6-ch':>8}{'8-ch':>8}{'diff':>8}")
    for s in seeds:
        L.append(f"{s:>5}{base[s]['macro_f1']:>8.3f}{swir[s]['macro_f1']:>8.3f}"
                 f"{swir[s]['macro_f1'] - base[s]['macro_f1']:>+8.3f}")

    L.append(
        "\nNotes: both runs use the same seeds and block splits, but "
        "training is stochastic, so differences below about 0.01 are noise. "
        "If the patch files differ slightly (SWIR adds a few invalid-pixel "
        "rejections), the test sets are nearly, not exactly, identical.")

    text = "\n".join(L)
    print(text)
    (OUTPUT_DIR / "swir_comparison.txt").write_text(text, encoding="utf-8")
    print("\nSaved:", OUTPUT_DIR / "swir_comparison.txt")


if __name__ == "__main__":
    main()