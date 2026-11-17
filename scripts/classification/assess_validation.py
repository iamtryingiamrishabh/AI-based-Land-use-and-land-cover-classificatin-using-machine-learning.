"""Accuracy assessment of the map against hand-labelled points.

Uses stratified estimators (Olofsson et al. 2014): classes are weighted by
the mapped area, not by the number of sample points.

Usage:
    python .\\scripts\\classification\\assess_validation.py
"""

import csv

import numpy as np
import rasterio

from lulc_common import OUTPUT_DIR, CLASS_NAMES, SUFFIX

MAP_FILE = OUTPUT_DIR / f"landcover_map{SUFFIX}.tif"
BLANK_CSV = OUTPUT_DIR / f"validation_points{SUFFIX}.csv"
KEY_CSV = OUTPUT_DIR / f"validation_points_key{SUFFIX}.csv"
REPORT = OUTPUT_DIR / f"validation_report{SUFFIX}.txt"

REF = CLASS_NAMES + ["Other"]
K = len(CLASS_NAMES)
NODATA = 255
Z = 1.96


def main():
    with open(KEY_CSV, newline="", encoding="utf-8") as f:
        key = {int(r["id"]): int(r["map_class"]) for r in csv.DictReader(f)}

    n = np.zeros((K, len(REF)))
    skipped = 0
    with open(BLANK_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lab = r["true_class"].strip()
            if lab not in REF:
                skipped += 1
                continue
            n[key[int(r["id"])], REF.index(lab)] += 1

    ni = n.sum(axis=1)
    if (ni < 2).any():
        raise RuntimeError(
            f"Need at least 2 labelled points per map class. Got {ni.astype(int)}.")

    with rasterio.open(MAP_FILE) as src:
        cls = src.read(1)
    N = np.array([(cls == c).sum() for c in range(K)], dtype=float)
    W = N / N.sum()

    p = W[:, None] * n / ni[:, None]              # estimated area proportions
    diag = np.array([n[i, i] for i in range(K)])
    U = diag / ni                                  # user's accuracy
    OA = np.trace(p[:, :K])
    se_oa = np.sqrt(np.sum(W ** 2 * U * (1 - U) / (ni - 1)))
    se_u = np.sqrt(U * (1 - U) / (ni - 1))

    PA = np.full(K, np.nan)
    se_p = np.full(K, np.nan)
    for j in range(K):
        col_tot = p[:, j].sum()
        if col_tot == 0:
            continue
        PA[j] = p[j, j] / col_tot
        Nhat = np.sum(N * n[:, j] / ni)
        term1 = N[j] ** 2 * (1 - PA[j]) ** 2 * U[j] * (1 - U[j]) / (ni[j] - 1)
        term2 = 0.0
        for i in range(K):
            if i != j:
                frac = n[i, j] / ni[i]
                term2 += N[i] ** 2 * frac * (1 - frac) / (ni[i] - 1)
        se_p[j] = np.sqrt((term1 + PA[j] ** 2 * term2) / Nhat ** 2)

    L = ["VALIDATION AGAINST HAND-LABELLED POINTS", "=" * 40,
         f"Labelled points used: {int(n.sum())}  (skipped/unsure: {skipped})",
         "", "Confusion matrix (rows = map, cols = reference, counts):",
         f"{'':12s}" + "".join(f"{r:>13s}" for r in REF)]
    for i, name in enumerate(CLASS_NAMES):
        L.append(f"{name:12s}" + "".join(f"{int(v):>13d}" for v in n[i]))

    L += ["", f"Overall accuracy: {OA:.3f} +/- {Z * se_oa:.3f} (95% CI)", "",
          f"{'class':12s}{'User acc':>16s}{'Producer acc':>18s}{'n':>6s}"]
    for i, name in enumerate(CLASS_NAMES):
        pa = ("n/a" if np.isnan(PA[i])
              else f"{PA[i]:.3f} +/- {Z * se_p[i]:.3f}")
        L.append(f"{name:12s}{U[i]:>9.3f} +/- {Z * se_u[i]:.3f}{pa:>18s}"
                 f"{int(ni[i]):>6d}")

    L += ["", "Mapped area share: " + ", ".join(
        f"{CLASS_NAMES[i]} {100 * W[i]:.1f}%" for i in range(K)),
        "",
        "Notes: user's accuracy = how often a mapped class is right on the "
        "ground; producer's accuracy = how much of the real class was found. "
        "Points labelled 'Other' count as errors for whichever class the map "
        "gave them. With ~50 points per class, the confidence intervals are "
        "wide; treat small differences as unresolved."]

    text = "\n".join(L)
    print(text)
    REPORT.write_text(text, encoding="utf-8")
    print("\nSaved:", REPORT)


if __name__ == "__main__":
    main()