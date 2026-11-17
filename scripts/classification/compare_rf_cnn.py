"""Fair Random Forest vs CNN comparison on the SAME patches, splits and classes.

- Same patch file (patches_cnn.npz), same 4 classes, same geographic
  block splits (same seeds as run_multi_seed.py).
- Two RF variants:
    RF-centre : the 6 features of the patch centre (2x2 mean). This is the
                pixel-level view, like your original RF baseline.
    RF-stats  : per-patch mean and std of each of the 6 channels (12 features).
                Gives the RF patch-level information, but no spatial
                structure.
- CNN results are read from outputs/cnn_multi_seed.csv, so run
  run_multi_seed.py first with the same seeds.

Usage (from D:\\major_project):
    python .\\scripts\\classification\\compare_rf_cnn.py
    python .\\scripts\\classification\\compare_rf_cnn.py 42 43 44 45 46
"""

import csv
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from lulc_common import (
    OUTPUT_DIR,
    PATCH_FILE,
    CLASS_NAMES,
    FEATURE_NAMES,
    block_split,
)

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
N_CLASSES = len(CLASS_NAMES)
PATCH_SIZE = 32
CNN_CSV = OUTPUT_DIR / "cnn_multi_seed.csv"


# ------------------------------------------------------------------ data
def load_data():
    with np.load(PATCH_FILE, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        rows = data["rows"]
        cols = data["cols"]
        width = int(data["W"])
        resolution = int(data["res_m"])
        names = [str(v) for v in data["bands"].tolist()]

    if names != FEATURE_NAMES:
        raise ValueError("Feature order mismatch in patch file.")
    if not np.isfinite(X).all():
        raise ValueError("Dataset contains NaN or infinite values.")

    return X, y, rows, cols, width, resolution


def centre_features(X):
    c = PATCH_SIZE // 2
    return X[:, :, c - 1:c + 1, c - 1:c + 1].mean(axis=(2, 3))


def stats_features(X):
    return np.concatenate(
        [X.mean(axis=(2, 3)), X.std(axis=(2, 3))], axis=1)


def load_cnn_results():
    if not CNN_CSV.exists():
        raise FileNotFoundError(
            f"{CNN_CSV} not found. Run run_multi_seed.py first.")

    results = {}
    with open(CNN_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seed = int(r["seed"])
            f1s = []
            for n in CLASS_NAMES:
                v = r[f"f1_{n}"]
                f1s.append(float(v) if v != "" else np.nan)
            results[seed] = {
                "acc": float(r["test_acc"]),
                "macro_f1": float(r["test_macro_f1"]),
                "f1_per": np.array(f1s),
            }
    return results


# ----------------------------------------------------------------- model
def run_rf(F, y, train_idx, test_idx, seed):
    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(F[train_idx], y[train_idx])
    preds = rf.predict(F[test_idx])
    y_test = y[test_idx]

    present = np.unique(y_test)
    acc = accuracy_score(y_test, preds)
    macro = f1_score(y_test, preds, labels=present,
                     average="macro", zero_division=0)

    _, _, f1_per, support = precision_recall_fscore_support(
        y_test, preds, labels=list(range(N_CLASSES)), zero_division=0)
    f1_per = np.where(support > 0, f1_per, np.nan)

    cm = confusion_matrix(y_test, preds, labels=list(range(N_CLASSES)))

    return {"acc": acc, "macro_f1": macro, "f1_per": f1_per, "cm": cm}


def mean_std(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "n/a"
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():.3f} +/- {sd:.3f}"


# ------------------------------------------------------------------ main
def main():
    seeds = [int(s) for s in sys.argv[1:]] or DEFAULT_SEEDS

    if not PATCH_FILE.exists():
        raise FileNotFoundError(f"Run make_patches.py first: {PATCH_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading patches and CNN results...")
    X, y, rows, cols, width, resolution = load_data()
    cnn = load_cnn_results()

    feats = {
        "RF-centre": centre_features(X),
        "RF-stats": stats_features(X),
    }
    model_names = ["CNN", "RF-stats", "RF-centre"]
    res = {m: {} for m in model_names}
    used = []

    for seed in seeds:
        if seed not in cnn:
            print(f"Seed {seed}: no CNN result in CSV, skipped.")
            continue

        train_mask, val_mask, test_mask = block_split(
            rows, cols, width, resolution,
            patch=PATCH_SIZE, block_m=10000, seed=seed)

        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)

        # Same skip rule as run_multi_seed.py
        if (np.bincount(y[train_idx], minlength=N_CLASSES) == 0).any():
            print(f"Seed {seed}: class missing in training, skipped.")
            continue
        if (np.bincount(y[np.flatnonzero(val_mask)],
                        minlength=N_CLASSES) == 0).any():
            print(f"Seed {seed}: class missing in validation, skipped.")
            continue

        print(f"\n=== Seed {seed} ===")
        res["CNN"][seed] = cnn[seed]
        for name, F in feats.items():
            res[name][seed] = run_rf(F, y, train_idx, test_idx, seed)
            r = res[name][seed]
            print(f"{name:10s} acc {r['acc']:.4f} | macro F1 {r['macro_f1']:.4f}")
        print(f"{'CNN':10s} acc {cnn[seed]['acc']:.4f} | "
              f"macro F1 {cnn[seed]['macro_f1']:.4f}")
        used.append(seed)

    if not used:
        raise RuntimeError("No usable seeds.")

    # ---------------------------------------------------------- summary
    L = []
    L.append("CNN vs RANDOM FOREST (same patches, classes, geographic splits)")
    L.append("=" * 63)
    L.append(f"Seeds used: {used}\n")

    L.append("Per-seed macro F1")
    L.append(f"{'seed':>5} " + " ".join(f"{m:>10}" for m in model_names))
    for s in used:
        L.append(f"{s:>5} " + " ".join(
            f"{res[m][s]['macro_f1']:>10.3f}" for m in model_names))

    L.append("\nMean +/- std over seeds")
    L.append(f"{'':14s}" + " ".join(f"{m:>16}" for m in model_names))
    L.append(f"{'Accuracy':14s}" + " ".join(
        f"{mean_std([res[m][s]['acc'] for s in used]):>16}"
        for m in model_names))
    L.append(f"{'Macro F1':14s}" + " ".join(
        f"{mean_std([res[m][s]['macro_f1'] for s in used]):>16}"
        for m in model_names))
    for i, cname in enumerate(CLASS_NAMES):
        L.append(f"F1 {cname:11s}" + " ".join(
            f"{mean_std([res[m][s]['f1_per'][i] for s in used]):>16}"
            for m in model_names))

    L.append("\nPaired difference, CNN minus RF (same seed)")
    for rf_name in ("RF-stats", "RF-centre"):
        d = np.array([res["CNN"][s]["macro_f1"] - res[rf_name][s]["macro_f1"]
                      for s in used])
        wins = int((d > 0).sum())
        L.append(f"vs {rf_name:10s}: macro F1 diff {d.mean():+.3f} "
                 f"(min {d.min():+.3f}, max {d.max():+.3f}); "
                 f"CNN better in {wins}/{len(used)} seeds")
        for i, cname in enumerate(CLASS_NAMES):
            dc = np.array([res["CNN"][s]["f1_per"][i] - res[rf_name][s]["f1_per"][i]
                           for s in used])
            dc = dc[~np.isnan(dc)]
            if len(dc):
                L.append(f"    {cname:12s} F1 diff {dc.mean():+.3f}")

    for rf_name in ("RF-stats", "RF-centre"):
        cm = sum(res[rf_name][s]["cm"] for s in used)
        L.append(f"\nSummed confusion matrix, {rf_name} (rows = truth):")
        L.append(np.array2string(cm))

    L.append(
        "\nNotes: CNN numbers come from run_multi_seed.py. The seeds re-split "
        "the same patches, so test sets overlap between seeds; the std "
        "understates uncertainty from new places or dates. Labels are ESA "
        "WorldCover, not ground truth. With 5 seeds, treat small differences "
        "(~0.01) as ties.")

    text = "\n".join(L)
    print("\n" + text)
    (OUTPUT_DIR / "cnn_vs_rf.txt").write_text(text, encoding="utf-8")

    with open(OUTPUT_DIR / "cnn_vs_rf.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "model", "acc", "macro_f1"]
                   + [f"f1_{n}" for n in CLASS_NAMES])
        for s in used:
            for m in model_names:
                r = res[m][s]
                w.writerow([s, m, f"{r['acc']:.4f}", f"{r['macro_f1']:.4f}"]
                           + ["" if np.isnan(v) else f"{v:.4f}"
                              for v in r["f1_per"]])

    print("\nSaved:", OUTPUT_DIR / "cnn_vs_rf.txt")
    print("Saved:", OUTPUT_DIR / "cnn_vs_rf.csv")


if __name__ == "__main__":
    main()