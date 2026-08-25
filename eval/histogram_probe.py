"""
Cheap, spatially-blind probe: does the intensity histogram alone (no spatial
structure) of the non-lung region predict TB status?

Complements the CNN-based inverse-mask ablation (lung_crop.py --mode
complement + train_diagnostic.py --lung-complement). If a plain histogram
logistic regression matches that CNN's AUC, the model isn't reading anatomy
(shoulders, diaphragm edges) at all — it's reading global acquisition/exposure
statistics, which would make a region-decomposition ablation moot. See
code/readme.md's Phase 2 validation results.

Example:
    python -m eval.histogram_probe --shenzhen ../datasets/tb-shenzen
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from data.dataset import load_montgomery, load_shenzhen, load_tbx11k
from utils.lung_mask import lung_bbox


def _histogram_feature(path: Path, n_bins: int) -> np.ndarray:
    """Normalized intensity histogram of every pixel OUTSIDE the lung bbox — no spatial info."""
    arr = np.array(Image.open(path).convert("L"))
    x0, y0, x1, y1 = lung_bbox(arr)
    mask = np.ones_like(arr, dtype=bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = False
    hist, _ = np.histogram(arr[mask], bins=n_bins, range=(0, 255), density=True)
    return hist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shenzhen",   type=str, default=None)
    parser.add_argument("--montgomery", type=str, default=None)
    parser.add_argument("--tbx11k",     type=str, default=None)
    parser.add_argument("--bins",       type=int, default=32)
    args = parser.parse_args()

    samples = []
    if args.shenzhen:
        samples += load_shenzhen(Path(args.shenzhen))
    if args.montgomery:
        samples += load_montgomery(Path(args.montgomery))
    if args.tbx11k:
        root = Path(args.tbx11k)
        samples += load_tbx11k(root, split="train") + load_tbx11k(root, split="val")
    if not samples:
        raise ValueError("Provide at least one dataset path.")

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(samples))
    n_val = max(1, int(len(samples) * 0.15))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    print(f"Computing lung-region-excluded intensity histograms for {len(samples)} images...")
    X = np.stack([_histogram_feature(p, args.bins) for p, _ in tqdm(samples)])
    y = np.array([label for _, label in samples])

    Xtr, ytr = X[train_idx], y[train_idx]
    Xval, yval = X[val_idx], y[val_idx]

    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    auc = roc_auc_score(yval, clf.predict_proba(Xval)[:, 1])
    print(f"\nHistogram-only (no spatial structure) val AUC: {auc:.4f}  "
          f"(train={len(train_idx)}, val={len(val_idx)})")
    if auc > 0.75:
        print("  -> Well above chance from intensity distribution alone: a meaningful share "
              "of the TB-predictive signal outside the lung field is global acquisition "
              "statistics (exposure/processing), not anatomy.")
    else:
        print("  -> Close to chance: the non-lung region's signal is not explained by intensity "
              "distribution alone — spatial structure (anatomy, markers, borders) is doing the work.")


if __name__ == "__main__":
    main()
