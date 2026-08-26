"""
Diagnose *why* the frozen-threshold check (frozen_threshold_check.py) fails
on TBX11K despite a high ceiling, and turn the "needs per-site calibration"
caveat into a concrete deployment spec — no retraining, reuses the existing
winning-config checkpoint's inference on validation + held-out.

Two analyses:

1. Score-distribution diagnosis. Plots val vs. held-out score histograms,
   split by label. Quantifies whether the val->held-out shift is a pure
   offset (same shape, different location — fixable with a handful of
   calibration cases) or a shape change (fixable only with more data / a
   different approach), via the ratio of interquartile ranges (spread) and
   the per-class median offsets.

2. Calibration-set-size sweep. For calibration sizes n in {25, 50, 100, 200,
   500}: draw n cases at random from the held-out pool (preserving true
   prevalence), set a threshold at the 70th percentile of that draw's
   *negative* scores (this is what "calibrate to 70% specificity" means and
   needs no positives), apply that frozen threshold to the remaining
   held-out cases, and record the achieved specificity there. Repeat 500x
   per size and report the mean and spread. Answers "how many labeled cases
   does a deployment site need to reliably hit its target specificity."

Example:
    python -m eval.score_distribution_diagnosis \
        --checkpoint outputs/step3_mild_lungcrop_tbx11k/best_model.pt \
        --shenzhen ../datasets/tb-shenzen --montgomery ../datasets/tb-montgomery \
        --tbx11k ../datasets/tbx11k --held-out tbx11k --lung-crop \
        --output-dir outputs/tbx11k_winning_diagnosis
"""
import argparse
from pathlib import Path

import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import DataConfig, TrainConfig
from data.transforms import get_val_transforms
from eval.metrics import apply_threshold
from models.tb_model import build_model
from train_diagnostic import build_splits, make_loader


def run_inference(model, loader, device):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            with torch.amp.autocast(device.type):
                logits = model(imgs.to(device))
            all_probs.append(torch.sigmoid(logits).squeeze(1).float().cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def plot_histograms(val_probs, val_y, held_probs, held_y, held_out_name, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    bins = np.linspace(0, 1, 41)
    for ax, probs, y, title in [
        (axes[0], val_probs, val_y, "Validation (Shenzhen + Montgomery)"),
        (axes[1], held_probs, held_y, f"Held-out ({held_out_name})"),
    ]:
        ax.hist(probs[y == 0], bins=bins, density=True, alpha=0.55, label="negative", color="steelblue")
        ax.hist(probs[y == 1], bins=bins, density=True, alpha=0.55, label="positive", color="firebrick")
        ax.set_title(title)
        ax.set_ylabel("density")
        ax.legend()
    axes[1].set_xlabel("predicted probability")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def describe_shift(val_probs, val_y, held_probs, held_y):
    print("\n─── Score-distribution diagnosis ───")
    for label, name in [(0, "negative"), (1, "positive")]:
        v = val_probs[val_y == label]
        h = held_probs[held_y == label]
        v_med, h_med = np.median(v), np.median(h)
        v_iqr = np.percentile(v, 75) - np.percentile(v, 25)
        h_iqr = np.percentile(h, 75) - np.percentile(h, 25)
        offset = h_med - v_med
        spread_ratio = h_iqr / v_iqr if v_iqr > 0 else float("nan")
        print(f"  {name:8s}: val median={v_med:.3f} IQR={v_iqr:.3f}  |  "
              f"held-out median={h_med:.3f} IQR={h_iqr:.3f}  |  "
              f"offset={offset:+.3f}  spread_ratio={spread_ratio:.2f}")


def calibration_size_sweep(held_probs, held_y, sizes, n_repeats, target_spec, seed=42):
    """
    Sizes are counts of *negatives*, not total cases — positives contribute
    nothing to placing a spec-target threshold (it's a percentile of the
    negative score distribution), so a calibration set's binding resource is
    how many negatives it contains, not its total size.
    """
    rng = np.random.default_rng(seed)
    neg_idx_all = np.where(held_y == 0)[0]
    n_neg = len(neg_idx_all)
    results = {}
    for size in sizes:
        if size >= n_neg:
            continue
        achieved = []
        for _ in range(n_repeats):
            cal_neg_idx = rng.choice(neg_idx_all, size=size, replace=False)
            cal_mask = np.zeros(len(held_y), dtype=bool)
            cal_mask[cal_neg_idx] = True
            cal_neg_probs = held_probs[cal_mask]
            rem_probs, rem_y = held_probs[~cal_mask], held_y[~cal_mask]

            threshold = np.percentile(cal_neg_probs, target_spec * 100)
            point = apply_threshold(rem_y, rem_probs, threshold)
            achieved.append(point["specificity"])
        results[size] = np.array(achieved)
    return results


def plot_calibration_sweep(results, target_spec, out_path):
    sizes = sorted(results.keys())
    means = [results[s].mean() for s in sizes]
    p10 = [np.percentile(results[s], 10) for s in sizes]
    p90 = [np.percentile(results[s], 90) for s in sizes]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, means, "o-", color="steelblue", label="mean achieved specificity")
    ax.fill_between(sizes, p10, p90, alpha=0.25, color="steelblue", label="10th-90th percentile")
    ax.axhline(target_spec, color="firebrick", linestyle="--", label=f"target ({target_spec:.0%})")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: str(int(v))))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("calibration set size (labeled negatives)")
    ax.set_ylabel("achieved specificity on remainder")
    ax.set_title("Threshold-transfer reliability vs. calibration set size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--shenzhen",   type=str, default=None)
    parser.add_argument("--montgomery", type=str, default=None)
    parser.add_argument("--tbx11k",     type=str, default=None)
    parser.add_argument("--held-out",   type=str, required=True,
                        choices=["shenzhen", "montgomery", "tbx11k", "tbx11k-val"])
    parser.add_argument("--lung-crop",  action="store_true")
    parser.add_argument("--exclude-tbx11k-tag", type=str, nargs="+", default=None)
    parser.add_argument("--tbx11k-neg-cap", type=int, default=None)
    parser.add_argument("--variant",    type=str, default=None)
    parser.add_argument("--backbone",   type=str, default="efficientnet_b0")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target-spec", type=float, default=0.70)
    parser.add_argument("--cal-sizes", type=int, nargs="+", default=[25, 50, 100, 200, 500])
    parser.add_argument("--n-repeats", type=int, default=500)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variant = args.variant if args.variant is not None else ("lungcrop" if args.lung_crop else "")
    _, val_s, held_s = build_splits(args, DataConfig(), variant=variant)
    print(f"Validation samples: {len(val_s)}  Held-out [{args.held_out}] samples: {len(held_s)}  "
          f"(variant={variant or 'none'})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_tf = get_val_transforms(args.image_size)
    val_loader, _ = make_loader(val_s, val_tf, TrainConfig(batch_size=args.batch_size))
    held_loader, _ = make_loader(held_s, val_tf, TrainConfig(batch_size=args.batch_size))
    val_probs, val_y = run_inference(model, val_loader, device)
    held_probs, held_y = run_inference(model, held_loader, device)

    hist_path = out_dir / "score_histograms.png"
    plot_histograms(val_probs, val_y, held_probs, held_y, args.held_out, hist_path)
    print(f"Saved score histograms -> {hist_path}")
    describe_shift(val_probs, val_y, held_probs, held_y)

    held_neg = held_probs[held_y == 0]
    full_threshold = np.percentile(held_neg, args.target_spec * 100)
    print(f"\nPooled held-out negative distribution's {args.target_spec:.0%}-percentile "
          f"(the value a calibration set is trying to estimate): {full_threshold:.3f}")

    print(f"\n─── Calibration-set-size sweep (target spec={args.target_spec:.0%}, "
          f"{args.n_repeats} repeats/size, sizes are negative counts) ───")
    results = calibration_size_sweep(held_probs, held_y, args.cal_sizes, args.n_repeats, args.target_spec)
    for size in sorted(results.keys()):
        vals = results[size]
        mean, std = vals.mean(), vals.std()
        p10, p90 = np.percentile(vals, [10, 90])
        within_5pt = np.mean(np.abs(vals - args.target_spec) <= 0.05)
        print(f"  n={size:4d}: achieved spec mean={mean:.3f} std={std:.3f}  "
              f"[10-90th pct: {p10:.3f}, {p90:.3f}]  "
              f"P(within ±5pt of target)={within_5pt:.1%}  (n_valid_repeats={len(vals)})")

    sweep_path = out_dir / "calibration_size_sweep.png"
    plot_calibration_sweep(results, args.target_spec, sweep_path)
    print(f"Saved calibration-size sweep -> {sweep_path}")


if __name__ == "__main__":
    main()
