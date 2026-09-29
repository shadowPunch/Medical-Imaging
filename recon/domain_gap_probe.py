"""
Measures the DRR -> real CXR domain gap for a trained reconstruction checkpoint,
without paired real ground truth (none exists publicly).

Two proxies:
  1. Feature side — encode held-out-CT DRRs and real CXRs from the encoder's
     pooled features. Two numbers: a linear classifier's 5-fold CV AUC for
     telling them apart (same method as Phase 2's shortcut classifier; it
     saturates at 1.0 for large gaps, so it can't show partial progress), and
     a continuous separation ratio — squared distance between the two domains'
     feature means over their average within-domain variance. Scale-free, so
     comparable across checkpoints whose feature spaces differ. Lower = the
     encoder treats real films more like what it trained on.
  2. Output side — per-volume statistics of the predicted volumes (near-zero air
     fraction, mean density, percentiles) for DRR inputs vs. real-CXR inputs,
     both compared against the same statistics of the real CTs themselves.
     A patient's CT is unknown for a real CXR, but the *population* of chest
     CTs is known — a reconstruction that is plausible should look like it.

Real CXRs come from TBX11K's val split, which no Phase 3 training run used
(shape induction draws from TBX11K train / Rahman).

Example:
    python -m recon.domain_gap_probe \
        --checkpoint outputs/phase3_recon_run2_fullscale/latest.pt \
        --ct-dir ../datasets/lidc-idri/dicom_heldout --tbx11k ../datasets/tbx11k \
        --image-size 320 --volume-size 128
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.dataset import load_tbx11k
from recon import tracking
from recon.checkpoint import load_recon_model
from recon.ct_data import build_drr, load_ct_volume_cached, random_pose
from recon.train_recon import drr_to_model_input, real_cxr_to_model_input

STAT_NAMES = ["frac_below_0.02", "mean", "p50", "p90", "p99"]


def volume_stats(vol: torch.Tensor) -> list[float]:
    v = vol.flatten().float()
    q = torch.quantile(v[torch.randperm(v.numel(), device=v.device)[:200_000]],
                       torch.tensor([0.5, 0.9, 0.99], device=v.device))
    return [(v < 0.02).float().mean().item(), v.mean().item(), *q.tolist()]


@torch.no_grad()
def encode(model, x: torch.Tensor) -> tuple[np.ndarray, list[float]]:
    feats = model.encoder(x)
    pooled = feats[-1].mean(dim=(2, 3)).squeeze(0).float().cpu().numpy()
    return pooled, volume_stats(model.recon_head(feats))


def _robust_var(a: np.ndarray) -> np.ndarray:
    mad = np.median(np.abs(a - np.median(a, 0)), 0)
    return (1.4826 * mad) ** 2  # MAD scaled to match the std of a normal distribution


def separation_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Median/MAD version, so a few exploding inputs can't inflate the
    within-domain spread and fake a small gap."""
    between = float(((np.median(a, 0) - np.median(b, 0)) ** 2).sum())
    within = 0.5 * float(_robust_var(a).sum() + _robust_var(b).sum())
    return between / within


def summarize(rows: list[list[float]]) -> dict:
    a = np.array(rows)
    return {n: {"mean": float(a[:, i].mean()), "std": float(a[:, i].std()),
                "median": float(np.median(a[:, i]))}
            for i, n in enumerate(STAT_NAMES)}


def count_outliers(rows: list[list[float]], ct: dict, k: float = 5.0) -> int:
    """Predicted volumes whose mean density is > k real-CT stds from the real-CT mean."""
    means = np.array(rows)[:, STAT_NAMES.index("mean")]
    return int((np.abs(means - ct["mean"]["mean"]) > k * ct["mean"]["std"]).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ct-dir", type=str, required=True)
    parser.add_argument("--tbx11k", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--volume-size", type=int, default=128)
    parser.add_argument("--poses-per-ct", type=int, default=25)
    parser.add_argument("--n-real", type=int, default=300)
    parser.add_argument("--drr-realism", type=str, default="off",
                        help="Render the probe's DRRs with this realism strength "
                             "(recon/drr_realism.py) instead of clean.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    run = tracking.start_run("domain_gap_probe", vars(args),
                             name=tracking.run_name("gap", args.checkpoint),
                             tags=["phase3", "probe"])
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_recon_model(args.checkpoint, args.volume_size, device)

    drr_feats, drr_stats, ct_stats = [], [], []
    for series in sorted(p for p in Path(args.ct_dir).iterdir() if p.is_dir()):
        subject = load_ct_volume_cached(series)
        target = F.interpolate(subject.density.data.to(device)[None].float(),
                               size=(args.volume_size,) * 3, mode="trilinear", align_corners=False)
        ct_stats.append(volume_stats(target))
        drr = build_drr(subject, height=args.image_size, device=device)
        for _ in range(args.poses_per_ct):
            rot, trans = random_pose(device=device)
            with torch.no_grad():
                img = drr(rot, trans, parameterization="euler_angles", convention="ZXY")
            f, s = encode(model, drr_to_model_input(img, args.image_size, args.drr_realism, gen))
            drr_feats.append(f)
            drr_stats.append(s)
        del drr
        torch.cuda.empty_cache()

    real_paths = [p for p, _ in load_tbx11k(Path(args.tbx11k), split="val")]
    real_paths = random.sample(real_paths, min(args.n_real, len(real_paths)))
    real_feats, real_stats = [], []
    for p in real_paths:
        f, s = encode(model, real_cxr_to_model_input(p, args.image_size).to(device))
        real_feats.append(f)
        real_stats.append(s)

    X = np.vstack([drr_feats, real_feats])
    y = np.array([0] * len(drr_feats) + [1] * len(real_feats))
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    aucs = cross_val_score(probe, X, y, cv=cv, scoring="roc_auc")
    sep = separation_ratio(np.vstack(drr_feats), np.vstack(real_feats))

    ct, drr_s, real_s = summarize(ct_stats), summarize(drr_stats), summarize(real_stats)

    def z(pred):  # standardized distance of the typical prediction from the real-CT population
        return {n: abs(pred[n]["median"] - ct[n]["mean"]) / (ct[n]["std"] + 1e-8) for n in STAT_NAMES}

    result = {
        "checkpoint": args.checkpoint, "probe_drr_realism": args.drr_realism,
        "separation_ratio": sep,
        "n_drr": len(drr_feats), "n_real": len(real_feats), "n_ct": len(ct_stats),
        "domain_probe_auc": {"mean": float(aucs.mean()), "std": float(aucs.std())},
        "ct_stats": ct, "drr_pred_stats": drr_s, "real_pred_stats": real_s,
        "drr_pred_z_from_ct": z(drr_s), "real_pred_z_from_ct": z(real_s),
        "n_outliers": {"drr": count_outliers(drr_stats, ct), "real": count_outliers(real_stats, ct)},
    }

    print(f"Samples: {len(drr_feats)} DRRs ({len(ct_stats)} held-out CTs), {len(real_feats)} real CXRs")
    print(f"\nDomain probe AUC (DRR vs real, 5-fold CV): {aucs.mean():.3f} ± {aucs.std():.3f}"
          f"   (1.0 = fully separable, 0.5 = indistinguishable)")
    print(f"Feature separation ratio: {sep:.3f}   (lower = real CXRs closer to training DRRs)")
    print(f"Exploding predictions (mean > 5 CT-std off): {result['n_outliers']['drr']}/{len(drr_feats)} DRR, "
          f"{result['n_outliers']['real']}/{len(real_feats)} real")
    print(f"\n{'stat':<16}{'real CT':>18}{'pred | DRR':>18}{'pred | real CXR':>18}{'z DRR':>8}{'z real':>8}")
    for n in STAT_NAMES:
        fmt = lambda d: f"{d[n]['median']:.3f}"
        print(f"{n:<16}{fmt(ct):>18}{fmt(drr_s):>18}{fmt(real_s):>18}"
              f"{result['drr_pred_z_from_ct'][n]:>8.2f}{result['real_pred_z_from_ct'][n]:>8.2f}")
    print("\nPredictions: median over samples. z = |median - real-CT mean| / real-CT std; lower is more CT-like.")

    tracking.finish(run, {
        "separation_ratio": sep, "domain_probe_auc": float(aucs.mean()),
        "n_outliers_drr": result["n_outliers"]["drr"], "n_outliers_real": result["n_outliers"]["real"],
        **{f"z_real/{k}": v for k, v in result["real_pred_z_from_ct"].items()},
        **{f"z_drr/{k}": v for k, v in result["drr_pred_z_from_ct"].items()}})

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=2))
        print(f"Saved -> {args.json_out}")


if __name__ == "__main__":
    main()
