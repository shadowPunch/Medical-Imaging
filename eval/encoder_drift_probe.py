"""
Does Phase 3 reconstruction training break Head A?

The proposal's architecture shares one encoder between the diagnostic head
(Head A) and the reconstruction head (Head B). Phase 3 trains the encoder
together with Head B, on DRRs rendered from CT — a different input
distribution from the lung-cropped real CXRs Head A was tuned on. §11a's
firewall guarantees Head B never *feeds* Head A at inference; it says nothing
about the encoder drifting underneath Head A during Phase 3 training. That is
a training-time coupling, and it is measurable.

This grafts a Phase 3 checkpoint's encoder onto a Phase 2 diagnostic
checkpoint's head and scores the held-out set the Phase 2 result was reported
on. Held-out AUC vs. the unmodified Phase 2 checkpoint is the drift.

Example:
    python -m eval.encoder_drift_probe \
        --checkpoint outputs/step3_mild_lungcrop_tbx11k/best_model.pt \
        --recon-checkpoint outputs/phase3_recon_run4_nosi/latest.pt \
        --shenzhen ../datasets/tb-shenzen --montgomery ../datasets/tb-montgomery \
        --tbx11k ../datasets/tbx11k --held-out tbx11k --lung-crop
"""
import argparse
import json
from pathlib import Path

import torch.multiprocessing as mp

mp.set_start_method("fork", force=True)

import numpy as np
import torch

from config import DataConfig, TrainConfig
from data.transforms import get_val_transforms
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval.metrics import bootstrap_sens_at_spec_ci, compute_auc, sens_at_spec
from models.tb_model import build_model
from train_diagnostic import build_splits, make_loader

ENCODER_PREFIX = "encoder."


def graft_encoder(base: dict, donor: dict) -> dict:
    """Base state dict with its encoder weights replaced by the donor's."""
    donor_encoder = {k: v for k, v in donor.items() if k.startswith(ENCODER_PREFIX)}
    if not donor_encoder:
        raise ValueError("donor checkpoint has no encoder.* keys to graft")
    out = dict(base)
    for k, v in donor_encoder.items():
        if k in out and out[k].shape != v.shape:
            raise ValueError(f"shape mismatch on {k}: {tuple(out[k].shape)} vs {tuple(v.shape)}")
        out[k] = v
    return out


@torch.no_grad()
def score(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    probs, labels = [], []
    for imgs, y in loader:
        with torch.amp.autocast(device.type):
            logits = model(imgs.to(device))
        probs.append(torch.sigmoid(logits).squeeze(1).float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


@torch.no_grad()
def pooled_features(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    for imgs, y in loader:
        with torch.amp.autocast(device.type):
            f = model.encoder(imgs.to(device))[-1].mean(dim=(2, 3))
        feats.append(f.float().cpu().numpy())
        labels.append(y.numpy())
    return np.vstack(feats), np.concatenate(labels)


def fit_linear_head(train_X, train_y, test_X) -> np.ndarray:
    """A fresh linear head on frozen features — separates 'the encoder lost the
    TB information' from 'the old head no longer matches the new features'."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    clf.fit(train_X, train_y)
    return clf.predict_proba(test_X)[:, 1]


def report(name: str, y_true: np.ndarray, y_prob: np.ndarray, target_spec: float) -> dict:
    auc = compute_auc(y_true, y_prob)
    point = sens_at_spec(y_true, y_prob, target_spec)
    out = {"auc": auc, "sens_at_spec": point["sensitivity"], "spec": point["specificity"]}
    print(f"{name:<28} AUC {auc:.4f}", end="")
    if point["threshold"] is None:
        print(f"   no operating point reaches spec>={target_spec:.0%}")
        out["sens_at_spec"] = None
        return out
    ci = bootstrap_sens_at_spec_ci(y_true, y_prob, target_spec)
    lo, hi = ci["sensitivity_ci"]
    out["sens_ci"] = [lo, hi]
    print(f"   ceiling sens@{target_spec:.0%} {point['sensitivity']:.3f} [{lo:.3f}, {hi:.3f}]")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Phase 2 diagnostic checkpoint (provides Head A)")
    parser.add_argument("--recon-checkpoint", type=str, required=True, nargs="+",
                        help="One or more Phase 3 checkpoints whose encoders get grafted on")
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
    parser.add_argument("--linear-probe", action="store_true",
                        help="Also refit a fresh linear head on each encoder's frozen "
                             "features, to test whether the TB signal survived the drift.")
    parser.add_argument("--json-out",   type=str, default=None)
    args = parser.parse_args()

    variant = args.variant if args.variant is not None else ("lungcrop" if args.lung_crop else "")
    train_s, _, held_s = build_splits(args, DataConfig(), variant=variant)
    print(f"Held-out [{args.held_out}]: {len(held_s)} samples (variant={variant or 'none'})\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone, pretrained=False).to(device)
    base_sd = torch.load(args.checkpoint, map_location=device, weights_only=False)["model"]
    tcfg = TrainConfig(batch_size=args.batch_size)
    loader, _ = make_loader(held_s, get_val_transforms(args.image_size), tcfg)
    train_loader = (make_loader(train_s, get_val_transforms(args.image_size), tcfg)[0]
                    if args.linear_probe else None)

    model.load_state_dict(base_sd)
    model.eval()
    results = {"baseline": report("Phase 2 encoder (baseline)", *score(model, loader, device),
                                  args.target_spec)}

    for ck in args.recon_checkpoint:
        donor = torch.load(ck, map_location=device, weights_only=False)["model"]
        model.load_state_dict(graft_encoder(base_sd, donor))
        model.eval()
        results[ck] = report(Path(ck).parent.name, *score(model, loader, device), args.target_spec)
        drop = results["baseline"]["auc"] - results[ck]["auc"]
        results[ck]["auc_drop_vs_baseline"] = drop
        print(f"{'':<28} AUC drop vs baseline: {drop:+.4f}")
        if args.linear_probe:
            train_X, train_y = pooled_features(model, train_loader, device)
            test_X, test_y = pooled_features(model, loader, device)
            probe_auc = compute_auc(test_y, fit_linear_head(train_X, train_y, test_X))
            results[ck]["linear_probe_auc"] = probe_auc
            print(f"{'':<28} refit linear head on frozen features: AUC {probe_auc:.4f}")
        print()

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"Saved -> {args.json_out}")


if __name__ == "__main__":
    main()
