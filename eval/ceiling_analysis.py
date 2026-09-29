"""
Ceiling analysis: sensitivity at spec>=0.70 read off the ROC curve of an
already-trained checkpoint's held-out predictions — no retraining needed.

This separates two questions the frozen-threshold held-out result conflates:
does the encoder discriminate at all on this held-out source (an unconstrained
per-site threshold ceiling), versus does one specific frozen global threshold
happen to transfer. A low ceiling means discrimination itself is the problem
(domain shift in the representation); a high ceiling despite a bad
frozen-threshold result means the failure is purely calibration/threshold
transfer, fixable with per-deployment recalibration rather than a better
encoder. See docs/investigation-log.md's Phase 2 validation results.

Example:
    python -m eval.ceiling_analysis --checkpoint outputs/diagnostic_montgomery/best_model.pt \
        --shenzhen ../datasets/tb-shenzen --montgomery ../datasets/tb-montgomery --tbx11k ../datasets/tbx11k \
        --held-out montgomery
"""
import argparse
from pathlib import Path

import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)

import numpy as np
import torch

from config import DataConfig, TrainConfig
from data.transforms import get_val_transforms
from eval.metrics import bootstrap_sens_at_spec_ci, compute_auc, sens_at_spec
from models.tb_model import build_model
from train_diagnostic import build_splits, make_loader


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
    parser.add_argument("--variant",    type=str, default=None,
                        help="Raw variant string override (e.g. lungcrop_clahe) — bypasses --lung-crop")
    parser.add_argument("--backbone",   type=str, default="efficientnet_b0",
                        help="Must match the checkpoint's backbone (e.g. xrv_densenet121-res224-all)")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target-spec", type=float, default=0.70)
    args = parser.parse_args()

    variant = args.variant if args.variant is not None else ("lungcrop" if args.lung_crop else "")
    _, _, held_s = build_splits(args, DataConfig(), variant=variant)
    print(f"Held-out [{args.held_out}] samples: {len(held_s)}  (variant={variant or 'none'})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}  (val AUC at train time={ckpt['metrics']['auc']:.4f})")

    held_loader, _ = make_loader(held_s, get_val_transforms(args.image_size),
                                 TrainConfig(batch_size=args.batch_size))

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in held_loader:
            with torch.amp.autocast(device.type):
                logits = model(imgs.to(device))
            all_probs.append(torch.sigmoid(logits).squeeze(1).float().cpu().numpy())
            all_labels.append(labels.numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    auc = compute_auc(y_true, y_prob)
    point = sens_at_spec(y_true, y_prob, args.target_spec)

    print(f"\nHeld-out AUC (this checkpoint, this held-out set): {auc:.4f}")
    if point["threshold"] is None:
        print(f"No operating point on this held-out set reaches spec>={args.target_spec:.0%}.")
        return

    ci = bootstrap_sens_at_spec_ci(y_true, y_prob, args.target_spec)
    lo, hi = ci["sensitivity_ci"]
    print(f"Ceiling — best sensitivity at spec>={args.target_spec:.0%}: "
          f"{point['sensitivity']:.3f}  (95% CI [{lo:.3f}, {hi:.3f}], n_boot={ci['n_boot_used']})")
    print(f"  achieved at spec={point['specificity']:.3f}, threshold={point['threshold']:.3f}")
    if point["sensitivity"] < 0.90:
        print("  -> Below the 90% WHO-TPP sensitivity target even under a perfect per-site "
              "threshold: discrimination itself is binding here, not just calibration.")
    else:
        print("  -> At/above the 90% WHO-TPP sensitivity target under a per-site threshold: "
              "the frozen-threshold failure is a calibration/threshold-transfer problem, "
              "not a discrimination problem, on this held-out set.")


if __name__ == "__main__":
    main()
