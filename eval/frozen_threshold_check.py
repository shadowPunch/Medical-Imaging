"""
Frozen-threshold check on an already-trained checkpoint — no retraining needed.

Companion to ceiling_analysis.py: that script answers "can this encoder
discriminate at all here, under a perfect per-site threshold" (the ceiling).
This one answers the deployable question instead — derives the WHO-TPP
threshold from validation data only, freezes it, and scores the held-out
source at that frozen threshold with a bootstrap CI. Never re-derives a
threshold from the held-out set itself; see eval/metrics.py's apply_threshold
docstring for why that would leak.

Example:
    python -m eval.frozen_threshold_check \
        --checkpoint outputs/step3_mild_lungcrop_tbx11k/best_model.pt \
        --shenzhen ../datasets/tb-shenzen --montgomery ../datasets/tb-montgomery \
        --tbx11k ../datasets/tbx11k --held-out tbx11k --lung-crop
"""
import argparse

import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)

import numpy as np
import torch

from config import DataConfig, TrainConfig
from data.transforms import get_val_transforms
from eval.metrics import apply_threshold, bootstrap_ci, evaluate, print_frozen_report
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
    parser.add_argument("--variant",    type=str, default=None,
                        help="Raw variant string override (e.g. lungcrop_clahe) — bypasses --lung-crop")
    parser.add_argument("--backbone",   type=str, default="efficientnet_b0",
                        help="Must match the checkpoint's backbone")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    variant = args.variant if args.variant is not None else ("lungcrop" if args.lung_crop else "")
    _, val_s, held_s = build_splits(args, DataConfig(), variant=variant)
    print(f"Validation samples: {len(val_s)}  Held-out [{args.held_out}] samples: {len(held_s)}  "
          f"(variant={variant or 'none'})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}  (val AUC at train time={ckpt['metrics']['auc']:.4f})")

    val_tf = get_val_transforms(args.image_size)
    val_loader, _ = make_loader(val_s, val_tf, TrainConfig(batch_size=args.batch_size))
    val_probs, val_y = run_inference(model, val_loader, device)
    val_metrics = evaluate(val_y, val_probs)
    print(f"\nValidation AUC (source of frozen threshold): {val_metrics['auc']:.4f}")

    if val_metrics["who_tpp"] is None:
        print("WHO TPP threshold undefined on validation (90% sensitivity never "
              "reached) — cannot score held-out at a frozen operating point.")
        return

    threshold = val_metrics["who_tpp"]["threshold"]
    held_loader, _ = make_loader(held_s, val_tf, TrainConfig(batch_size=args.batch_size))
    held_probs, held_y = run_inference(model, held_loader, device)
    held_auc = evaluate(held_y, held_probs)["auc"]
    point = apply_threshold(held_y, held_probs, threshold)
    ci = bootstrap_ci(held_y, held_probs, threshold)

    print(f"Held-out AUC: {held_auc:.4f}  (gap vs. val: {val_metrics['auc'] - held_auc:+.4f})")
    print_frozen_report(point, ci)


if __name__ == "__main__":
    main()
