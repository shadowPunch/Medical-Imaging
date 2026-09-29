"""
In-domain complement-AUC monitor: reproduces the exact single-source 85/15
val split a `--held-out none` run used (same seed=42 rng.permutation in
train_diagnostic.build_splits — reusing it with a different `variant`
reproduces the identical split identity, since sample order only depends on
filename and the rng is seeded the same way), then scores a trained
checkpoint on that SAME validation split's *complement* variant (lungs
blanked) instead of whatever variant it was trained on.

Directly comparable to the in-domain complement-AUC baselines in
docs/investigation-log.md (Shenzhen-only, no intervention: 0.933). If this number stays
near that baseline after an intervention, the model still reads the
confound; if it drops toward 0.5, the intervention worked. This is the
"standing diagnostic" every robustness intervention should be checked
against, not just the headline held-out number.

Example:
    python -m eval.complement_monitor --checkpoint outputs/step2_shenzhen_textureaug/best_model.pt \
        --shenzhen ../datasets/tb-shenzen
"""
import argparse

import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)

import numpy as np
import torch

from config import DataConfig, TrainConfig
from data.transforms import get_val_transforms
from eval.metrics import compute_auc
from models.tb_model import build_model
from train_diagnostic import build_splits, make_loader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--shenzhen",   type=str, default=None)
    parser.add_argument("--montgomery", type=str, default=None)
    parser.add_argument("--backbone",   type=str, default="efficientnet_b0")
    parser.add_argument("--complement-variant", type=str, default="lungcomplement",
                        help="e.g. lungcomplement_d25 for the +25px dilated complement")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.tbx11k = None
    args.held_out = "none"

    _, val_s, _ = build_splits(args, DataConfig(), variant=args.complement_variant)
    print(f"In-domain val split ({args.complement_variant}): {len(val_s)} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_loader, _ = make_loader(val_s, get_val_transforms(args.image_size),
                                TrainConfig(batch_size=args.batch_size))
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            with torch.amp.autocast(device.type):
                logits = model(imgs.to(device))
            all_probs.append(torch.sigmoid(logits).squeeze(1).float().cpu().numpy())
            all_labels.append(labels.numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    auc = compute_auc(y_true, y_prob)
    print(f"\nIn-domain complement-AUC ({args.complement_variant}, same val split): {auc:.4f}")
    print("  Reference points (docs/investigation-log.md, Shenzhen-only): "
          "no intervention 0.933 (margin-only) / 0.951 (+25px dilated); "
          "mild-aug full-frame 0.893")


if __name__ == "__main__":
    main()
