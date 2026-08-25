"""
AdaBN — test-time adaptation via BatchNorm recalibration, using only
unlabeled target-domain images (the held-out source's own images, no
labels, no backward pass). Cheap: one forward pass per target image.
Attacks exactly the kind of acquisition-statistics shift this project's
confound audit found — BatchNorm's running mean/var are literally first- and
second-order feature statistics, which is what differs between
scanners/sites. See code/readme.md's Phase 2 validation results.

Standard AdaBN protocol: the same unlabeled target images are used both to
recompute BN statistics and to evaluate afterward — there is no separate
"adaptation" split, since the whole point is adapting with only what a
deployment site has (its own unlabeled images). This is not label leakage
(labels are never used for adaptation), but the model does see the images
themselves before being scored on them — standard for this method, worth
naming plainly rather than letting it pass unremarked.

Example:
    python -m eval.adabn --checkpoint outputs/step3_mild_lungcrop_tbx11k/best_model.pt \
        --shenzhen ../datasets/tb-shenzen --montgomery ../datasets/tb-montgomery --tbx11k ../datasets/tbx11k \
        --held-out tbx11k --lung-crop
"""
import argparse
import copy

import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)

import numpy as np
import torch
import torch.nn as nn

from config import DataConfig, TrainConfig
from data.transforms import get_val_transforms
from eval.metrics import bootstrap_sens_at_spec_ci, compute_auc, sens_at_spec
from models.tb_model import build_model
from train_diagnostic import build_splits, make_loader


@torch.no_grad()
def adabn_adapt(model: nn.Module, loader, device: torch.device) -> nn.Module:
    """Adapts BatchNorm running stats toward unlabeled target images.
    Mutates a deep copy — the original checkpoint's model is left untouched.

    Blends from the *trained* running stats rather than a full
    reset_running_stats(): resetting all 48 BN layers in this network to
    scratch simultaneously is numerically unstable (verified — activations
    reach 10^7 within a few batches, and the accumulated running_var goes
    non-finite by the end, producing NaN predictions). Starting from a
    well-calibrated state and blending with a small EMA momentum keeps the
    network numerically sane throughout adaptation.
    """
    model = copy.deepcopy(model)
    model.eval()  # recursively puts everything (incl. BN) in eval mode first —
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for bn in bns:
        bn.momentum = 0.05
        bn.training = True  # ...then flip only the BN *leaves* back on, set directly
        # (not bn.train(), which would recurse — moot for a leaf, but the point is
        # this must come after model.eval(), never before: any later .eval()/.train()
        # call on a parent container recursively resets its children, silently
        # undoing this. That's what made the first version of this function a no-op.)

    for imgs, _ in loader:
        model(imgs.to(device))

    model.eval()
    return model


def _score(model, loader, device):
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
                        choices=["shenzhen", "montgomery", "tbx11k"])
    parser.add_argument("--lung-crop",  action="store_true")
    parser.add_argument("--variant",    type=str, default=None)
    parser.add_argument("--backbone",   type=str, default="efficientnet_b0")
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

    held_loader, _ = make_loader(held_s, get_val_transforms(args.image_size),
                                 TrainConfig(batch_size=args.batch_size))

    probs_before, y = _score(model, held_loader, device)
    auc_before = compute_auc(y, probs_before)
    point_before = sens_at_spec(y, probs_before, args.target_spec)
    print(f"\nBefore AdaBN — AUC: {auc_before:.4f}  "
          f"sens@spec{args.target_spec:.0%}: {point_before['sensitivity']:.3f}")

    adapted = adabn_adapt(model, held_loader, device)
    # held_loader shuffles each pass, so probs_after/y_after are independently
    # paired from probs_before/y — that's fine, each _score() call returns a
    # correctly-matched (probs, labels) pair on its own.
    probs_after, y_after = _score(adapted, held_loader, device)
    auc_after = compute_auc(y_after, probs_after)
    point_after = sens_at_spec(y_after, probs_after, args.target_spec)
    ci_after = bootstrap_sens_at_spec_ci(y_after, probs_after, args.target_spec)
    lo, hi = ci_after["sensitivity_ci"]

    print(f"After AdaBN  — AUC: {auc_after:.4f}  "
          f"sens@spec{args.target_spec:.0%}: {point_after['sensitivity']:.3f}  "
          f"(95% CI [{lo:.3f}, {hi:.3f}], n_boot={ci_after['n_boot_used']})")
    print(f"\nDelta — AUC: {auc_after - auc_before:+.4f}  "
          f"sens@spec{args.target_spec:.0%}: {point_after['sensitivity'] - point_before['sensitivity']:+.4f}")


if __name__ == "__main__":
    main()
