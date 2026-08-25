"""
Entry point for Phase 1 + 2: train the diagnostic head and evaluate
cross-source generalisation on the held-out source.

Example:
    python train_diagnostic.py \
        --shenzhen  /data/shenzhen \
        --montgomery /data/montgomery \
        --tbx11k    /data/tbx11k \
        --held-out  montgomery \
        --output    outputs/diagnostic
"""
import argparse
import random
from pathlib import Path

import torch.multiprocessing as mp
# Python 3.14 switched Linux's default multiprocessing start method to
# 'forkserver', which breaks DataLoader workers unless set back explicitly.
mp.set_start_method("fork", force=True)

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import Config, DataConfig, ModelConfig, TrainConfig
from data.dataset import TBCXRDataset, load_shenzhen, load_montgomery, load_tbx11k
from data.transforms import get_texture_aug_transforms, get_train_transforms, get_val_transforms
from eval.metrics import apply_threshold, bootstrap_ci, evaluate, print_frozen_report
from models.tb_model import build_model
from training.loss import LabelSmoothBCE, compute_pos_weight
from training.trainer import Trainer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_splits(args, cfg: DataConfig, variant: str = ""):
    """
    Load all requested sources.
    The held-out source is fully withheld from training/validation and
    used only for cross-source generalisation evaluation.
    For every training source: 85% train / 15% val split.
    """
    rng = np.random.default_rng(42)
    train_samples, val_samples, held_out = [], [], []

    for name, path_str, loader in [
        ("shenzhen",   args.shenzhen,   load_shenzhen),
        ("montgomery", args.montgomery, load_montgomery),
    ]:
        if not path_str:
            continue
        samples = loader(Path(path_str), variant=variant)
        if name == args.held_out:
            held_out.extend(samples)
        else:
            idx   = rng.permutation(len(samples))
            n_val = max(1, int(len(samples) * 0.15))
            val_samples.extend(  [samples[i] for i in idx[:n_val]])
            train_samples.extend([samples[i] for i in idx[n_val:]])

    if args.tbx11k:
        root = Path(args.tbx11k)
        if args.held_out == "tbx11k":
            # TBX11K's own train/val labels are still valid here — both just
            # become part of the withheld cross-source evaluation set.
            for split in ("train", "val"):
                held_out.extend(load_tbx11k(
                    root, split=split, variant=variant,
                    latent_as_positive=cfg.tbx11k_latent_as_positive,
                ))
        else:
            for split, target in [("train", train_samples), ("val", val_samples)]:
                s = load_tbx11k(root, split=split, variant=variant,
                                latent_as_positive=cfg.tbx11k_latent_as_positive)
                target.extend(s)

    return train_samples, val_samples, held_out


def make_loader(
    samples: list,
    transform,
    cfg: TrainConfig,
    weighted_sampling: bool = False,
) -> tuple[DataLoader, TBCXRDataset]:
    dataset = TBCXRDataset(samples, transform)
    sampler = None
    if weighted_sampling:
        labels  = np.array(dataset.labels)
        counts  = np.bincount(labels)
        weights = (1.0 / counts)[labels]       # per-sample weight
        sampler = WeightedRandomSampler(weights, len(weights))
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True,
    )
    return loader, dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="TB diagnostic head — Phase 1 & 2")
    parser.add_argument("--shenzhen",   type=str, default=None, help="Path to Shenzhen dataset root")
    parser.add_argument("--montgomery", type=str, default=None, help="Path to Montgomery dataset root")
    parser.add_argument("--tbx11k",    type=str, default=None, help="Path to TBX11K dataset root")
    parser.add_argument("--held-out",  type=str, default="montgomery",
                        choices=["shenzhen", "montgomery", "tbx11k", "none"],
                        help="Source reserved for cross-source generalisation eval")
    parser.add_argument("--output",    type=str, default="outputs/diagnostic")
    parser.add_argument("--backbone",  type=str, default="efficientnet_b0")
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--batch-size",type=int, default=16)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--lung-crop", action="store_true",
                        help="Train on lung-cropped images (see lung_crop.py) instead of "
                             "full frames — ablation for scanner/marker shortcut reliance")
    parser.add_argument("--lung-complement", action="store_true",
                        help="Train on lung-complement images (see lung_crop.py --mode complement) "
                             "— the lung field blanked out, only surrounding anatomy/border visible. "
                             "Inverse-mask ablation: can this alone predict the TB label in-domain?")
    parser.add_argument("--dilate-px", type=int, default=0,
                        help="Match a --lung-crop/--lung-complement variant generated with "
                             "lung_crop.py --dilate-px N")
    parser.add_argument("--region",    type=str, default=None,
                        choices=["lungs", "shoulders", "spine", "diaphragm", "other"],
                        help="Train on a single anatomical region only (see region_decompose.py) "
                             "— region-decomposition ablation")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Override config.py's default (320) — e.g. 32 for a thumbnail-only "
                             "acquisition-statistics probe")
    parser.add_argument("--variant",   type=str, default=None,
                        help="Raw variant string override for one-off ablations (e.g. "
                             "patch_top_left_96, highpass_r20) — bypasses --lung-crop/"
                             "--lung-complement/--region, which only cover the named ablations")
    parser.add_argument("--shuffle-labels", action="store_true",
                        help="Permute training-set labels only (val/held-out stay real) — "
                             "sanity control: a real result should collapse to ~0.5 val AUC. "
                             "If it doesn't, something in the split/pipeline is leaking.")
    parser.add_argument("--texture-aug", action="store_true",
                        help="Randomize the acquisition-texture signature during training "
                             "(resolution round-trip, sharpen/blur jitter, noise, JPEG "
                             "compression, gamma) — targets the confound found in the "
                             "inverse-mask/high-pass/patch tests, see code/readme.md")
    parser.add_argument("--aug-strength", type=str, default="aggressive",
                        choices=["mild", "medium", "aggressive"],
                        help="Only with --texture-aug — dial between destroying the confound "
                             "and destroying real diagnostic texture (cavitation, nodules); "
                             "see get_texture_aug_transforms in data/transforms.py")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()
    if sum([args.lung_crop, args.lung_complement, args.region is not None, args.variant is not None]) > 1:
        parser.error("--lung-crop, --lung-complement, --region, and --variant are mutually exclusive")

    data_cfg = DataConfig()
    if args.image_size is not None:
        data_cfg.image_size = args.image_size

    cfg = Config(
        data=data_cfg,
        model=ModelConfig(backbone=args.backbone, pretrained=not args.no_pretrained),
        train=TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr),
        output_dir=Path(args.output),
        seed=args.seed,
    )
    seed_everything(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────
    variant = (args.variant if args.variant is not None else
               "lungcrop" if args.lung_crop else
               "lungcomplement" if args.lung_complement else
               f"region_{args.region}" if args.region else "")
    if variant and args.dilate_px and args.variant is None:
        variant += f"_d{args.dilate_px}"
    train_s, val_s, held_s = build_splits(args, cfg.data, variant=variant)
    print(f"Samples — train: {len(train_s)}  val: {len(val_s)}  held-out: {len(held_s)}"
          + (f"  (variant={variant})" if variant else ""))

    if args.shuffle_labels:
        rng = np.random.default_rng(cfg.seed)
        paths = [p for p, _ in train_s]
        shuffled = rng.permutation([l for _, l in train_s])
        train_s = list(zip(paths, (int(l) for l in shuffled)))
        print("⚠ LABEL-SHUFFLE CONTROL — training labels permuted; val/held-out labels are real. "
              "A non-leaking pipeline should land near 0.5 val AUC.")

    train_tf = (get_texture_aug_transforms(cfg.data.image_size, args.aug_strength) if args.texture_aug
               else get_train_transforms(cfg.data.image_size))
    val_tf   = get_val_transforms(cfg.data.image_size)

    # Weighted sampling corrects class imbalance at the batch level
    train_loader, train_ds = make_loader(train_s, train_tf, cfg.train, weighted_sampling=True)
    val_loader,   _        = make_loader(val_s,   val_tf,   cfg.train)

    # ── Model ───────────────────────────────────────────────────────────
    model = build_model(cfg.model.backbone, cfg.model.pretrained, cfg.model.dropout).to(device)

    pos_weight = compute_pos_weight(train_ds).to(device)
    criterion  = LabelSmoothBCE(pos_weight=pos_weight, smoothing=cfg.train.label_smoothing)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                    weight_decay=cfg.train.weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    # ── Train ───────────────────────────────────────────────────────────
    trainer = Trainer(model, optimizer, criterion, scheduler, device, cfg.output_dir)
    trainer.fit(train_loader, val_loader, epochs=cfg.train.epochs, patience=cfg.train.patience)

    # ── Held-out source evaluation (Phase 2 — cross-source generalisation) ──
    # The WHO-TPP operating point is chosen on validation data and then frozen:
    # the held-out source is scored against that threshold, never used to pick
    # it. Deriving the threshold from the same data it's scored on leaks the
    # test set and makes the reported sensitivity/specificity fiction.
    if held_s and args.held_out != "none":
        print(f"\n─── Held-out source [{args.held_out}] evaluation ───")
        trainer.load_best()

        val_probs_final, val_y_final = trainer.evaluate_loader(val_loader)
        val_metrics = evaluate(val_y_final, val_probs_final)
        print(f"  Validation AUC (source of frozen threshold): {val_metrics['auc']:.4f}")

        if val_metrics["who_tpp"] is None:
            print("  WHO TPP threshold undefined on validation (90% sensitivity "
                  "never reached) — cannot score held-out at a frozen operating point.")
        else:
            threshold = val_metrics["who_tpp"]["threshold"]
            held_loader, _  = make_loader(held_s, val_tf, cfg.train)
            held_probs, held_y = trainer.evaluate_loader(held_loader)
            held_auc = evaluate(held_y, held_probs)["auc"]
            point = apply_threshold(held_y, held_probs, threshold)
            ci    = bootstrap_ci(held_y, held_probs, threshold)

            print(f"  Held-out AUC: {held_auc:.4f}  "
                  f"(gap vs. val: {val_metrics['auc'] - held_auc:+.4f})")
            print_frozen_report(point, ci)


if __name__ == "__main__":
    main()
