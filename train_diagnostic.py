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
from data.transforms import get_train_transforms, get_val_transforms
from eval.metrics import evaluate, print_report
from models.tb_model import build_model
from training.loss import LabelSmoothBCE, compute_pos_weight
from training.trainer import Trainer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_splits(args, cfg: DataConfig):
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
        samples = loader(Path(path_str))
        if name == args.held_out:
            held_out.extend(samples)
        else:
            idx   = rng.permutation(len(samples))
            n_val = max(1, int(len(samples) * 0.15))
            val_samples.extend(  [samples[i] for i in idx[:n_val]])
            train_samples.extend([samples[i] for i in idx[n_val:]])

    if args.tbx11k:
        root = Path(args.tbx11k)
        for split, target in [("train", train_samples), ("val", val_samples)]:
            s = load_tbx11k(root, split=split,
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
                        choices=["shenzhen", "montgomery", "none"],
                        help="Source reserved for cross-source generalisation eval")
    parser.add_argument("--output",    type=str, default="outputs/diagnostic")
    parser.add_argument("--backbone",  type=str, default="efficientnet_b0")
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--batch-size",type=int, default=16)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    cfg = Config(
        data=DataConfig(),
        model=ModelConfig(backbone=args.backbone, pretrained=not args.no_pretrained),
        train=TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr),
        output_dir=Path(args.output),
        seed=args.seed,
    )
    seed_everything(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────
    train_s, val_s, held_s = build_splits(args, cfg.data)
    print(f"Samples — train: {len(train_s)}  val: {len(val_s)}  held-out: {len(held_s)}")

    train_tf = get_train_transforms(cfg.data.image_size)
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
    if held_s and args.held_out != "none":
        print(f"\n─── Held-out source [{args.held_out}] evaluation ───")
        trainer.load_best()                    # evaluate with the best checkpoint
        held_loader, _ = make_loader(held_s, val_tf, cfg.train)
        probs, labels  = trainer.evaluate_loader(held_loader)
        print_report(evaluate(labels, probs))


if __name__ == "__main__":
    main()
