"""
Shortcut-detector baseline: how well can a classifier tell which *dataset* an
image came from, rather than whether it shows TB?

If this is near-perfect, the model has an easy, TB-irrelevant signal available
(scanner artifacts, corner markers, burned-in text) and single-split AUC on
any one source risks riding that shortcut rather than a thoracic one —
cross-source AUC is the metric that matters. See
TB_CXR_Diagnostic_3D_Proposal.md §5 ("the real engineering problem").

Example:
    python -m eval.shortcut_baseline \
        --shenzhen /data/tb-shenzen --montgomery /data/tb-montgomery --tbx11k /data/tbx11k

Pass --lung-crop (after running lung_crop.py) to check whether cropping to the
lung field reduces the shortcut, as a companion to the lung-crop ablation in
train_diagnostic.py.
"""
import argparse
from pathlib import Path

import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset

from data.dataset import load_montgomery, load_shenzhen, load_tbx11k
from data.transforms import get_train_transforms, get_val_transforms
from models.encoder import SharedEncoder


class SourceDataset(Dataset):
    """(image_path, source_id) pairs — the TB label is irrelevant here."""

    def __init__(self, samples: list[tuple[Path, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, source_id = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), source_id


def build_source_samples(args, variant: str) -> tuple[list[tuple[Path, int]], list[str]]:
    """
    Loads each requested source, discards its TB label, relabels every sample
    with its source index, and subsamples every source down to the smallest
    source's count — otherwise a trivial "always predict the majority source"
    classifier (TBX11K dwarfs Shenzhen/Montgomery) would look artificially good.
    """
    rng = np.random.default_rng(42)
    sources: list[tuple[str, list[tuple[Path, int]]]] = []

    if args.shenzhen:
        sources.append(("shenzhen", load_shenzhen(Path(args.shenzhen), variant=variant)))
    if args.montgomery:
        sources.append(("montgomery", load_montgomery(Path(args.montgomery), variant=variant)))
    if args.tbx11k:
        root = Path(args.tbx11k)
        tb = (load_tbx11k(root, split="train", variant=variant)
              + load_tbx11k(root, split="val", variant=variant))
        sources.append(("tbx11k", tb))

    if len(sources) < 2:
        raise ValueError("Need at least 2 sources to run the shortcut baseline.")

    n = min(len(s) for _, s in sources)
    names = [name for name, _ in sources]
    relabeled = []
    for source_id, (_, samples) in enumerate(sources):
        idx = rng.permutation(len(samples))[:n]
        relabeled.extend((samples[i][0], source_id) for i in idx)
    return relabeled, names


def split_train_val(samples: list, val_frac: float = 0.15) -> tuple[list, list]:
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(samples))
    n_val = max(1, int(len(samples) * val_frac))
    return [samples[i] for i in idx[n_val:]], [samples[i] for i in idx[:n_val]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Source-classifier shortcut baseline")
    parser.add_argument("--shenzhen",   type=str, default=None)
    parser.add_argument("--montgomery", type=str, default=None)
    parser.add_argument("--tbx11k",     type=str, default=None)
    parser.add_argument("--lung-crop",  action="store_true",
                        help="Run on lung-cropped images (see lung_crop.py)")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs",     type=int, default=5)
    parser.add_argument("--lr",         type=float, default=3e-4)
    args = parser.parse_args()

    variant = "lungcrop" if args.lung_crop else ""
    samples, names = build_source_samples(args, variant)
    train_s, val_s = split_train_val(samples)
    print(f"Sources: {names}  (balanced to {len(samples)} total, "
          f"train={len(train_s)} val={len(val_s)}, variant={variant or 'none'})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        SourceDataset(train_s, get_train_transforms(args.image_size)),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        SourceDataset(val_s, get_val_transforms(args.image_size)),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    # Same SharedEncoder as Head A, with a plain N-way source head bolted on —
    # this is a diagnostic probe, not a model variant, so it isn't added to models/.
    encoder = SharedEncoder("efficientnet_b0", pretrained=True).to(device)
    head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(encoder.out_channels, len(names)),
    ).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    # AMP, same as Trainer — this fits the 4GB GPU budget documented in
    # code/readme.md; without it this OOMs at the default batch size.
    scaler = torch.amp.GradScaler(device.type)

    for epoch in range(1, args.epochs + 1):
        encoder.train(); head.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.amp.autocast(device.type):
                logits = head(encoder(imgs)[-1])
                loss = criterion(logits, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels)
        print(f"Epoch {epoch}  train_loss={total_loss / len(train_s):.4f}")

    encoder.eval(); head.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            with torch.amp.autocast(device.type):
                logits = head(encoder(imgs.to(device))[-1])
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    chance = 1.0 / len(names)
    print(f"\nSource-classification accuracy: {acc:.4f}  (chance = {chance:.4f})")
    print(f"Confusion matrix (rows=true, cols=pred, order={names}):")
    print(confusion_matrix(all_labels, all_preds))
    if acc > 2 * chance:
        print("\n⚠ Well above chance — sources are trivially distinguishable. "
              "Single-split AUC on any one source risks riding this shortcut; "
              "treat cross-source AUC as the primary metric.")


if __name__ == "__main__":
    main()
