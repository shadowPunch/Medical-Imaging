"""
Tests the specific mechanism proposed for the negative-class score spread
found by score_distribution_diagnosis.py: Shenzhen/Montgomery's negatives
are almost entirely *healthy* films, while TBX11K's "negative" class also
includes *sick_but_non-tb* cases (other pathology, other consolidations) —
a population the training negatives never taught the model to place. If
that's the mechanism, TBX11K's long right-tail negatives should be
overwhelmingly sick_but_non-tb, not healthy.

Splits TBX11K's held-out negatives by their raw annotation tag (healthy vs.
sick_but_non-tb — collapsed together into class 0 everywhere else in this
codebase) and scores each subgroup separately through the existing
winning-config checkpoint. No retraining.

Example:
    python -m eval.negative_composition_probe \
        --checkpoint outputs/step3_mild_lungcrop_tbx11k/best_model.pt \
        --tbx11k ../datasets/tbx11k --lung-crop \
        --output-dir outputs/tbx11k_winning_diagnosis
"""
import argparse
import json
from pathlib import Path

import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.transforms import get_val_transforms
from models.tb_model import build_model


class _PathDataset(Dataset):
    """Loads images from a flat path list — no labels needed, subgroup is tracked separately."""

    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def load_tbx11k_negative_subgroups(root: Path, variant: str = "") -> tuple[list[Path], list[str]]:
    """
    Returns (paths, subgroup) for TBX11K negatives only, subgroup in
    {'healthy', 'sick_non_tb'}, pooled across train+val — matches how
    build_splits(..., held_out='tbx11k') pools both splits into the
    held-out set.
    """
    paths, subgroups = [], []
    for split in ("train", "val"):
        img_dir = root / split / (f"img_{variant}" if variant else "img")
        ann_dir = root / split / "ann"
        for img_path in sorted(img_dir.glob("*.png")):
            ann_path = ann_dir / f"{img_path.name}.json"
            if not ann_path.exists():
                continue
            ann = json.loads(ann_path.read_text())
            tag_name = ann["tags"][0]["name"] if ann["tags"] else "no_tag"
            if tag_name == "healthy":
                paths.append(img_path)
                subgroups.append("healthy")
            elif tag_name == "sick_but_non-tb":
                paths.append(img_path)
                subgroups.append("sick_non_tb")
    return paths, subgroups


def run_inference(model, loader, device) -> np.ndarray:
    all_probs = []
    with torch.no_grad():
        for imgs in loader:
            with torch.amp.autocast(device.type):
                logits = model(imgs.to(device))
            all_probs.append(torch.sigmoid(logits).squeeze(1).float().cpu().numpy())
    return np.concatenate(all_probs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tbx11k",     type=str, required=True)
    parser.add_argument("--lung-crop",  action="store_true")
    parser.add_argument("--variant",    type=str, default=None)
    parser.add_argument("--backbone",   type=str, default="efficientnet_b0")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tail-threshold", type=float, default=0.5,
                        help="Score above which a negative counts as 'in the tail' for the composition breakdown.")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variant = args.variant if args.variant is not None else ("lungcrop" if args.lung_crop else "")

    paths, subgroups = load_tbx11k_negative_subgroups(Path(args.tbx11k), variant=variant)
    subgroups = np.array(subgroups)
    n_healthy = int((subgroups == "healthy").sum())
    n_sick = int((subgroups == "sick_non_tb").sum())
    print(f"TBX11K negatives: {len(paths)} total  ({n_healthy} healthy, {n_sick} sick_but_non-tb)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = _PathDataset(paths, get_val_transforms(args.image_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    probs = run_inference(model, loader, device)

    healthy_probs = probs[subgroups == "healthy"]
    sick_probs = probs[subgroups == "sick_non_tb"]

    print("\n─── Score distribution by negative subgroup ───")
    for name, p in [("healthy", healthy_probs), ("sick_but_non-tb", sick_probs)]:
        med, iqr = np.median(p), np.percentile(p, 75) - np.percentile(p, 25)
        print(f"  {name:16s}: n={len(p):5d}  median={med:.3f}  IQR={iqr:.3f}")

    tail_healthy = int((healthy_probs > args.tail_threshold).sum())
    tail_sick = int((sick_probs > args.tail_threshold).sum())
    tail_total = tail_healthy + tail_sick
    print(f"\n─── Tail composition (score > {args.tail_threshold}) ───")
    if tail_total == 0:
        print("  No negatives above the tail threshold.")
    else:
        print(f"  healthy:         {tail_healthy:4d} / {n_healthy:4d} "
              f"({tail_healthy / max(n_healthy, 1):.1%} of all healthy)")
        print(f"  sick_but_non-tb: {tail_sick:4d} / {n_sick:4d} "
              f"({tail_sick / max(n_sick, 1):.1%} of all sick_but_non-tb)")
        print(f"  Of {tail_total} tail negatives: {tail_sick / tail_total:.1%} are sick_but_non-tb "
              f"(base rate in the full negative pool: {n_sick / len(paths):.1%})")

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 41)
    ax.hist(healthy_probs, bins=bins, density=True, alpha=0.55, label=f"healthy (n={n_healthy})", color="seagreen")
    ax.hist(sick_probs, bins=bins, density=True, alpha=0.55, label=f"sick_but_non-tb (n={n_sick})", color="darkorange")
    ax.axvline(args.tail_threshold, color="gray", linestyle=":", label=f"tail threshold ({args.tail_threshold})")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("density")
    ax.set_title("TBX11K negative-class scores by subgroup (held-out, winning config)")
    ax.legend()
    fig.tight_layout()
    fig_path = out_dir / "negative_composition.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved -> {fig_path}")


if __name__ == "__main__":
    main()
