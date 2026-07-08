import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

# TBX11K uses per-image JSON tags (Supervisely format, not COCO).
# Maps tag name → binary label; None = exclude from training.
_TBX11K_LABEL_MAP: dict[str, int | None] = {
    "healthy":          0,
    "sick_but_non-tb":  0,
    "active_tb":        1,
    "active&latent_tb": 1,    # has an active TB component → positive
    "latent_tb":        None, # no radiographic finding; excluded by default
    "no_tag":           None, # test split is unannotated
}


# ---------------------------------------------------------------------------
# Per-source loaders — each returns a flat list of (image_path, label) pairs
# ---------------------------------------------------------------------------

def _cxr_image_dir(root: Path, size: int = 512) -> Path:
    """
    Returns the pre-resized image directory if it exists (created by preprocess.py),
    otherwise falls back to the original high-resolution directory.
    Run preprocess.py once to avoid the ~300 ms/image load cost during training.
    """
    cached = root / f"images_{size}"
    return cached if cached.exists() else root / "images" / "images"


def load_shenzhen(root: Path, size: int = 512) -> list[tuple[Path, int]]:
    """
    Shenzhen CXR: label encoded in filename suffix.
    CHNCXR_XXXX_0.png = normal, CHNCXR_XXXX_1.png = TB.
    """
    samples = []
    for p in sorted(_cxr_image_dir(root, size).glob("*.png")):
        label = int(p.stem.rsplit("_", 1)[-1])
        samples.append((p, label))
    return samples


def load_montgomery(root: Path, size: int = 512) -> list[tuple[Path, int]]:
    """
    Montgomery CXR: same filename convention as Shenzhen.
    """
    samples = []
    for p in sorted(_cxr_image_dir(root, size).glob("*.png")):
        label = int(p.stem.rsplit("_", 1)[-1])
        samples.append((p, label))
    return samples


def load_tbx11k(
    root: Path,
    split: str = "train",
    latent_as_positive: bool = False,
) -> list[tuple[Path, int]]:
    """
    TBX11K Supervisely format: one JSON per image in {split}/ann/.
    Each JSON has a 'tags' list with a single tag that names the category.

    Tag mapping:
      healthy / sick_but_non-tb → 0
      active_tb / active&latent_tb → 1
      latent_tb → excluded (or 1 if latent_as_positive=True)
      no_tag (test split) → excluded

    Skips the test split silently — it has no labels.
    """
    img_dir = root / split / "img"
    ann_dir = root / split / "ann"

    label_map = dict(_TBX11K_LABEL_MAP)
    if latent_as_positive:
        label_map["latent_tb"] = 1

    samples = []
    for img_path in sorted(img_dir.glob("*.png")):
        ann_path = ann_dir / f"{img_path.name}.json"
        if not ann_path.exists():
            continue

        ann = json.loads(ann_path.read_text())
        tag_name = ann["tags"][0]["name"] if ann["tags"] else "no_tag"
        label = label_map.get(tag_name)

        if label is None:
            continue
        samples.append((img_path, label))

    return samples


# ---------------------------------------------------------------------------
# Unified dataset class
# ---------------------------------------------------------------------------

class TBCXRDataset(Dataset):
    """
    Generic CXR dataset. Takes a flat list of (image_path, label) pairs
    produced by the load_* functions above.
    """

    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples   = samples
        self.transform = transform
        self.labels    = [s[1] for s in samples]  # exposed for WeightedRandomSampler

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def class_counts(self) -> tuple[int, int]:
        """Returns (num_negative, num_positive)."""
        pos = sum(self.labels)
        return len(self.labels) - pos, pos
