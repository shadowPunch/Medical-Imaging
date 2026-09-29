"""
One-time preprocessing: crop to the lung bounding box, apply CLAHE, then
z-score using ONLY in-mask (true lung) pixels — not the whole crop rectangle,
which still contains ribs/mediastinum/border. Normalizing on in-mask
statistics removes the global exposure/contrast offset that differs between
sites without border regions dragging the statistics around. Intervention #1
(lung-crop, already in place) + in-mask normalization, per the robustness
plan in docs/investigation-log.md's Phase 2 validation results — the new reference
point before texture augmentation (intervention #2).

Run once, then train (use --held-out for the leave-one-source-out pattern
already built into train_diagnostic.py — train on 2 sources, test on 1):
    python lung_crop_clahe.py --shenzhen /data/tb-shenzen --montgomery /data/tb-montgomery --tbx11k /data/tbx11k
    python train_diagnostic.py --shenzhen ... --montgomery ... --tbx11k ... \
        --held-out montgomery --variant lungcrop_clahe
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.lung_mask import lung_mask_and_bbox


def _process_image(src: Path, dst: Path, size: int) -> None:
    img = Image.open(src).convert("L")
    arr = np.array(img)
    mask, (x0, y0, x1, y1) = lung_mask_and_bbox(arr)
    if not (x1 > x0 and y1 > y0):
        Image.fromarray(arr).resize((size, size), Image.LANCZOS).save(dst, format="PNG", optimize=True)
        return

    crop = arr[y0:y1, x0:x1]
    cmask = mask[y0:y1, x0:x1]
    crop = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop)

    if cmask.sum() < 100:  # segmentation failure inside the crop — CLAHE only, no z-score
        out = crop
    else:
        vals = crop[cmask].astype(np.float32)
        mu, sd = vals.mean(), vals.std() + 1e-6
        z = np.clip((crop.astype(np.float32) - mu) / sd, -3, 3)
        out = ((z + 3) / 6 * 255).astype(np.uint8)

    Image.fromarray(out).resize((size, size), Image.LANCZOS).save(dst, format="PNG", optimize=True)


def _process_dir(src_dir: Path, out_dir: Path, size: int, desc: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.png"))
    for src in tqdm(files, desc=desc):
        dst = out_dir / src.name
        if dst.exists():
            continue
        _process_image(src, dst, size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shenzhen",   type=str, default=None, help="Path to Shenzhen dataset root")
    parser.add_argument("--montgomery", type=str, default=None, help="Path to Montgomery dataset root")
    parser.add_argument("--tbx11k",     type=str, default=None, help="Path to TBX11K dataset root")
    parser.add_argument("--size",       type=int, default=512,
                        help="Output resolution — match the images_{size}/ cache from preprocess.py")
    args = parser.parse_args()
    variant = "lungcrop_clahe"

    for name, root_str in [("shenzhen", args.shenzhen), ("montgomery", args.montgomery)]:
        if not root_str:
            continue
        root = Path(root_str)
        src_dir = root / f"images_{args.size}"
        if not src_dir.exists():
            src_dir = root / "images" / "images"
        out_dir = root / f"images_{args.size}_{variant}"
        print(f"[lungcrop_clahe] {src_dir} -> {out_dir}")
        _process_dir(src_dir, out_dir, args.size, desc=name)

    if args.tbx11k:
        root = Path(args.tbx11k)
        for split in ("train", "val"):
            src_dir = root / split / "img"
            if not src_dir.exists():
                continue
            out_dir = root / split / f"img_{variant}"
            print(f"[lungcrop_clahe] {src_dir} -> {out_dir}")
            _process_dir(src_dir, out_dir, args.size, desc=f"tbx11k/{split}")

    print("Done.")


if __name__ == "__main__":
    main()
