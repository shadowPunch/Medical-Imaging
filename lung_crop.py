"""
One-time preprocessing: derive lung-field-crop or lung-field-complement
variants of each CXR (via utils/lung_mask.py) before the usual resize.

--mode crop (default) crops to the lung bounding box, so a classifier can't
cheat on corner markers, scanner borders, or burned-in text sitting outside
the lung fields.

--mode complement does the opposite: blanks out the lung field and keeps only
the surrounding anatomy (shoulders, diaphragm edges) and border/text region.
This is the inverse-mask ablation — train on the complement *within a single
source* to test whether the TB label is confounded with acquisition
characteristics even in-domain, independent of the cross-source setup. See
TB_CXR_Diagnostic_3D_Proposal.md §5 and code/readme.md.

Run once, after preprocess.py for Shenzhen/Montgomery (TBX11K ships pre-sized
already):
    python lung_crop.py --shenzhen /data/tb-shenzen --montgomery /data/tb-montgomery --tbx11k /data/tbx11k
    python lung_crop.py --shenzhen /data/tb-shenzen --mode complement

Then train with:
    python train_diagnostic.py ... --lung-crop
    python train_diagnostic.py --shenzhen /data/tb-shenzen --held-out none --lung-complement
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.lung_mask import lung_bbox

_VARIANT = {"crop": "lungcrop", "complement": "lungcomplement"}


def _process_image(src: Path, dst: Path, size: int, mode: str, dilate_px: int = 0) -> None:
    img = Image.open(src).convert("L")
    x0, y0, x1, y1 = lung_bbox(np.array(img), extra_px=dilate_px)
    has_box = x1 > x0 and y1 > y0

    if mode == "crop":
        result = img.crop((x0, y0, x1, y1)) if has_box else img
    else:
        arr = np.array(img)
        if has_box:
            arr = arr.copy()
            arr[y0:y1, x0:x1] = 0
        result = Image.fromarray(arr)

    result = result.resize((size, size), Image.LANCZOS)
    result.save(dst, format="PNG", optimize=True)


def _process_dir(src_dir: Path, out_dir: Path, size: int, mode: str, desc: str, dilate_px: int = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.png"))
    for src in tqdm(files, desc=desc):
        dst = out_dir / src.name
        if dst.exists():
            continue
        _process_image(src, dst, size, mode, dilate_px)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shenzhen",   type=str, default=None, help="Path to Shenzhen dataset root")
    parser.add_argument("--montgomery", type=str, default=None, help="Path to Montgomery dataset root")
    parser.add_argument("--tbx11k",     type=str, default=None, help="Path to TBX11K dataset root")
    parser.add_argument("--mode",       type=str, default="crop", choices=["crop", "complement"])
    parser.add_argument("--dilate-px",  type=int, default=0,
                        help="Extra flat pixel padding beyond the 5%% margin — dilates the mask to "
                             "check whether an ablation result survives a more generous lung box "
                             "(chest segmentation models systematically under-cover the apices and "
                             "costophrenic angles)")
    parser.add_argument("--size",       type=int, default=512,
                        help="Output resolution — match the images_{size}/ cache from preprocess.py")
    args = parser.parse_args()
    variant = _VARIANT[args.mode] + (f"_d{args.dilate_px}" if args.dilate_px else "")

    for name, root_str in [("shenzhen", args.shenzhen), ("montgomery", args.montgomery)]:
        if not root_str:
            continue
        root = Path(root_str)
        # Prefer preprocess.py's resized cache (much faster to read than the
        # ~4900x4000 originals); the segmentation model resizes to 512
        # internally anyway, so this costs no accuracy.
        src_dir = root / f"images_{args.size}"
        if not src_dir.exists():
            src_dir = root / "images" / "images"
        out_dir = root / f"images_{args.size}_{variant}"
        print(f"[{args.mode}] {src_dir} -> {out_dir}")
        _process_dir(src_dir, out_dir, args.size, args.mode, desc=name, dilate_px=args.dilate_px)

    if args.tbx11k:
        root = Path(args.tbx11k)
        for split in ("train", "val"):
            src_dir = root / split / "img"
            if not src_dir.exists():
                continue
            out_dir = root / split / f"img_{variant}"
            print(f"[{args.mode}] {src_dir} -> {out_dir}")
            _process_dir(src_dir, out_dir, args.size, args.mode, desc=f"tbx11k/{split}", dilate_px=args.dilate_px)

    print("Done.")


if __name__ == "__main__":
    main()
