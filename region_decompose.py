"""
One-time preprocessing for the region-decomposition ablation: for each named
region (lungs, shoulders, spine, diaphragm, other — see
utils/lung_mask.py's region_masks()), produce a variant where ONLY that
region is visible and everything else is blanked. Training one model per
region (single source, in-domain) turns "something outside the lungs
predicts TB" into a specific mechanism: corners/other carrying signal is an
acquisition artifact, shoulders is body habitus, and so on. See
code/readme.md's Phase 2 validation results.

Run once, then train per region:
    python region_decompose.py --shenzhen /data/tb-shenzen
    python train_diagnostic.py --shenzhen /data/tb-shenzen --held-out none --region shoulders
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.lung_mask import region_masks

REGIONS = ["lungs", "shoulders", "spine", "diaphragm", "other"]


def _process_image(src: Path, dst_by_region: dict[str, Path], size: int) -> None:
    arr = np.array(Image.open(src).convert("L"))
    masks = region_masks(arr)
    for region, dst in dst_by_region.items():
        if dst.exists():
            continue
        out = np.where(masks[region], arr, 0)
        Image.fromarray(out).resize((size, size), Image.LANCZOS).save(dst, format="PNG", optimize=True)


def _process_dir(src_dir: Path, images_root: Path, size: int, desc: str) -> None:
    out_dirs = {r: images_root.parent / f"{images_root.name}_region_{r}" for r in REGIONS}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.png"))
    for src in tqdm(files, desc=desc):
        dst_by_region = {r: out_dirs[r] / src.name for r in REGIONS}
        if all(d.exists() for d in dst_by_region.values()):
            continue
        _process_image(src, dst_by_region, size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shenzhen",   type=str, default=None, help="Path to Shenzhen dataset root")
    parser.add_argument("--montgomery", type=str, default=None, help="Path to Montgomery dataset root")
    parser.add_argument("--size",       type=int, default=512,
                        help="Output resolution — match the images_{size}/ cache from preprocess.py")
    args = parser.parse_args()

    for name, root_str in [("shenzhen", args.shenzhen), ("montgomery", args.montgomery)]:
        if not root_str:
            continue
        root = Path(root_str)
        src_dir = root / f"images_{args.size}"
        if not src_dir.exists():
            src_dir = root / "images" / "images"
        images_root = root / f"images_{args.size}"
        print(f"[region-decompose] {src_dir} -> {images_root}_region_*/")
        _process_dir(src_dir, images_root, args.size, desc=name)

    print("Done.")


if __name__ == "__main__":
    main()
