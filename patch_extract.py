"""
One-time preprocessing for the single-patch test: crop a small, fixed-location
patch — literally the same pixel coordinates for every image, no per-image
segmentation — from a corner well away from any anatomy. Tests whether the
region-decomposition signal is a pervasive high-frequency processing
signature (sharpening kernel, compression, detector characteristics) present
in every pixel, rather than anatomy-specific: if a bare corner patch predicts
TB status, region decomposition was measuring the same thing five times. See
code/readme.md's Phase 2 validation results.

Example:
    python patch_extract.py --shenzhen /data/tb-shenzen --loc top_left --patch-size 96
    python train_diagnostic.py --shenzhen /data/tb-shenzen --held-out none \
        --variant patch_top_left_96 --image-size 96
"""
import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

_LOCATIONS = {
    "top_left":     lambda w, h, s: (0, 0),
    "top_right":    lambda w, h, s: (w - s, 0),
    "bottom_left":  lambda w, h, s: (0, h - s),
    "bottom_right": lambda w, h, s: (w - s, h - s),
}


def _process_dir(src_dir: Path, out_dir: Path, loc: str, patch_size: int, desc: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.png"))
    for src in tqdm(files, desc=desc):
        dst = out_dir / src.name
        if dst.exists():
            continue
        img = Image.open(src).convert("L")
        w, h = img.size
        x0, y0 = _LOCATIONS[loc](w, h, patch_size)
        img.crop((x0, y0, x0 + patch_size, y0 + patch_size)).save(dst, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shenzhen",   type=str, default=None, help="Path to Shenzhen dataset root")
    parser.add_argument("--montgomery", type=str, default=None, help="Path to Montgomery dataset root")
    parser.add_argument("--loc",        type=str, required=True, choices=list(_LOCATIONS))
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--size",       type=int, default=512,
                        help="Source cache resolution (images_{size}/ from preprocess.py)")
    args = parser.parse_args()
    variant = f"patch_{args.loc}_{args.patch_size}"

    for name, root_str in [("shenzhen", args.shenzhen), ("montgomery", args.montgomery)]:
        if not root_str:
            continue
        root = Path(root_str)
        src_dir = root / f"images_{args.size}"
        if not src_dir.exists():
            src_dir = root / "images" / "images"
        out_dir = root / f"images_{args.size}_{variant}"
        print(f"[patch:{args.loc}] {src_dir} -> {out_dir}")
        _process_dir(src_dir, out_dir, args.loc, args.patch_size, desc=name)

    print("Done.")


if __name__ == "__main__":
    main()
