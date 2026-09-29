"""
One-time preprocessing for the high-pass residual test: subtract a heavy
Gaussian blur from each image (on the *original* image, before any masking,
so the residual isn't dominated by the mask rectangle's own hard edge), then
blank the lung region in the residual. Anatomy mostly disappears in a
high-pass residual; acquisition-level texture (sharpening kernel, compression
blocking, detector noise) survives — companion to the single-patch test.
See docs/investigation-log.md's Phase 2 validation results.

Example:
    python highpass_complement.py --shenzhen /data/tb-shenzen --blur-radius 20
    python train_diagnostic.py --shenzhen /data/tb-shenzen --held-out none \
        --variant highpass_r20 --image-size 320
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

from utils.lung_mask import lung_bbox


def _process_image(src: Path, dst: Path, size: int, blur_radius: int) -> None:
    img = Image.open(src).convert("L")
    arr = np.array(img).astype(np.int16)
    blurred = np.array(img.filter(ImageFilter.GaussianBlur(blur_radius))).astype(np.int16)
    # Residual centered at 128 (the "no local deviation" point) so it stays a valid uint8 image.
    residual = np.clip(arr - blurred + 128, 0, 255).astype(np.uint8)

    x0, y0, x1, y1 = lung_bbox(np.array(img))
    if x1 > x0 and y1 > y0:
        residual[y0:y1, x0:x1] = 128  # neutral, not 0 — 0 would itself be a spurious high-pass edge

    Image.fromarray(residual).resize((size, size), Image.LANCZOS).save(dst, format="PNG", optimize=True)


def _process_dir(src_dir: Path, out_dir: Path, size: int, blur_radius: int, desc: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.png"))
    for src in tqdm(files, desc=desc):
        dst = out_dir / src.name
        if dst.exists():
            continue
        _process_image(src, dst, size, blur_radius)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shenzhen",   type=str, default=None, help="Path to Shenzhen dataset root")
    parser.add_argument("--montgomery", type=str, default=None, help="Path to Montgomery dataset root")
    parser.add_argument("--blur-radius", type=int, default=20)
    parser.add_argument("--size",       type=int, default=512,
                        help="Output resolution — match the images_{size}/ cache from preprocess.py")
    args = parser.parse_args()
    variant = f"highpass_r{args.blur_radius}"

    for name, root_str in [("shenzhen", args.shenzhen), ("montgomery", args.montgomery)]:
        if not root_str:
            continue
        root = Path(root_str)
        src_dir = root / f"images_{args.size}"
        if not src_dir.exists():
            src_dir = root / "images" / "images"
        out_dir = root / f"images_{args.size}_{variant}"
        print(f"[highpass r={args.blur_radius}] {src_dir} -> {out_dir}")
        _process_dir(src_dir, out_dir, args.size, args.blur_radius, desc=name)

    print("Done.")


if __name__ == "__main__":
    main()
