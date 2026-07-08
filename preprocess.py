"""
One-time preprocessing: resize Shenzhen and Montgomery CXRs to a fixed
size so training data loading isn't bottlenecked by large PNGs.

Saves resized images to <dataset_root>/images_512/ alongside the originals.
Run once before training:
    python preprocess.py --shenzhen /path/to/tb-shenzen \
                         --montgomery /path/to/tb-montgomery \
                         --size 512
"""
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def resize_dataset(src_dir: Path, out_dir: Path, size: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.png"))
    for src in tqdm(files, desc=f"{src_dir.parent.parent.name}"):
        dst = out_dir / src.name
        if dst.exists():
            continue
        img = Image.open(src).convert("L")  # keep as grayscale — CXR has no colour info
        img = img.resize((size, size), Image.LANCZOS)
        img.save(dst, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shenzhen",   type=str, required=True)
    parser.add_argument("--montgomery", type=str, required=True)
    parser.add_argument("--size",       type=int, default=512)
    args = parser.parse_args()

    for root_str in [args.shenzhen, args.montgomery]:
        root    = Path(root_str)
        src_dir = root / "images" / "images"
        out_dir = root / f"images_{args.size}"
        print(f"Resizing {src_dir} → {out_dir}")
        resize_dataset(src_dir, out_dir, args.size)

    print("Done.")


if __name__ == "__main__":
    main()
