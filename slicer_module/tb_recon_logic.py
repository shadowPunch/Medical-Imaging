"""
Pure logic for the 3D Slicer module — deliberately free of any `slicer` or Qt
import so it can be unit-tested in this repo's normal test run, while the
module file around it stays a thin UI shell (proposal §7: "Keep the Slicer
module thin").

It reads what `recon/export_for_slicer.py` produced; inference happens outside
Slicer, so Slicer's Python needs no torch.
"""
import json
from pathlib import Path

import numpy as np

from recon.export import SYNTHESIZED_TAG

BANNER = "⚠ " + SYNTHESIZED_TAG

REQUIRED = {
    "volume": "volume.nrrd",
    "attention": "attention.nrrd",
    "mask": "attention_mask.nrrd",
    "report": "report.json",
}


def find_outputs(directory: str | Path) -> dict[str, Path]:
    """Locate one pipeline output set, naming whatever is missing."""
    directory = Path(directory)
    found, missing = {}, []
    for key, name in REQUIRED.items():
        path = directory / name
        (found.setdefault(key, path) if path.exists() else missing.append(name))
    if missing:
        raise FileNotFoundError(
            f"{directory} is missing {', '.join(missing)} — run recon/export_for_slicer.py first")
    return found


def load_report(directory: str | Path) -> dict:
    return json.loads(find_outputs(directory)["report"].read_text())


def probability_summary(report: dict) -> str:
    """The number, and where it came from — the volume must never be read as the
    diagnosis (proposal §11a)."""
    return (f"TB probability (Head A, diagnostic model): {report['tb_probability']:.3f}\n"
            f"The 3D volume below is a visualization aid and played no part in it.")


def attention_voxels_above(attention: np.ndarray, threshold: float) -> int:
    """Voxel count at a threshold — what the slider reports as it moves."""
    return int((np.asarray(attention) >= threshold).sum())
