"""
Export a synthesized volume (and optionally a back-projected heatmap overlay)
to NRRD/NIfTI for 3D Slicer (TB_CXR_Diagnostic_3D_Proposal.md §6-7).

Both outputs are tagged as synthesized at the file level (a description
field, not just a UI label) — belt-and-braces alongside the §11a code-level
guarantee in models/tb_model.py that this data never reaches Head A.
"""
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

SYNTHESIZED_TAG = "SYNTHESIZED — not for measurement or diagnosis"


def _to_sitk_image(volume: torch.Tensor | np.ndarray, spacing_mm: float) -> sitk.Image:
    """volume: (D, H, W) or (1, D, H, W) or (1, 1, D, H, W) — any batch/channel dims are squeezed."""
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    volume = np.squeeze(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume after squeezing, got shape {volume.shape}")

    img = sitk.GetImageFromArray(volume.astype(np.float32))
    img.SetSpacing((spacing_mm, spacing_mm, spacing_mm))
    img.SetMetaData("description", SYNTHESIZED_TAG)
    return img


def export_volume(volume: torch.Tensor | np.ndarray, out_path: str | Path,
                  spacing_mm: float = 2.5) -> Path:
    """
    Writes a synthesized density volume to NRRD or NIfTI — format is inferred
    from out_path's extension (.nrrd, .nii, .nii.gz).
    """
    out_path = Path(out_path)
    img = _to_sitk_image(volume, spacing_mm)
    sitk.WriteImage(img, str(out_path))
    return out_path


def export_heatmap_overlay(heatmap: torch.Tensor | np.ndarray, out_path: str | Path,
                           spacing_mm: float = 2.5, threshold: float = 0.5) -> Path:
    """
    Writes a back-projected localization heatmap as a thresholded label
    volume (§6's "thresholded overlay"), co-registered with export_volume's
    output (same spacing/origin convention) so Slicer can load both.
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    heatmap = np.squeeze(heatmap)
    mask = (heatmap >= threshold).astype(np.uint8)

    out_path = Path(out_path)
    img = sitk.GetImageFromArray(mask)
    img.SetSpacing((spacing_mm, spacing_mm, spacing_mm))
    img.SetMetaData("description", SYNTHESIZED_TAG + " (thresholded heatmap overlay)")
    sitk.WriteImage(img, str(out_path))
    return out_path
