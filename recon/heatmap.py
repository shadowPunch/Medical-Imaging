"""
Back-projects Head A's 2D localization heatmap into the synthesized 3D volume
(proposal §12's Phase 4: "heatmap back-projection ... thresholded overlay").

A single radiograph carries no depth information, so a 2D attention map cannot
be localized in depth: back-projection smears it along the projection axis,
which is what a ray actually tells you. Anything more (attenuating along the
ray, "focusing" the attention at a depth) would invent precision the input
does not contain, on a volume §11a already ships as an unvalidated
visualization.

Two honesty measures are built in:
  * attention is masked by the predicted density, so it appears on synthesized
    tissue rather than floating in air, where it would mean nothing;
  * the projection axis is *measured* against the DRR (infer_projection_axis)
    rather than assumed, because the predicted volume's axis order comes from
    the CT training target, not from any convention in this file.

Direction of flow matters for §11a: Head A -> visualization only. Nothing here
returns a value into the diagnostic path.
"""
import torch
import torch.nn.functional as F


def _resize_2d(heatmap: torch.Tensor, size: int) -> torch.Tensor:
    hm = heatmap.detach().float()
    while hm.ndim < 4:
        hm = hm.unsqueeze(0)
    hm = F.interpolate(hm, size=(size, size), mode="bilinear", align_corners=False)
    return hm.squeeze(0).squeeze(0)


def backproject(heatmap: torch.Tensor, volume_size: int, axis: int = 0) -> torch.Tensor:
    """(H, W) attention -> (V, V, V), constant along `axis` (the ray direction)."""
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2 — got {axis}")
    plane = _resize_2d(heatmap, volume_size)
    return plane.unsqueeze(axis).expand(
        *[volume_size if i == axis else -1 for i in range(3)]).contiguous()


def infer_projection_axis(volume: torch.Tensor, drr: torch.Tensor) -> tuple[int, float]:
    """Which axis the DRR integrates along, found by matching each mean-projection
    against the rendered image. Returns (axis, correlation)."""
    target = _resize_2d(drr, volume.shape[-1])
    target = (target - target.mean()).flatten()
    best, best_corr = 0, -2.0
    for axis in (0, 1, 2):
        proj = volume.float().mean(dim=axis)
        proj = _resize_2d(proj, volume.shape[-1])
        proj = (proj - proj.mean()).flatten()
        denom = proj.norm() * target.norm()
        corr = float((proj @ target) / denom) if denom > 0 else 0.0
        if corr > best_corr:
            best, best_corr = axis, corr
    return best, best_corr


# Measured on four held-out CTs with run 7: the predicted volume and the CT's own
# density both project along axis 2 (correlations 0.52-0.66 and 0.43-0.57), so this
# is a property of the training geometry rather than of any one checkpoint.
DEFAULT_PROJECTION_AXIS = 2
MIN_CORRELATION = 0.2


def choose_axis(default: int, inferred: int, correlation: float,
                min_correlation: float = MIN_CORRELATION) -> tuple[int, str]:
    """Keep the geometry-derived axis, and say so when the per-image evidence is
    weak or disagrees. A real radiograph's intensity convention need not match DRR
    attenuation, so a weak or negative per-image correlation is expected and is not
    grounds for overriding the measured default."""
    if abs(correlation) < min_correlation:
        return default, (f"weak per-image evidence (|corr|={abs(correlation):.2f} < "
                         f"{min_correlation}); keeping measured axis {default}")
    if inferred != default:
        return default, (f"per-image inference disagrees (axis {inferred}, corr "
                         f"{correlation:.2f}); keeping measured axis {default}")
    return default, ""


def attention_volume(heatmap: torch.Tensor, density: torch.Tensor, axis: int = 0,
                     density_floor: float = 0.02) -> torch.Tensor:
    """Back-projected attention masked by predicted tissue, normalized to [0, 1].

    density_floor matches the near-zero air threshold used throughout Phase 3's
    diagnostics (real CT is ~50-63% near-zero background).
    """
    density = density.detach().float().squeeze()
    if density.ndim != 3:
        raise ValueError(f"density must be a 3D volume, got shape {tuple(density.shape)}")

    smeared = backproject(heatmap, density.shape[-1], axis).to(density.device)
    masked = smeared * (density > density_floor).float()
    peak = masked.max()
    return masked / peak if peak > 0 else masked
