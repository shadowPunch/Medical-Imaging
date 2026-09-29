"""
DRR realism augmentation: makes a clean DiffDRR render look more like a real
chest radiograph, with every effect randomized per sample so real films fall
inside the training distribution rather than outside it.

Ordered by how much of the measured DRR-vs-real gap each effect addresses
(see docs/investigation-log.md's Phase 3 domain-gap section):
  1. Framing — a LIDC CT's field of view leaves the body in a small central
     box (~55% black background, hard truncated edges); real films have
     anatomy running off the frame edge. Crop to the body with a random
     margin that is often negative, i.e. inside the CT slab, to remove its
     rectangular outline (bounded so lung apices/bases stay in frame).
  2. Tone — real films are displayed through a nonlinear lookup table that
     lifts mid-tones (measured mean 0.54-0.67 vs 0.23 for a raw DRR).
  3. Film physics — scatter haze, detector blur, vendor edge enhancement,
     quantum noise.

Operates on (B, 1, H, W) images already min-max normalized to [0, 1]. Input-
only: the paired CT target is never touched.
"""
import math

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

BODY_THRESHOLD = 0.05

# (low, high) uniform ranges per effect. "mild" vs "strong" mirrors Phase 2's
# --aug-strength dial, where mild beat aggressive texture augmentation.
STRENGTHS = {
    "mild": {
        "margin": (-0.12, 0.05), "gamma": (0.45, 0.9), "scatter": (0.0, 0.15),
        "blur": (0.0, 1.0), "unsharp": (0.0, 0.6), "noise": (0.0, 0.02),
    },
    "strong": {
        "margin": (-0.20, 0.10), "gamma": (0.3, 1.1), "scatter": (0.0, 0.30),
        "blur": (0.0, 2.0), "unsharp": (0.0, 1.2), "noise": (0.0, 0.05),
    },
}


def _uniform(lo_hi, n, generator):
    lo, hi = lo_hi
    return lo + (hi - lo) * torch.rand(n, generator=generator)


def _blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return img
    k = 2 * math.ceil(3 * sigma) + 1
    return gaussian_blur(img, kernel_size=[k, k], sigma=[sigma, sigma])


def crop_to_body(x: torch.Tensor, margin: torch.Tensor) -> torch.Tensor:
    """Square crop around each sample's body bounding box, expanded by
    margin * box size on each side, resized back to the input resolution."""
    out = x.clone()
    _, _, H, W = x.shape
    for i in range(x.shape[0]):
        mask = x[i, 0] > BODY_THRESHOLD
        if not mask.any():
            continue
        rows = torch.nonzero(mask.any(dim=1)).squeeze(1)
        cols = torch.nonzero(mask.any(dim=0)).squeeze(1)
        r0, r1, c0, c1 = rows[0].item(), rows[-1].item() + 1, cols[0].item(), cols[-1].item() + 1
        side = max(r1 - r0, c1 - c0) * (1 + 2 * margin[i].item())
        side = int(min(max(side, 1), H, W))
        cy, cx = (r0 + r1) / 2, (c0 + c1) / 2
        top = int(min(max(round(cy - side / 2), 0), H - side))
        left = int(min(max(round(cx - side / 2), 0), W - side))
        crop = x[i:i + 1, :, top:top + side, left:left + side]
        out[i:i + 1] = F.interpolate(crop, size=(H, W), mode="bilinear", align_corners=False)
    return out


def realistic_drr(x: torch.Tensor, strength: str = "mild",
                  generator: torch.Generator | None = None, prob: float = 1.0) -> torch.Tensor:
    """prob: per-sample chance of applying the effect; the rest pass through
    clean, so the model keeps seeing clean DRRs during training."""
    if strength == "off":
        return x
    p = STRENGTHS[strength]
    n = x.shape[0]
    apply = torch.rand(n, generator=generator) < prob
    draw = {name: _uniform(rng, n, generator) for name, rng in p.items()}
    _, _, H, _ = x.shape

    x_clean = x
    x = crop_to_body(x, draw["margin"])
    out = []
    for i in range(n):
        xi = x[i:i + 1].clamp(0, 1) ** draw["gamma"][i].item()
        a = draw["scatter"][i].item()
        xi = (1 - a) * xi + a * _blur(xi, H / 8)
        xi = _blur(xi, draw["blur"][i].item())
        xi = xi + draw["unsharp"][i].item() * (xi - _blur(xi, max(1.0, H / 100)))
        noise = torch.randn(xi.shape, generator=generator).to(xi.device)
        xi = xi + draw["noise"][i].item() * noise
        out.append(xi.clamp(0, 1))
    return torch.where(apply.view(-1, 1, 1, 1).to(x.device), torch.cat(out), x_clean)
