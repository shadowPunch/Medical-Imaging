"""
Lung-field bounding box via a pretrained segmentation model.

Uses torchxrayvision's off-the-shelf ChestX-Det PSPNet — no lung-mask training
data of our own needed. Used by lung_crop.py to crop out corner markers,
scanner borders, and burned-in text that a classifier could latch onto as a
source/scanner shortcut instead of the actual thoracic signal (see the
cross-source generalisation risk in TB_CXR_Diagnostic_3D_Proposal.md §5).
"""
import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
from PIL import Image, ImageFilter

_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_transform = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(512),
])

# Grouped PSPNet organ classes for the region-decomposition ablation —
# which specific out-of-lung region carries TB-predictive signal.
_REGION_GROUPS = {
    "shoulders": ["Left Clavicle", "Right Clavicle", "Left Scapula", "Right Scapula"],
    "spine":     ["Spine", "Mediastinum", "Aorta", "Weasand"],
    "diaphragm": ["Facies Diaphragmatica"],
}


def _get_model():
    global _model
    if _model is None:
        _model = xrv.baseline_models.chestx_det.PSPNet().to(_device)
        _model.eval()
    return _model


def _segment(img_gray: np.ndarray) -> np.ndarray:
    """Runs PSPNet, returns a (14, 512, 512) bool array — one channel per model.targets class."""
    model = _get_model()
    img = xrv.datasets.normalize(img_gray.astype(np.float32), maxval=255)
    img = img[None, ...]  # (1, H, W) — torchxrayvision's channel-first convention
    img = _transform(img)  # (1, 512, 512): center-cropped to square, then resized
    img_t = torch.from_numpy(img).unsqueeze(0).float().to(_device)  # (1, 1, 512, 512)
    with torch.no_grad():
        pred = model(img_t)
    return (pred[0] > 0.5).cpu().numpy()


def _to_original_space(mask512: np.ndarray, h: int, w: int) -> np.ndarray:
    """Maps a (512, 512) bool mask back to original (h, w) pixel space, undoing
    XRayCenterCrop's square center-crop then XRayResizer's resize to 512."""
    crop_size = min(h, w)
    crop_x0, crop_y0 = (w - crop_size) // 2, (h - crop_size) // 2
    resized = Image.fromarray((mask512 * 255).astype(np.uint8)).resize(
        (crop_size, crop_size), Image.NEAREST)
    canvas = np.zeros((h, w), dtype=bool)
    canvas[crop_y0:crop_y0 + crop_size, crop_x0:crop_x0 + crop_size] = np.array(resized) > 127
    return canvas


def _bbox_from_mask(mask: np.ndarray, h: int, w: int, margin: float, extra_px: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, w, h
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mx, my = int((x1 - x0) * margin) + extra_px, int((y1 - y0) * margin) + extra_px
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(w, x1 + mx), min(h, y1 + my)
    return x0, y0, x1, y1


def lung_mask_and_bbox(
    img_gray: np.ndarray, margin: float = 0.05, extra_px: int = 0
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Both the precise boolean lung mask (Left Lung | Right Lung, original (H, W)
    pixel space) and its padded bounding box, from a single segmentation
    pass. The mask is the true segmentation, not just the bbox rectangle —
    used for in-mask normalization (CLAHE/z-scoring computed only from real
    lung pixels, which still excludes ribs/mediastinum/border sitting inside
    the crop rectangle).
    """
    h, w = img_gray.shape
    model = _get_model()
    pred = _segment(img_gray)
    mask512 = pred[model.targets.index("Left Lung")] | pred[model.targets.index("Right Lung")]
    mask = _to_original_space(mask512, h, w)
    bbox = _bbox_from_mask(mask, h, w, margin, extra_px)
    return mask, bbox


def lung_bbox(img_gray: np.ndarray, margin: float = 0.05, extra_px: int = 0) -> tuple[int, int, int, int]:
    """
    img_gray: (H, W) uint8 grayscale array, values in [0, 255].
    Returns (x0, y0, x1, y1) in the *original* image's pixel coordinates,
    padded by `margin` (fraction of box size) plus a flat `extra_px` on each
    side. `extra_px` exists because chest segmentation models systematically
    under-cover the lung apices (above the clavicles — the single most
    TB-predominant region) and the costophrenic angles (where pleural
    effusion, a TB finding, shows up): a tight mask can leave real lung
    signal sitting just outside the "cropped" box, or just inside the
    "blanked" box for the complement — dilating tests whether an ablation
    result survives a more generous mask.
    Falls back to the full image if segmentation finds no lung pixels —
    the crop then degrades to a no-op for that image rather than crashing
    or guessing.
    """
    h, w = img_gray.shape
    model = _get_model()
    pred = _segment(img_gray)
    mask = pred[model.targets.index("Left Lung")] | pred[model.targets.index("Right Lung")]

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, w, h

    # Map the bbox from the 512x512 segmentation space back to original pixel
    # coordinates: XRayCenterCrop took the largest centered square first, so
    # undo that crop offset before undoing the resize.
    crop_size = min(h, w)
    crop_x0, crop_y0 = (w - crop_size) // 2, (h - crop_size) // 2
    scale = crop_size / 512

    x0 = crop_x0 + int(xs.min() * scale)
    x1 = crop_x0 + int(xs.max() * scale)
    y0 = crop_y0 + int(ys.min() * scale)
    y1 = crop_y0 + int(ys.max() * scale)

    mx, my = int((x1 - x0) * margin) + extra_px, int((y1 - y0) * margin) + extra_px
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(w, x1 + mx), min(h, y1 + my)
    return x0, y0, x1, y1


def region_masks(img_gray: np.ndarray, dilate_px_512: int = 8) -> dict[str, np.ndarray]:
    """
    Boolean masks, in original-image pixel space, decomposing the frame into
    disjoint anatomical/positional regions for the region-decomposition
    ablation (which specific out-of-lung region carries TB-predictive
    signal): "lungs", each group in _REGION_GROUPS (shoulders, spine,
    diaphragm), and "other" (corners/background/marker text — everything not
    covered by any labeled organ nor the lungs). `dilate_px_512` grows each
    group mask slightly (PSPNet's structures, especially the diaphragm line,
    are thin) before computing "other" as the complement of their union.
    """
    h, w = img_gray.shape
    model = _get_model()
    pred = _segment(img_gray)  # (14, 512, 512) bool

    lungs512 = pred[model.targets.index("Left Lung")] | pred[model.targets.index("Right Lung")]
    masks = {"lungs": _to_original_space(lungs512, h, w)}

    covered512 = lungs512.copy()
    for group, organs in _REGION_GROUPS.items():
        m512 = np.zeros_like(lungs512)
        for organ in organs:
            m512 |= pred[model.targets.index(organ)]
        if dilate_px_512 and m512.any():
            m512 = np.array(Image.fromarray((m512 * 255).astype(np.uint8))
                            .filter(ImageFilter.MaxFilter(2 * dilate_px_512 + 1))) > 127
        masks[group] = _to_original_space(m512, h, w)
        covered512 |= m512

    masks["other"] = _to_original_space(~covered512, h, w)
    return masks
