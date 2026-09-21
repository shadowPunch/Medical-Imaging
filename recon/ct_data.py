"""
CT volume loading and DRR pair generation for Phase 3 (reconstruction head).

Wraps DiffDRR (github.com/eigenvivek/DiffDRR) — verified against real
LIDC-IDRI data in datasets/lidc-idri/dicom/: DiffDRR's read() loads a DICOM
series directory directly (no manual conversion), and a 128-ish-voxel
resample (resample_target in mm, passed to torchio.transforms.Resample)
keeps DRR generation under ~2GB peak GPU memory — fits the local 4GB card,
matching the proposal's §9 128³ latent/patch-based training target. Full-
resolution volumes OOM on this card (verified: >3.7GB for a ~512x512xN
volume) — always load through load_ct_volume with resample_target set.
"""
from pathlib import Path

import torch
from diffdrr.data import read
from diffdrr.drr import DRR

# mm — resamples a typical ~350mm chest CT FOV down to roughly 128 voxels/axis.
DEFAULT_RESAMPLE_MM = 2.5


def load_ct_volume(series_dir: str | Path, resample_mm: float = DEFAULT_RESAMPLE_MM):
    """Loads one DICOM series directory into a DiffDRR Subject (RAS+, HU->density)."""
    return read(str(series_dir), resample_target=resample_mm)


def load_ct_volume_cached(series_dir: str | Path, resample_mm: float = DEFAULT_RESAMPLE_MM,
                          cache_dir: str | Path | None = None):
    """
    Same as load_ct_volume, but caches the loaded+resampled Subject to disk
    (torch.save/load, not just the density array — the Subject also carries
    the affine/geometry build_drr needs). DICOM series read + resample is
    ~7-9s/call (measured), dwarfing the ~0.3-0.4s model forward+backward —
    at that ratio, training step count is bottlenecked by disk I/O, not GPU
    compute. A cache hit is essentially free by comparison.

    cache_dir defaults to a `.cache/` directory next to series_dir's parent
    (i.e. alongside the `dicom/` directory itself), gitignored like the
    DICOM data it derives from.
    """
    series_dir = Path(series_dir)
    if series_dir.suffix == ".pt":
        # Already a resampled Subject — the Kaggle/Colab runs ship these as a
        # dataset so the VM needs neither the DICOM tree nor network access.
        return torch.load(series_dir, weights_only=False)
    if cache_dir is None:
        cache_dir = series_dir.parent.parent / ".cache" / f"resample_{resample_mm}mm"
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{series_dir.name}.pt"

    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    subject = load_ct_volume(series_dir, resample_mm)
    torch.save(subject, cache_path)
    return subject


def build_drr(subject, height: int = 320, sdd: float = 1020.0, delx: float = 2.0,
              device: str | torch.device = "cuda") -> DRR:
    """
    Builds a differentiable DRR renderer for one loaded subject.
    sdd: source-to-detector distance (mm) — 1020mm is a standard clinical AP
    chest radiograph geometry. delx: detector pixel spacing (mm).
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    return DRR(subject, sdd=sdd, height=height, delx=delx).to(device)


def random_pose(batch_size: int = 1, device: str | torch.device = "cuda",
                rotation_deg: float = 5.0, translation_mm: float = 20.0,
                base_translation_y: float = 850.0):
    """
    Samples a small random perturbation around a standard AP chest pose —
    matches acquisition variability (patient positioning) without wandering
    into non-physical geometry. `random_pose()` (defaults) reproduces the
    canonical AP pose exactly (zero perturbation) for a fixed reference view.
    """
    rot = torch.deg2rad(torch.empty(batch_size, 3, device=device).uniform_(
        -rotation_deg, rotation_deg))
    trans = torch.empty(batch_size, 3, device=device).uniform_(
        -translation_mm, translation_mm)
    trans[:, 1] += base_translation_y  # keep the AP source-to-isocenter offset
    return rot, trans


def generate_drr_pair(series_dir: str | Path, height: int = 320,
                      resample_mm: float = DEFAULT_RESAMPLE_MM,
                      device: str | torch.device = "cuda", randomize_pose: bool = True):
    """
    Loads one CT series and renders a single DRR — a (volume, drr_image, pose)
    training pair for DiffDRR-supervised reconstruction training.
    Returns (subject, drr_image [1,1,H,W], rotations [1,3], translations [1,3]).
    """
    subject = load_ct_volume(series_dir, resample_mm)
    drr = build_drr(subject, height=height, device=device)
    dev = next(drr.parameters(), torch.empty(0)).device if list(drr.parameters()) else device
    rot, trans = random_pose(device=dev) if randomize_pose else random_pose(
        device=dev, rotation_deg=0.0, translation_mm=0.0)
    img = drr(rot, trans, parameterization="euler_angles", convention="ZXY")
    return subject, img, rot, trans
