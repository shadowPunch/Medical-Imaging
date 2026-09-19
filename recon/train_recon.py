"""
Phase 3 training: DiffDRR-paired supervision (LIDC-IDRI CT -> synthetic DRR)
plus shape induction on unpaired real CXRs (Sizikova et al. 2022 —
TB_CXR_Diagnostic_3D_Proposal.md §6), sharing one encoder with Head A.

Two loss terms per step, alternating by source:
  - Paired (LIDC-IDRI): render a real DRR from a real CT at a random pose,
    predict a volume from that DRR, compare directly against the CT's own
    density (resampled to match ReconHead's output size). Ground truth
    exists here, so this is a normal supervised voxel loss.
  - Unpaired shape induction (Shenzhen/Montgomery/TBX11K): no CT exists for
    these, so there's nothing to compare the predicted volume to directly.
    Instead, re-project the predicted volume back to 2D through the same
    differentiable DRR geometry and compare against the *original input*
    CXR — consistency between what the model predicted in 3D and what it
    was shown in 2D, with no 3D ground truth required. This is what lets
    the encoder see real (non-DRR) CXRs during Phase 3 training at all.

This is a first-pass implementation of the shape-induction *paradigm*
described in papers/2208.10937v2.pdf, not a line-by-line reproduction of
that paper's exact loss formulation — revisit against the paper before
trusting the loss weighting for real training. Verified to run end-to-end
on real data (LIDC-IDRI dev subset + Shenzhen) at small scale; not run to
convergence — that needs the cloud GPU this environment doesn't have (see
code/readme.md's Phase 3 status).

Example (smoke test):
    python -m recon.train_recon --lidc-dir ../datasets/lidc-idri/dicom \
        --shenzhen ../datasets/tb-shenzen --steps 5 --volume-size 32
"""
import argparse
import random
from pathlib import Path

import torch
import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)

import torch.nn.functional as F
from PIL import Image

from data.dataset import load_montgomery, load_shenzhen, load_tbx11k
from data.transforms import _MEAN, _STD
from models.tb_model import build_model
from recon.ct_data import build_drr, load_ct_volume_cached, random_pose
from recon.drr_realism import STRENGTHS, realistic_drr

_MEAN_T = torch.tensor(_MEAN).view(1, 3, 1, 1)
_STD_T = torch.tensor(_STD).view(1, 3, 1, 1)


def drr_to_model_input(drr_img: torch.Tensor, image_size: int, realism: str = "off",
                       generator: torch.Generator | None = None,
                       realism_prob: float = 1.0) -> torch.Tensor:
    """(B,1,H,W) raw DRR intensities -> (B,3,image_size,image_size) ImageNet-normalized,
    matching data/transforms.py's convention so the shared encoder sees a consistent
    input distribution regardless of whether it came from a real CXR or a synthetic DRR.
    realism: see recon/drr_realism.py — makes the DRR look like a real film."""
    x = drr_img
    x = (x - x.amin(dim=(2, 3), keepdim=True)) / (
        x.amax(dim=(2, 3), keepdim=True) - x.amin(dim=(2, 3), keepdim=True) + 1e-8)
    x = realistic_drr(x, realism, generator, realism_prob)
    x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    return (x - _MEAN_T.to(x.device)) / _STD_T.to(x.device)


def real_cxr_to_model_input(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    x = torch.from_numpy(__import__("numpy").array(img)).float().permute(2, 0, 1) / 255.0
    return ((x - _MEAN_T[0]) / _STD_T[0]).unsqueeze(0)


def paired_step(model, ct_series: Path, image_size: int, volume_size: int,
                device: torch.device, loss_fn: str = "mse", realism: str = "off",
                realism_prob: float = 1.0) -> torch.Tensor:
    """
    loss_fn: 'mse' (original) or 'l1'. Real CT density is highly skewed —
    roughly half the voxels are near-zero air/background — and plain MSE on
    a target like that has a well-known failure mode: the safest way to
    minimize squared error under uncertainty is a smoothed prediction near
    the target's mean everywhere, not sparse structure matching the true
    per-voxel values. L1 penalizes large errors less quadratically and is
    known to produce sparser, less-regression-to-the-mean solutions on
    skewed targets — testing whether that's actually what's happening here
    (see code/readme.md's Phase 3 section for the diagnostic that motivated
    this).
    """
    subject = load_ct_volume_cached(ct_series)
    drr = build_drr(subject, height=image_size, device=device)
    rot, trans = random_pose(device=device)
    drr_img = drr(rot, trans, parameterization="euler_angles", convention="ZXY")

    x = drr_to_model_input(drr_img, image_size, realism, realism_prob=realism_prob)
    pred_volume = model.forward_recon(x)  # (1, 1, V, V, V)

    target = subject.density.data.to(device).squeeze(0)  # torchio stores (1, D, H, W) -> (D, H, W)
    target = F.interpolate(target[None, None], size=(volume_size,) * 3,
                           mode="trilinear", align_corners=False)
    if loss_fn == "l1":
        return F.l1_loss(pred_volume, target)
    return F.mse_loss(pred_volume, target)


def shape_induction_step(model, cxr_path: Path, canonical_drr, canonical_volume_shape,
                         image_size: int, device: torch.device) -> torch.Tensor:
    x = real_cxr_to_model_input(cxr_path, image_size).to(device)
    pred_volume = model.forward_recon(x)  # (1, 1, V, V, V) — V = ReconHead's fixed cube size

    # canonical_drr's ray geometry (affine, grid dims) was fixed at build_drr()
    # time from the canonical CT's own resampled shape, which is NOT generally
    # a cube (mm-spacing resampling gives whatever voxel count each volume's
    # physical extent implies). Substituting a wrong-shaped density silently
    # scrambles the Siddon renderer's indexing rather than erroring — caught
    # this empirically (identical loss across different random inits, and
    # directly changing density values not changing the render, was the
    # tell). Resize to match before substituting; interpolate is
    # differentiable so this doesn't block the gradient.
    resized = F.interpolate(pred_volume, size=canonical_volume_shape,
                            mode="trilinear", align_corners=False)
    canonical_drr.density = resized.squeeze(0).squeeze(0)
    rot, trans = random_pose(device=device, rotation_deg=0.0, translation_mm=0.0)  # canonical AP pose
    reprojected = canonical_drr(rot, trans, parameterization="euler_angles", convention="ZXY")

    def _norm(t):
        return (t - t.mean()) / (t.std() + 1e-8)
    return F.mse_loss(_norm(reprojected), _norm(F.interpolate(
        x[:, :1], size=reprojected.shape[-2:], mode="bilinear", align_corners=False)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lidc-dir",   type=str, required=True, help="datasets/lidc-idri/dicom")
    parser.add_argument("--shenzhen",   type=str, default=None)
    parser.add_argument("--montgomery", type=str, default=None)
    parser.add_argument("--tbx11k",     type=str, default=None)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--volume-size", type=int, default=128)
    parser.add_argument("--steps",      type=int, default=5)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--shape-induction-weight", type=float, default=0.5)
    parser.add_argument("--drr-realism", type=str, default="off",
                        choices=["off", *STRENGTHS],
                        help="Film-like augmentation of the paired DRR input (recon/drr_realism.py) "
                             "to shrink the DRR -> real CXR domain gap. Target CT is untouched.")
    parser.add_argument("--drr-realism-prob", type=float, default=1.0,
                        help="Chance each paired DRR gets the realism effect. Below 1.0 keeps "
                             "clean DRRs in training — a realism-only model degraded on them.")
    parser.add_argument("--paired-loss", type=str, default="mse", choices=["mse", "l1"],
                        help="See paired_step's docstring — testing whether L1 avoids the "
                             "regression-to-the-mean-density behavior MSE showed on held-out CT.")
    parser.add_argument("--diagnose-every", type=int, default=0,
                        help="If >0, every N steps print the fraction of the current "
                             "prediction's voxels below 0.02 density (target CT is ~50-60%% "
                             "near-zero air/background; a healthy fit should trend toward that, "
                             "not stay at 0%%).")
    parser.add_argument("--output", type=str, default=None,
                        help="If set, saves {'model','optimizer','step'} here at the end — "
                             "same format recon/eval_paired.py expects. This script was a smoke "
                             "test with no checkpoint saving until the bias-init/caching fixes "
                             "made local runs long enough to be worth evaluating.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ct_series = sorted(Path(args.lidc_dir).glob("*"))
    if not ct_series:
        raise ValueError(f"No CT series found in {args.lidc_dir}")

    cxr_paths = []
    if args.shenzhen:
        cxr_paths += [p for p, _ in load_shenzhen(Path(args.shenzhen))]
    if args.montgomery:
        cxr_paths += [p for p, _ in load_montgomery(Path(args.montgomery))]
    if args.tbx11k:
        cxr_paths += [p for p, _ in load_tbx11k(Path(args.tbx11k), split="train")]
    print(f"CT series: {len(ct_series)}  unpaired CXRs: {len(cxr_paths)}")

    model = build_model("efficientnet_b0", pretrained=True, with_recon=True,
                        volume_size=args.volume_size).to(device)
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.recon_head.parameters()), lr=args.lr)

    # One reusable canonical geometry for shape-induction re-projection —
    # its own density buffer gets overwritten every step (see shape_induction_step).
    # canonical_volume_shape is captured now, before any substitution, since
    # canonical_drr.density itself gets overwritten each step and can't be
    # trusted to still reflect the original geometry after the first step.
    canonical_subject = load_ct_volume_cached(ct_series[0])
    canonical_drr = build_drr(canonical_subject, height=args.image_size, device=device)
    canonical_volume_shape = tuple(canonical_drr.density.shape)

    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()

        loss_paired = paired_step(model, random.choice(ct_series), args.image_size,
                                  args.volume_size, device, loss_fn=args.paired_loss,
                                  realism=args.drr_realism, realism_prob=args.drr_realism_prob)
        loss = loss_paired
        log = f"step {step}/{args.steps}  paired={loss_paired.item():.4f}"

        if cxr_paths:
            loss_shape = shape_induction_step(model, random.choice(cxr_paths), canonical_drr,
                                              canonical_volume_shape, args.image_size, device)
            loss = loss + args.shape_induction_weight * loss_shape
            log += f"  shape_induction={loss_shape.item():.4f}"

        loss.backward()
        optimizer.step()
        print(log + f"  total={loss.item():.4f}")

        if args.diagnose_every and step % args.diagnose_every == 0:
            model.eval()
            with torch.no_grad():
                probe_subject = load_ct_volume_cached(random.choice(ct_series))
                probe_drr = build_drr(probe_subject, height=args.image_size, device=device)
                probe_rot, probe_trans = random_pose(device=device, rotation_deg=0.0, translation_mm=0.0)
                probe_img = probe_drr(probe_rot, probe_trans, parameterization="euler_angles", convention="ZXY")
                probe_pred = model.forward_recon(drr_to_model_input(probe_img, args.image_size))
                target_probe = F.interpolate(
                    probe_subject.density.data.to(device)[None], size=(args.volume_size,) * 3,
                    mode="trilinear", align_corners=False)
            frac_zero_pred = (probe_pred < 0.02).float().mean().item()
            frac_zero_target = (target_probe < 0.02).float().mean().item()
            print(f"  [diagnostic] pred min={probe_pred.min().item():.4f} mean={probe_pred.mean().item():.4f} "
                  f"frac<0.02={frac_zero_pred:.1%}   target frac<0.02={frac_zero_target:.1%}")
            model.train()

    print("\nDone (smoke run — not trained to convergence; see recon/train_recon.py's docstring).")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "step": args.steps}, out_path)
        print(f"Saved checkpoint -> {out_path}")


if __name__ == "__main__":
    main()
