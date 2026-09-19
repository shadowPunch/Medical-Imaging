"""
Phase 3 quantitative validation against held-out paired CT — the proposal's
§10 metric this pipeline hasn't had until now. Every prior Phase 3 result in
code/readme.md demonstrates the pipeline *trains* stably; this is the first
script that asks whether the output is a thorax or plausible-looking noise.

For each held-out CT series (must be disjoint from whatever trained the
checkpoint — see datasets/lidc-idri/README.md's manifest_heldout*.tcia):
  1. Render a canonical-pose DRR from the real CT (the model's input).
  2. Predict a volume from that DRR via the trained ReconHead.
  3. PSNR + SSIM against the real CT's own resampled density (ground truth).
  4. LPIPS, computed per-axial-slice against the matching real-CT slice and
     averaged — LPIPS itself is a 2D perceptual metric with no standard 3D
     form, so this is a deliberate, documented adaptation, not an off-the-
     shelf 3D LPIPS.
  5. Projection consistency: re-project the predicted volume through the
     CT's own DRR geometry, compare to the original input DRR (same idea as
     train_recon.py's shape-induction loss, but as an eval metric here with
     real ground truth available to also check against).

Example:
    python -m recon.eval_paired \
        --checkpoint outputs/phase3_recon/latest.pt \
        --ct-dir ../datasets/lidc-idri/dicom_heldout
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from models.tb_model import build_model
from recon.ct_data import build_drr, load_ct_volume_cached, random_pose
from recon.train_recon import drr_to_model_input


def eval_one_series(model, series_dir: Path, image_size: int, volume_size: int,
                    device: torch.device, lpips_fn, realism: str = "off",
                    generator: torch.Generator | None = None) -> dict:
    subject = load_ct_volume_cached(series_dir)
    drr = build_drr(subject, height=image_size, device=device)
    rot, trans = random_pose(device=device, rotation_deg=0.0, translation_mm=0.0)  # canonical AP
    input_drr = drr(rot, trans, parameterization="euler_angles", convention="ZXY")

    # realism != off: film-like input, still scored against the true CT and
    # re-projected against the clean DRR -- the nearest thing to a real film
    # with known ground truth.
    x = drr_to_model_input(input_drr, image_size, realism, generator)
    with torch.no_grad():
        pred_volume = model.forward_recon(x)  # (1, 1, V, V, V)

    target = subject.density.data.to(device).squeeze(0)  # (D, H, W)
    target_shape = tuple(target.shape)
    target_resized = F.interpolate(target[None, None], size=(volume_size,) * 3,
                                   mode="trilinear", align_corners=False)

    pred_np = pred_volume.squeeze().float().cpu().numpy()
    target_np = target_resized.squeeze().float().cpu().numpy()

    data_range = float(max(pred_np.max(), target_np.max()) - min(pred_np.min(), target_np.min()))
    psnr = peak_signal_noise_ratio(target_np, pred_np, data_range=data_range)
    ssim = structural_similarity(target_np, pred_np, data_range=data_range)

    # LPIPS: no standard 3D form — average per-axial-slice 2D LPIPS instead,
    # a deliberate adaptation, not an off-the-shelf metric.
    with torch.no_grad():
        pred_t = pred_volume.squeeze(0).squeeze(0)  # (V, V, V)
        targ_t = target_resized.squeeze(0).squeeze(0)
        pmin, pmax = pred_t.min(), pred_t.max()
        tmin, tmax = targ_t.min(), targ_t.max()
        pred_n = 2 * (pred_t - pmin) / (pmax - pmin + 1e-8) - 1
        targ_n = 2 * (targ_t - tmin) / (tmax - tmin + 1e-8) - 1
        n_slices = pred_n.shape[0]
        lpips_vals = []
        for i in range(0, n_slices, max(1, n_slices // 32)):  # ~32 slices, not all — LPIPS is slow
            p_slice = pred_n[i].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            t_slice = targ_n[i].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            lpips_vals.append(lpips_fn(p_slice, t_slice).item())
        lpips_mean = float(np.mean(lpips_vals))

    # Projection consistency: re-project pred_volume through the CT's OWN
    # geometry (resize to its actual, non-cubic resampled shape first — the
    # documented gotcha, see train_recon.py's shape_induction_step) and
    # compare to the DRR the model was actually shown.
    resized_for_geom = F.interpolate(pred_volume, size=target_shape,
                                     mode="trilinear", align_corners=False)
    drr.density = resized_for_geom.squeeze(0).squeeze(0)
    reprojected = drr(rot, trans, parameterization="euler_angles", convention="ZXY")

    def _norm(t):
        return (t - t.mean()) / (t.std() + 1e-8)
    proj_mse = F.mse_loss(_norm(reprojected), _norm(input_drr)).item()

    return {
        "series": series_dir.name,
        "psnr": float(psnr),
        "ssim": float(ssim),
        "lpips": lpips_mean,
        "projection_mse": proj_mse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ct-dir",     type=str, required=True,
                        help="Directory of held-out CT series subdirs — MUST be disjoint "
                             "from whatever the checkpoint trained on.")
    parser.add_argument("--backbone",   type=str, default="efficientnet_b0")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--volume-size", type=int, default=128)
    parser.add_argument("--drr-realism", type=str, default="off",
                        help="Feed film-like DRRs (recon/drr_realism.py) instead of clean ones.")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()
    gen = torch.Generator().manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone, pretrained=False, with_recon=True,
                        volume_size=args.volume_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: step {ckpt.get('step', '?')}")

    import lpips as lpips_pkg
    lpips_fn = lpips_pkg.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    series_dirs = sorted(p for p in Path(args.ct_dir).iterdir() if p.is_dir())
    if not series_dirs:
        raise ValueError(f"No CT series subdirectories found in {args.ct_dir}")
    print(f"Held-out CT series: {len(series_dirs)}")

    results = []
    for series_dir in series_dirs:
        try:
            r = eval_one_series(model, series_dir, args.image_size, args.volume_size,
                                device, lpips_fn, args.drr_realism, gen)
            results.append(r)
            print(f"  {r['series']}: PSNR={r['psnr']:.2f}  SSIM={r['ssim']:.4f}  "
                  f"LPIPS={r['lpips']:.4f}  proj_mse={r['projection_mse']:.4f}")
        except Exception as e:
            print(f"  {series_dir.name}: FAILED — {type(e).__name__}: {e}")

    if not results:
        print("\nNo series evaluated successfully.")
        return

    psnrs = [r["psnr"] for r in results]
    ssims = [r["ssim"] for r in results]
    lpipss = [r["lpips"] for r in results]
    proj = [r["projection_mse"] for r in results]
    print(f"\n─── Summary (n={len(results)}) ───")
    print(f"  PSNR:  mean={np.mean(psnrs):.2f}  std={np.std(psnrs):.2f}  "
          f"[{np.min(psnrs):.2f}, {np.max(psnrs):.2f}]")
    print(f"  SSIM:  mean={np.mean(ssims):.4f}  std={np.std(ssims):.4f}  "
          f"[{np.min(ssims):.4f}, {np.max(ssims):.4f}]")
    print(f"  LPIPS: mean={np.mean(lpipss):.4f}  std={np.std(lpipss):.4f}  "
          f"[{np.min(lpipss):.4f}, {np.max(lpipss):.4f}]")
    print(f"  Projection MSE: mean={np.mean(proj):.4f}  std={np.std(proj):.4f}  "
          f"[{np.min(proj):.4f}, {np.max(proj):.4f}]")

    if args.json_out:
        summary = {"checkpoint": args.checkpoint, "drr_realism": args.drr_realism, "n": len(results),
                   **{k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                      for k, v in [("psnr", psnrs), ("ssim", ssims), ("lpips", lpipss),
                                   ("projection_mse", proj)]}}
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summary, indent=2))
        print(f"Saved -> {args.json_out}")


if __name__ == "__main__":
    main()
