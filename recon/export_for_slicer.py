"""
The full single pipeline (proposal §12, Phase 4): one chest X-ray in, and out
come the diagnosis, a synthesized 3D volume, and Head A's localization
back-projected into that volume — both written as NRRD for 3D Slicer.

§11a is respected by construction, not by convention: the probability comes
from the Phase 2 diagnostic model alone, and the reconstruction model is a
separate object whose output is never read back into it. The direction of flow
is Head A -> visualization, never the reverse (recon/firewall_test.py checks
the model-level guarantee).

Both written volumes carry the SYNTHESIZED tag at the file level, so the label
travels with the data rather than living only in a UI.

Example:
    python -m recon.export_for_slicer \
        --diagnostic-checkpoint outputs/step3_mild_lungcrop_tbx11k/best_model.pt \
        --recon-checkpoint outputs/phase3_recon_run7_frozen/latest.pt \
        --image ../datasets/tbx11k/val/img_lungcrop/h0004.png --out-dir outputs/slicer_demo
"""
import argparse
import json
from pathlib import Path

import torch

from models.tb_model import build_model
from recon import tracking
from recon.checkpoint import load_recon_model
from recon.export import export_attention, export_heatmap_overlay, export_volume
from recon.heatmap import (DEFAULT_PROJECTION_AXIS, attention_volume, choose_axis,
                           infer_projection_axis)
from recon.train_recon import real_cxr_to_model_input
from utils.gradcam import GradCAM


def diagnose_and_localize(diag_model, x: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Head A's probability and its Grad-CAM, from the diagnostic model only."""
    cam = GradCAM(diag_model, target_layer=diag_model.encoder.net.blocks[-1])
    heatmap = torch.from_numpy(cam(x))
    with torch.no_grad():
        prob = torch.sigmoid(diag_model(x)).item()
    return prob, heatmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diagnostic-checkpoint", type=str, required=True)
    parser.add_argument("--recon-checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True,
                        help="Chest X-ray. Head A's Phase 2 recipe expects a LUNG-CROPPED "
                             "image (lung_crop.py / the *_lungcrop dataset variant); a raw "
                             "full film gives a mis-calibrated probability. Head B was "
                             "trained on body-cropped DRRs, so the same input suits it.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--volume-size", type=int, default=128)
    parser.add_argument("--resample-mm", type=float, default=2.5)
    parser.add_argument("--projection-axis", type=int, default=DEFAULT_PROJECTION_AXIS,
                        help="Volume axis the view projects along, measured on held-out CTs.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Attention threshold for the Slicer label overlay.")
    args = parser.parse_args()

    run = tracking.start_run("export_for_slicer", vars(args),
                             name=tracking.run_name("slicer", args.recon_checkpoint),
                             tags=["phase3", "phase4", "inference"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = real_cxr_to_model_input(Path(args.image), args.image_size).to(device)

    diag_model = build_model("efficientnet_b0", pretrained=False).to(device)
    diag_model.load_state_dict(torch.load(args.diagnostic_checkpoint, map_location=device,
                                          weights_only=False)["model"])
    diag_model.eval()
    prob, heatmap = diagnose_and_localize(diag_model, x)

    recon_model = load_recon_model(args.recon_checkpoint, args.volume_size, device)
    with torch.no_grad():
        volume = recon_model.forward_recon(x).squeeze()

    # The axis comes from the training geometry (measured on held-out CTs); the
    # per-image match is kept as a check, since a real film's intensity convention
    # need not agree with DRR attenuation.
    inferred, corr = infer_projection_axis(volume.cpu(), x[0, 0].cpu())
    axis, warning = choose_axis(args.projection_axis, inferred, corr)
    if warning:
        print(f"[axis] {warning}")
    attention = attention_volume(heatmap, volume.cpu(), axis=axis)

    vol_path = export_volume(volume.cpu(), out_dir / "volume.nrrd", args.resample_mm)
    heat_path = export_attention(attention, out_dir / "attention.nrrd", args.resample_mm)
    mask_path = export_heatmap_overlay(attention, out_dir / "attention_mask.nrrd",
                                       args.resample_mm, args.threshold)
    report = {
        "image": args.image,
        "tb_probability": prob,
        "diagnostic_checkpoint": args.diagnostic_checkpoint,
        "recon_checkpoint": args.recon_checkpoint,
        "projection_axis": axis, "projection_axis_inferred": inferred,
        "projection_correlation": corr, "projection_axis_warning": warning,
        "attention_above_threshold_voxels": int((attention >= args.threshold).sum()),
        "volume": str(vol_path), "attention": str(heat_path), "attention_mask": str(mask_path),
        "threshold": args.threshold,
        "note": ("Probability comes from the diagnostic head alone. The volume and "
                 "overlay are an unvalidated visualization (proposal 11a) and must "
                 "not be measured or used diagnostically."),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print(f"TB probability (Head A):      {prob:.4f}")
    print(f"Projection axis:              {axis}  (per-image check: {inferred}, corr {corr:.3f})")
    print(f"Volume  -> {vol_path}")
    print(f"Overlay -> {heat_path} (continuous) and {mask_path} "
          f"({report['attention_above_threshold_voxels']} voxels >= {args.threshold})")
    print("Both files carry the SYNTHESIZED tag; the volume is not a measurement.")

    tracking.finish(run, {"tb_probability": prob, "projection_axis": axis,
                          "projection_correlation": corr,
                          "attention_voxels": report["attention_above_threshold_voxels"]})


if __name__ == "__main__":
    main()
