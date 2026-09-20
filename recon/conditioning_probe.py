"""
How much does a predicted volume actually depend on *which patient* the input
came from, as opposed to being the same averaged blob every time?

This matters because PSNR/SSIM against a sparse CT target reward getting the
bulk density right, which a constant, input-invariant prediction can do well.
A model that scores highly there may have learned the population average
rather than per-patient structure — a failure already observed on this
project's earlier checkpoints (0% near-zero voxels, r=0.68-0.79 correlation
between different patients' predictions).

Two numbers, both computed from the same set of predictions:
  * variance ratio — variance across patients over variance across poses of
    the same patient. 1.0 means the patient identity changes the prediction no
    more than jittering the camera does; higher is better conditioning.
  * cross-patient correlation — mean correlation between different patients'
    predicted volumes. Near 1.0 means every patient gets the same volume.

Example:
    python -m recon.conditioning_probe \
        --checkpoint outputs/phase3_recon_run2_fullscale/latest.pt \
        --ct-dir ../datasets/lidc-idri/dicom_heldout --image-size 320 --volume-size 128
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from recon.checkpoint import load_recon_model
from recon.ct_data import build_drr, load_ct_volume_cached, random_pose
from recon.train_recon import drr_to_model_input


@torch.no_grad()
def predict_series(model, series: Path, image_size: int, n_poses: int,
                   device: torch.device) -> np.ndarray:
    """(n_poses, n_voxels) predictions for one patient at different camera poses."""
    subject = load_ct_volume_cached(series)
    drr = build_drr(subject, height=image_size, device=device)
    out = []
    for _ in range(n_poses):
        rot, trans = random_pose(device=device)
        img = drr(rot, trans, parameterization="euler_angles", convention="ZXY")
        vol = model.recon_head(model.encoder(drr_to_model_input(img, image_size)))
        out.append(vol.flatten().float().cpu().numpy())
    del drr
    torch.cuda.empty_cache()
    return np.stack(out)


def cross_patient_correlation(patient_means: np.ndarray) -> float:
    c = np.corrcoef(patient_means)
    return float(c[np.triu_indices_from(c, k=1)].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ct-dir", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--volume-size", type=int, default=128)
    parser.add_argument("--poses-per-ct", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_recon_model(args.checkpoint, args.volume_size, device)

    per_patient = [predict_series(model, s, args.image_size, args.poses_per_ct, device)
                   for s in sorted(p for p in Path(args.ct_dir).iterdir() if p.is_dir())]

    stacked = np.stack(per_patient)                      # (patients, poses, voxels)
    patient_means = stacked.mean(axis=1)
    within = float(stacked.var(axis=1).mean())           # pose jitter, same patient
    between = float(patient_means.var(axis=0).mean())    # patient identity
    result = {
        "checkpoint": args.checkpoint,
        "n_patients": len(per_patient), "poses_per_ct": args.poses_per_ct,
        "variance_between_patients": between, "variance_within_patient": within,
        "variance_ratio": between / (within + 1e-12),
        "cross_patient_correlation": cross_patient_correlation(patient_means),
    }

    print(f"Patients: {result['n_patients']} x {args.poses_per_ct} poses")
    print(f"Variance ratio (between-patient / within-patient): {result['variance_ratio']:.2f}"
          f"   (1.0 = identity matters no more than camera jitter)")
    print(f"Cross-patient correlation: {result['cross_patient_correlation']:.3f}"
          f"   (1.0 = every patient gets the same volume)")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=2))
        print(f"Saved -> {args.json_out}")


if __name__ == "__main__":
    main()
