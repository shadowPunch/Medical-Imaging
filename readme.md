# TB Chest X-ray Diagnostic + Synthesized 3D Visualization

A tuberculosis screening model for chest X-rays targeting the WHO Target
Product Profile triage spec, plus a second head that synthesizes a 3D thoracic
volume from the same single radiograph and delivers it into 3D Slicer.

Two heads share one encoder:

- **Head A — diagnosis.** EfficientNet-B0 + GAP/FC. This is the product.
- **Head B — reconstruction.** Back-projection decoder producing a 128³ volume.
  This is an **unvalidated visualization aid**, never a diagnostic signal.

The separation is enforced in code, not by convention: `TBDiagnosticModel.forward()`
never references `recon_head`, checked by `recon/firewall_test.py`.

---

## Results

**Head A — diagnosis (held-out TBX11K):**

| Metric | Value |
|---|---|
| Held-out AUC | 0.889 |
| Sensitivity @ spec ≥70%, per-site recalibrated threshold | **92.6%** [90.3, 94.6] |
| Sensitivity / specificity at a validation-frozen threshold | 98.2% / 44.0% ✗ |

Meets the WHO TPP triage target (≥90% sensitivity, ≥70% specificity) **only
under per-site threshold recalibration**, which needs ≈250–300 site-specific
labelled negatives. A globally frozen threshold does not transfer. Training
negative-class composition must also match the deployment population — getting
it wrong cost ~28 points of ceiling sensitivity in one of three tested
directions.

**Head B — reconstruction (12 held-out LIDC-IDRI CT series):**

| Metric | Value |
|---|---|
| PSNR / SSIM | 22.81 / 0.495 |
| LPIPS | 0.559 |
| Projection consistency (MSE) | 0.064 |
| Cross-patient correlation | 0.938 |
| **Head A AUC drop while training Head B** | **0.0000** |

Not a validated reconstruction: real X-rays and synthetic DRRs remain trivially
separable in feature space (probe AUC 1.000), and predicted volumes are similar
across patients. It is a coarse shape prior, and every exported file says so in
its metadata.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Datasets are fetched separately — see `../datasets/README.md` (Shenzhen,
Montgomery and TBX11K via Kaggle; LIDC-IDRI CT via TCIA's REST API). Images and
checkpoints are gitignored and never committed.

Experiment tracking is required: runs log to Weights & Biases (project
`tb-phase3`) and **fail rather than run untracked**. Set `TB_WANDB=0` to
disable deliberately.

---

## Usage

**Train the diagnostic head** (`--held-out` also takes `tbx11k`, `tbx11k-val`, `none`):

```bash
python train_diagnostic.py \
    --shenzhen ../datasets/tb-shenzen --montgomery ../datasets/tb-montgomery \
    --tbx11k ../datasets/tbx11k --held-out montgomery \
    --lung-crop --texture-aug --aug-strength mild \
    --output outputs/diagnostic --epochs 50 --batch-size 16
```

`--lung-crop --texture-aug --aug-strength mild` is the recipe, not an optional
extra: without it a large share of accuracy rides on an acquisition-texture
confound rather than thoracic signal.

**Train the reconstruction head** (encoder frozen — see Key findings #3):

```bash
python -m recon.train_recon \
    --lidc-dir ../datasets/lidc-idri/dicom \
    --init-from outputs/diagnostic/best_model.pt --freeze-encoder \
    --drr-realism mild --drr-realism-prob 0.5 \
    --image-size 320 --volume-size 128 --steps 8000 \
    --checkpoint-every 200 --output outputs/recon/latest.pt
```

**Run the full pipeline** — X-ray in, probability + volume + localization out:

```bash
python -m recon.export_for_slicer \
    --diagnostic-checkpoint outputs/diagnostic/best_model.pt \
    --recon-checkpoint outputs/recon/latest.pt \
    --image <lung-cropped CXR>.png --out-dir outputs/case01
```

Then load `outputs/case01` in 3D Slicer via `slicer_module/` (see its README).

**Verify a change hasn't broken anything:**

```bash
python -m pytest tests/ -q          # 65 tests
python -m recon.firewall_test       # Head A/B isolation

# Head A must not move when Head B trains — exits non-zero if it does
python -m eval.encoder_drift_probe \
    --checkpoint outputs/diagnostic/best_model.pt \
    --recon-checkpoint outputs/recon/latest.pt \
    --shenzhen ../datasets/tb-shenzen --montgomery ../datasets/tb-montgomery \
    --tbx11k ../datasets/tbx11k --held-out tbx11k --lung-crop --max-auc-drop 0.005
```

---

## Repository layout

```
train_diagnostic.py     Head A training CLI
lung_crop.py            Preprocessing: lung-field crop (part of the recipe)
config.py               Hyperparameter dataclasses
data/                   Per-source loaders + augmentation pipelines
models/                 SharedEncoder, DiagnosticHead, ReconHead, TBDiagnosticModel
training/               Trainer, losses
eval/                   Metrics, frozen-threshold + bootstrap-CI evaluation,
                        confound probes, encoder_drift_probe.py (Head A gate)
recon/                  Head B: CT/DRR data, training, evaluation, probes,
                        heatmap back-projection, Slicer export
slicer_module/          3D Slicer scripted module
tests/                  pytest suite
docs/                   Full investigation log
figures/                Diagnostic plots referenced from the log
```

---

## Key findings

Short version; the evidence for each is in
[`docs/investigation-log.md`](docs/investigation-log.md).

1. **A large share of apparent accuracy was an acquisition-texture confound.**
   Found by ablation (inverse-mask, region decomposition, fixed-location patch,
   high-pass residual, label-shuffle control) and fixed by lung-cropping plus
   mild texture augmentation.

2. **Threshold transfer fails across sites.** Discrimination is fine;
   calibration is what breaks. Traced to a concrete training-data gap —
   TBX11K's negatives include sick-but-non-TB cases that Shenzhen and
   Montgomery's near-exclusively-healthy negatives never taught the model to
   place.

3. **Reconstruction training was silently destroying Head A.** Jointly training
   the shared encoder dropped held-out diagnostic AUC from 0.889 to 0.65 — and
   to 0.33, below chance, in one run. Refitting a fresh head recovered only to
   0.77, so the TB signal was genuinely destroyed, not merely misaligned. The
   firewall guaranteed Head B never *feeds* Head A at inference; it said nothing
   about the encoder drifting underneath Head A during training. Fixed by
   freezing the encoder, and now enforced by a gate.

4. **A headline reconstruction result was an averaged prediction.** An earlier
   checkpoint scoring PSNR 28.33 / SSIM 0.733 produced near-identical volumes
   for every patient (cross-patient r = 0.963) — the metric was rewarding an
   averaged chest. Measured by `recon/conditioning_probe.py`.

5. **Two plausible ideas did not survive full scale.** DRR realism augmentation
   looked good at 1000 local steps and inverted at 4000 and at full scale; the
   unpaired shape-induction term was actively hurting reconstruction geometry
   and was removed. Both are recorded as negative results rather than quietly
   dropped.

---

## Limitations

- Retrospective research splits only. No prospective or prevalence-adjusted
  evaluation, and no clinical validation.
- Head A needs per-site threshold recalibration (~250–300 labelled negatives)
  and negative-class composition matched to the deployment population.
- Head B is an unvalidated visualization: no comparison against X2CT or
  DuoLift was run, real and synthetic inputs remain trivially separable, and
  predictions vary little between patients.
- The Slicer module's logic is unit-tested, but it has never been executed
  inside 3D Slicer — an accepted gap, with the likely failure points named in
  the log.

---

## Documentation

- [`docs/investigation-log.md`](docs/investigation-log.md) — the full record:
  validation methodology, every ablation, the deployment package, resource
  costs, and the evidence behind the reversals above.
- [`slicer_module/README.md`](slicer_module/README.md) — Slicer install and use.
- `../TB_CXR_Diagnostic_3D_Proposal.md` — motivation, architecture, phased plan.
