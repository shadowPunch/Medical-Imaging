# TB CXR Diagnostic — Phase 1 & 2

Shared encoder + diagnostic head for TB detection from chest X-rays.
Targets the WHO TPP triage specification (sensitivity ≥ 90%, specificity ≥ 70%).

---

## Status (2026-08-25)

**Phase 1 — diagnostic head: done.** EfficientNet-B0 + GAP/FC, trained on
Shenzhen + Montgomery + TBX11K.

**Phase 2 — cross-source generalization: done, and it uncovered a real
confound before it found a real fix.**

- The original held-out evaluation had a threshold-leakage bug (operating
  point chosen on the same data it was scored on) — fixed; frozen-threshold
  scoring + bootstrap CIs are now standard for every held-out result
  (`eval/metrics.py`).
- Characterized, not just detected, a genuine acquisition-texture confound:
  a model trained with the entire lung field blanked out still hit 0.93 AUC
  in-domain (`lung_crop.py --mode complement`), survived a mask-dilation
  check, replicated in an independent source (Montgomery), and was
  confirmed via a fixed-location corner patch, a high-pass residual, and a
  region-by-region decomposition (`eval/histogram_probe.py`,
  `region_decompose.py`, `patch_extract.py`, `highpass_complement.py`) — all
  converging on the same acquisition-texture explanation. A label-shuffle
  control (`train_diagnostic.py --shuffle-labels`) confirmed the pipeline
  itself isn't leaking.
- Found a fix that actually works: **lung-crop + mild texture augmentation**
  gets TBX11K held-out sensitivity to **92.6% [90.3, 94.6] at 70%
  specificity** under a per-site threshold (`eval/ceiling_analysis.py`) —
  meets the WHO TPP triage sensitivity/specificity targets on held-out TBX11K
  (retrospective research split; no prospective or prevalence-adjusted
  evaluation), with a tight, non-overlapping CI against every earlier
  configuration. **This is a ceiling, not an out-of-the-box deployable
  number** — a threshold frozen from validation instead of TBX11K itself
  (`eval/frozen_threshold_check.py`) gives 98.2% sensitivity but only 44.0%
  specificity on TBX11K, well below the 70% floor; per-site threshold
  recalibration is required to actually realize 92.6%, not optional. Traced
  the mechanism to a training-data gap, not just acquisition texture:
  splitting TBX11K's held-out negatives by their own raw annotation
  (`eval/negative_composition_probe.py`) found that `sick_but_non-tb` cases
  score a median 0.722 (vs. `healthy`'s 0.055) — the model never learned to
  place non-TB pathology because Shenzhen/Montgomery's training negatives
  are almost entirely healthy. Turned the recalibration requirement into a
  concrete number via a calibration-set-size sweep: **≈250–300 labeled
  negatives from the target site to hit the 70% specificity target within
  ±5 points at ≥90% confidence**; below ~100, calibration is closer to
  noise than correction. **Then tested the direct fix**, not just the
  calibration workaround: retraining with TBX11K's own train split
  (including sick_but_non-tb) added to the negative class took
  frozen-threshold specificity from 43.8%→97.6% on identical held-out
  images and collapsed the healthy/sick_but_non-tb bimodality — confirming
  the mechanism is real and fixable at the source, though this specific
  test isn't a fair cross-source comparison (TBX11K informed training) so
  it isn't a new headline number. It also disproved this document's own
  earlier explanation for the calibration-size curve's shape (a bimodal
  "sparse valley" effect) — the curve barely moved even with the
  bimodality gone, pointing instead to a generic, distribution-free
  order-statistic effect. An augmentation-strength sweep
  (mild/medium/aggressive) confirmed the mechanism: too-aggressive texture
  randomization destroys real diagnostic texture (cavitation, nodules) along
  with the confound, and mild wins because it removes less signal overall
  while removing more of the *wrong* signal. A CXR-pretrained DenseNet
  encoder swap tied this result without improving on it; AdaBN test-time
  adaptation added a small free AUC gain with no measurable ceiling movement.
- Full methodology, numbers, and the "what didn't work and why" trail are in
  **Phase 2 validation results** and **Robustness interventions** below —
  worth reading in full before citing just the headline number, since
  several earlier configurations looked fine until checked harder.

**Phase 3 — reconstruction head: pipeline built and verified end-to-end on
real data; not trained to convergence.** `code/recon/`:

- `ct_data.py` — loads a real LIDC-IDRI DICOM series directly (DiffDRR reads
  the directory with no manual conversion) and generates DRRs at random
  poses. Verified: full-resolution volumes OOM the local 4GB card (>3.7GB);
  resampling to ~2.5mm spacing (~128 voxels/axis) keeps DRR generation under
  ~2GB, matching the proposal's §9 128³ target.
- `models/recon_head.py` — `ReconHead`, a right-sized back-projection
  network (2D feature -> channel-to-depth lift -> 3D conv decoder), 0.35M
  params, DuoLift-CNN-*inspired* per §6, not a vendored copy of the
  released code/weights. Wired into `TBDiagnosticModel` via
  `build_model(..., with_recon=True)`.
- `recon/firewall_test.py` — automated check of the §11a isolation
  guarantee (value identity + gradient isolation, both directions). Passes.
  This is enforced structurally (`forward()` never references
  `self.recon_head`), not just documented.
- `recon/train_recon.py` — paired DRR-supervised loss (real CT -> DRR ->
  predicted volume -> compare to the CT's own density) plus unpaired shape
  induction on real CXRs (Sizikova et al. — predicted volume re-projected
  through DiffDRR, compared against the original input image; no CT ground
  truth needed for this term). **Two real bugs found and fixed while
  verifying this on real data, not synthetic placeholders:** (1) an
  unconstrained final layer produced exactly-zero gradients through the
  re-projection loss — CT density is physically non-negative and the
  renderer degenerates on near-zero/negative input; fixed with `softplus`.
  (2) a silent shape mismatch between `ReconHead`'s fixed volume_size cube
  and each CT's actual (non-cubic) resampled geometry was scrambling the
  renderer's ray indexing without erroring — caught via an identical loss
  value across different random initializations, which shouldn't happen;
  fixed by resizing the prediction to the canonical geometry before
  substitution. After both fixes: real, non-zero gradients confirmed on both
  loss terms, and shape-induction loss measurably decreases over a short
  smoke run (2.22 → 0.78 over 8 steps).
- `recon/export.py` — NRRD/NIfTI export via SimpleITK, each file tagged
  "SYNTHESIZED" at the metadata level (belt-and-braces alongside the §11a
  code-level guarantee, not a replacement for it).

**Training run (completed).** Driven headlessly via the Kaggle CLI
(`kaggle kernels push`), not the browser — `code/tb_phase3_kaggle.ipynb`,
GPU T4, ~4 hours (06:12→10:20 UTC). Verified from the full run log, not
assumed from a "COMPLETE" status alone:
- 150/150 LIDC-IDRI CT series fetched, 0 failures.
- 10,696 unpaired real CXRs available for the shape-induction term (this
  confirms the `/kaggle/input/datasets/<owner>/<slug>/` mount-path fix
  actually worked — the prior run silently found 0 and skipped the term
  entirely).
- 2000/2000 training steps completed, 0 errors/tracebacks in the entire log.
- Paired DRR-supervision loss: 0.3782 → 0.1511 → 0.0789 (clean, substantial,
  monotonic-ish decrease).
- Shape-induction loss: 1.4689 → 0.6563 → 1.1237 (noisier, as expected for
  an unpaired re-projection term, but consistently well below the ~2.0
  uncorrelated-baseline ceiling).
- Final export ran on a real held-out CXR (`s1559.png`), not a synthetic
  fallback — confirms a genuine end-to-end real-data path, not just a
  shape-check.

This is a single run, not a validated model: no PSNR/SSIM/LPIPS against
paired CT ground truth yet (§10's quantitative validation), no held-out
generalization check, no hyperparameter search. It demonstrates the
pipeline trains stably end-to-end on real data at real scale — it does not
demonstrate reconstruction quality. Output artifacts pulled down to
`outputs/phase3_recon/` (gitignored, like every other checkpoint dir) and
inspected directly, not just trusted from the log: `latest.pt` is a real
386-key model state dict at step 2000/2000; `sample_reconstruction.nrrd` is
a 128×128×128 float32 volume at 2.5mm spacing (matches training config),
values in [0.096, 4.09] — non-degenerate, not all-zero or NaN — with the
`SYNTHESIZED` safety tag intact. An unfiltered `kernels output` pull is
impractical here (it lists the redundant 11 GB DICOM input tree first,
alphabetically ahead of the actual outputs, and rate-limits (HTTP 429)
paginating at the CLI's default page size of 20) — fixed by filtering to
`phase3_recon/.*` and raising `--page-size` to its max (200), which cut the
number of listing calls enough to avoid the rate limit; download completed
in under a minute once that was in place.

Earlier runs on the way to this one (kept here since they surfaced real
bugs, not because they're results to cite): a first pass crashed on
`/kaggle/working` disk quota at `N_SERIES=400` (~29 GB projected against a
~20 GB quota) — the real `OSError: [Errno 28] No space left on device` was
masked by a confusing downstream nbconvert HTML-export failure until the
full log was read; fixed by trimming to `N_SERIES=150` and adding an
explicit disk-space guard. A second pass completed all steps but ran the
shape-induction term over 0 CXRs due to the mount-path assumption above
being wrong; fixed and is what produced the numbers reported here.

**CT data.** `datasets/lidc-idri/dicom/` holds 150 real CT series (11 GB, 0
failures) — fetched directly via TCIA's public REST API
(`datasets/lidc-idri/fetch_dicom.py`), no desktop NBIA Data Retriever
needed after all. See `datasets/lidc-idri/README.md`.

**Git status.** `datasets/` is fully committed (including the LIDC-IDRI
fetch work). `code/` has substantial uncommitted changes — the Stage 1
follow-up work and everything in this Phase 3 section — ask before
committing.

---

## Architecture

```
CXR image (320×320 by default — see GPU memory budget below)
       │
       ▼
SharedEncoder  — EfficientNet-B0 backbone (timm, features_only=True)
               — returns 5 feature maps at scales [1/2, 1/4, 1/8, 1/16, 1/32]
               — channels: [16, 24, 40, 112, 320]
       │
       ▼ features[-1]  (320 ch, 10×10 for 320 input)
       │
DiagnosticHead — GAP → Dropout → Linear(320, 1)
               — outputs a raw logit; sigmoid at inference → TB probability
```

The encoder is **shared** by design: Phase 3 (reconstruction head) will plug into the
same backbone without retraining from scratch.

Grad-CAM localization runs on the final encoder feature map (`encoder.net.blocks[-1]`)
to produce a TB attention heatmap without any additional localization supervision.

---

## Why these choices

| Decision | Reason |
|---|---|
| EfficientNet-B0 | Small enough to INT8-quantize and run on CPU (<50 ms); strong ImageNet init |
| `features_only=True` | Exposes all 5 scale levels cleanly — needed later for Head B back-projection |
| WeightedRandomSampler + pos_weight | Two-level imbalance correction; sampler at batch level, pos_weight in loss |
| Label smoothing | Clinical thresholds require calibrated probabilities, not just ranking |
| Cosine LR schedule | Avoids sharp LR transitions; works well with AdamW for fine-tuning pretrained models |
| Held-out source eval | The main risk in TB classifiers is scanner/site overfitting; single-split AUC hides this |

---

## File structure

```
code/
├── train_diagnostic.py   — CLI entry point (Phase 1 + 2)
├── preprocess.py         — One-time resize of Shenzhen/Montgomery to a fixed size
├── lung_crop.py           — One-time lung-field crop / complement (--mode crop|complement)
├── region_decompose.py    — One-time region decomposition (lungs/shoulders/spine/diaphragm/other)
├── patch_extract.py       — One-time fixed-location corner-patch extraction
├── highpass_complement.py — One-time high-pass residual (Gaussian-blur subtraction)
├── lung_crop_clahe.py     — One-time lung-crop + in-mask CLAHE/z-score normalization
├── sanity_check.py       — Probes max (image_size, batch_size) that fits in VRAM
├── config.py             — Hyperparameter dataclasses
├── data/
│   ├── dataset.py        — Per-source loaders + TBCXRDataset
│   └── transforms.py     — Train / val augmentation pipelines
├── models/
│   ├── encoder.py        — SharedEncoder (backbone)
│   ├── diagnostic_head.py— Head A: GAP + FC
│   └── tb_model.py       — TBDiagnosticModel + build_model()
├── training/
│   ├── loss.py           — LabelSmoothBCE + pos_weight helper
│   └── trainer.py        — Trainer: fit / evaluate / checkpoint
├── eval/
│   ├── metrics.py            — AUC, WHO TPP operating point, frozen-threshold eval, bootstrap CI, sens@spec ceiling
│   ├── ceiling_analysis.py   — Sens @ spec≥70% from an existing checkpoint — no retraining
│   ├── frozen_threshold_check.py     — Val-frozen threshold scored on held-out — no retraining
│   ├── score_distribution_diagnosis.py — Val-vs-held-out score histograms + calibration-set-size sweep — no retraining
│   ├── negative_composition_probe.py — Splits held-out negatives by raw annotation subgroup (healthy vs. sick_but_non-tb) — no retraining
│   ├── projection_probe.py   — Does in-domain AUC ride on a linear source direction?
│   ├── histogram_probe.py    — Spatially-blind intensity-histogram probe
│   ├── shortcut_baseline.py  — Source-classifier shortcut-detector baseline
│   └── complement_monitor.py — Standing in-domain confound diagnostic for any intervention
├── figures/               — Diagnostic plots referenced from this readme (tracked; regenerate via the eval/ scripts above, not hand-edited)
└── utils/
    ├── gradcam.py        — Grad-CAM + heatmap overlay
    └── lung_mask.py      — Pretrained lung-field segmentation (torchxrayvision)
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Shenzhen and Montgomery ship as full-resolution PNGs (up to ~4900×4000px), which
makes the DataLoader the bottleneck (~300ms/image). Resize them once:

```bash
python preprocess.py --shenzhen /data/shenzhen --montgomery /data/montgomery --size 512
```

This caches 512px grayscale copies to `<root>/images_512/`; `dataset.py` picks
these up automatically. TBX11K ships pre-sized at 512px already.

---

## GPU memory budget

Measured on a 4GB RTX 3050 Ti Mobile, training mode, AMP on (`torch.amp.autocast`
+ `GradScaler`, as used by `Trainer`):

| image_size | batch_size | peak VRAM |
|---|---|---|
| 512 | 16 | OOM |
| 512 | 8  | 1.92 GB |
| 384 | 16 | 2.15 GB |
| 320 | 16 | 1.52 GB |
| 320 | 32 | 2.98 GB |
| 256 | 16 | 0.98 GB |

`config.py` defaults to **320px / batch 16** (1.52 GB peak), leaving headroom for
the OS and other processes. Run `python sanity_check.py` to re-probe this on a
different GPU — it sweeps the same grid and reports what fits.

---

## Running

```bash
python train_diagnostic.py \
    --shenzhen   /data/shenzhen \
    --montgomery /data/montgomery \
    --tbx11k     /data/tbx11k \
    --held-out   montgomery \       # withheld from training; used for cross-source eval
    --output     outputs/diagnostic \
    --epochs     50 \
    --batch-size 16
```

`--held-out` also accepts `tbx11k` (train on Shenzhen+Montgomery, test on
TBX11K — the other cross-source direction), `tbx11k-val` (TBX11K's train
split joins training, only its val split is held out — used for the
negative-class-composition retraining experiment; not a cross-source test,
see Robustness interventions), or `none` (train on all sources, no
held-out evaluation).

---

## Expected outputs

```
outputs/diagnostic/
└── best_model.pt    — checkpoint at best val AUC (model weights + epoch + metrics)
```

Console reports AUC and WHO TPP operating point after each epoch, and a final
held-out source evaluation showing the cross-source generalisation gap.

---

## Phase 2 validation methodology

A checkpoint is not a validated result. The held-out evaluation is specifically
built to avoid the two ways this kind of number goes wrong:

- **The WHO-TPP operating point is frozen, never re-derived on the held-out
  set.** After training, the threshold is chosen from validation data only
  (`find_who_tpp_point` on `val_loader`'s probabilities); the held-out source
  is then scored against that fixed threshold via `apply_threshold`
  ([eval/metrics.py](eval/metrics.py)) — picking the threshold on the same
  data it's scored on leaks the test set and invalidates the reported
  sensitivity/specificity.
- **Every held-out metric is reported with a 95% bootstrap CI**
  (`bootstrap_ci`, 2000 resamples), not a bare point estimate. Montgomery is
  ~138 images — a point estimate there hides how little the sample size can
  actually resolve.

Two supporting checks, run alongside the held-out eval, before trusting any of
these numbers as "the diagnosis works":

- **Shortcut-detector baseline** (`eval/shortcut_baseline.py`) — trains a
  classifier to predict *which dataset* an image came from (not TB status).
  Near-perfect accuracy means scanner artifacts/borders/markers are a
  trivially available shortcut, and single-split AUC on any one source can't
  be trusted — cross-source AUC is what matters.
  ```bash
  python -m eval.shortcut_baseline \
      --shenzhen /data/tb-shenzen --montgomery /data/tb-montgomery --tbx11k /data/tbx11k
  ```
- **Lung-crop ablation** — crop every image to its lung-field bounding box
  (via a pretrained segmentation model, no extra training data needed) before
  training, removing corner markers/borders/burned-in text as a possible
  shortcut surface. If cross-source AUC holds and shortcut-baseline accuracy
  drops, the diagnostic signal is probably thoracic; if AUC collapses, it
  wasn't.
  ```bash
  python lung_crop.py --shenzhen /data/tb-shenzen --montgomery /data/tb-montgomery --tbx11k /data/tbx11k
  python train_diagnostic.py ... --lung-crop
  python -m eval.shortcut_baseline ... --lung-crop
  ```

---

## Phase 2 validation results (2026-08-25)

Run with `--epochs 50 --batch-size 16` (early-stopped at patience 10), the
default EfficientNet-B0 backbone, 320px. **Neither held-out direction clears
the WHO TPP bar.**

| Direction | Val AUC | Held-out AUC (95% CI) | Frozen-threshold sens / spec | Ceiling: sens @ spec≥70% (95% CI) |
|---|---|---|---|---|
| Train Shenzhen+TBX11K → held-out **Montgomery** | 0.997 | **0.580** [0.477, 0.682] | 37.9% / 82.5% | **44.8%** [31.5%, 61.5%] |
| Train Shenzhen+Montgomery → held-out **TBX11K** | 0.974 | **0.810** [0.793, 0.827] | 95.3% / 30.0% | **78.5%** [75.4%, 81.6%] |
| Montgomery, **+lung-crop** | 0.994 | **0.832** [0.750, 0.905] | 36.2% / 100.0% | **79.3%** [66.7%, 90.0%] |

An AUC of 0.58 (CI touching below chance) means the model has essentially no
useful signal on Montgomery when trained without it — not "a bit worse than
val," a near-total generalization failure. The TBX11K direction generalizes
far better on ranking (AUC 0.81, tight CI) but the threshold picked for 90%
sensitivity on validation data over-triages TBX11K badly (30% specificity vs.
the 70% floor).

**The frozen-threshold sens/spec numbers above are not a discrimination
measurement** — they show where one specific threshold happened to land, not
what the encoder can do. The "Ceiling" column (`eval/ceiling_analysis.py`,
no retraining — reads sensitivity off the ROC curve at spec≥70% on each
checkpoint's existing held-out predictions) answers the real question: even
under a perfect per-site threshold, could this encoder discriminate TB on
this held-out source at all? **All three ceilings sit clearly below the 90%
sensitivity target** (44.8%, 78.5%, 79.3%). That settles it: **discrimination
is binding, not calibration** — a better per-deployment threshold alone would
not fix this, on any of the three runs measured so far.

**Shortcut-classifier baseline**
(`eval.shortcut_baseline`, 10 epochs, balanced across sources):

| Variant | Source-classification accuracy | Chance |
|---|---|---|
| No lung-crop | **95.2%** | 33.3% |
| Lung-crop | **95.2%** | 33.3% |

**Lung-crop ablation reads as a split result, not a clean fix.** It
substantially improved Montgomery's held-out AUC (0.58 → 0.83, CI no longer
touching chance) — real evidence that at least part of the original
cross-source gap came from something the crop removes (borders, corner
markers/text — see the "EREC[T]" marker example this crop catches). But it
did **not** move the shortcut-classifier accuracy at all (95.2% → 95.2%,
identical), and did not fix the frozen-threshold sensitivity collapse
(37.9% → 36.2%, still far below the 90% target) — consistent with the ceiling
analysis above: the crop bought back ranking quality, not calibration.

**Inverse-mask ablation** (`lung_crop.py --mode complement`, then
`train_diagnostic.py --lung-complement`) — the sharper, in-domain version of
the same question: within **Shenzhen alone** (no cross-source comparison at
all), does blanking out the entire lung field and training on only the
surrounding anatomy (shoulders, diaphragm edges) and border/marker text still
predict the TB label?

| Shenzhen-only, in-domain | Val AUC |
|---|---|
| Full frame (baseline) | 0.962 |
| **Lung field blanked out (complement only)** | **0.933** |

A model that never sees a single lung-field pixel recovers almost all of the
full-frame AUC. This is a **within-source** confound — it needs no
cross-dataset comparison to demonstrate, which means it also inflates every
single-split, single-source AUC reported anywhere in this pipeline (including
each run's "Val AUC" column above). The label is substantially predictable
from acquisition-correlated cues (patient positioning, body habitus, marker
placement) independent of any real thoracic finding.

**Mask-dilation check — does this survive under-coverage of the apices and
costophrenic angles?** Chest segmentation models systematically under-cover
the lung apices (the single most TB-predominant region) and the costophrenic
angles (where pleural effusion, a TB finding, shows up) — a tight mask could
leave real lung signal just inside the "blanked" box, making 0.933 an
artifact of reading leaked lung tissue rather than a genuine confound.
Checked directly: `lung_crop.py --mode complement --dilate-px 25` (a visibly
much larger blanked region, `utils/lung_mask.py`'s `extra_px` parameter) then
retrained the same way.

| Shenzhen-only, in-domain | Val AUC |
|---|---|
| Complement, margin only (5%) | 0.933 |
| **Complement, +25px dilated** | **0.951** |

The result **survives dilation — if anything it rose slightly.** This isn't
mask under-coverage; the signal genuinely lives outside the lung field.

**Global-acquisition-statistics probes** — two cheap, spatially-limited
tests to check whether this is actually about anatomy (shoulders, diaphragm
edges) or just global exposure/processing statistics, before spending
compute on decomposing it further:

| Probe | Val AUC |
|---|---|
| Intensity histogram of non-lung region only, no spatial structure (`eval.histogram_probe`) | 0.720 |
| 32×32 thumbnail of the complement (`train_diagnostic.py --lung-complement --image-size 32`) | 0.802 |

Both land clearly below the full-resolution complement's 0.933–0.951 and
below the ~0.85 threshold that would make region decomposition moot — global
intensity statistics and coarse layout alone don't explain the effect. Fine
spatial detail is doing real work, so decomposing by region is informative,
not redundant.

**Region decomposition** (`region_decompose.py`, using PSPNet's actual organ
masks rather than hand-drawn boxes — clavicle+scapula for shoulders,
spine+mediastinum+aorta+weasand for spine, facies diaphragmatica for
diaphragm, everything uncovered by any organ mask nor the lungs for
"other") — training one Shenzhen-only model per isolated region turns
"something outside the lungs predicts TB" into a specific mechanism:

| Region visible (everything else blanked) | Val AUC |
|---|---|
| Full frame (baseline) | 0.962 |
| Lungs only | 0.963 |
| Corners / border / marker text / other | **0.938** |
| Shoulders / clavicles | **0.935** |
| Spine / mediastinum | 0.889 |
| Diaphragm | 0.866 |

"Lungs only" essentially matching full-frame (0.963 vs. 0.962) is the sanity
check working as expected — the lung field does carry the primary signal, as
it should. But **every non-lung region individually clears or approaches
the 0.85 threshold** the proposal's own decision rule uses to call a region
an unambiguous confound (corners: unambiguous acquisition artifact by that
rule; shoulders: consistent with body habitus, a real biological TB
correlate rather than a pure artifact, but still a shortcut for a triage
tool; spine and diaphragm: weaker but still well above chance). The signal
isn't concentrated in one attributable region — it's distributed across
essentially the whole frame outside the lungs, which argues against a single
clean "it's just the corner markers" story and toward some combination of a
pervasive acquisition/processing signature and genuine habitus correlation.
**Caution on framing:** confounding in Shenzhen/Montgomery from partly
different positive/negative collection contexts has been noted in prior TB
CXR literature — this should be positioned as a known confound quantified
precisely (first measurement of how much in-domain AUC survives complete
lung removal, plus the region breakdown), not as a novel discovery. See
**Literature check** below — done, not pending.

**Label-shuffle control** — permute training-set labels only
(`train_diagnostic.py --shuffle-labels`, val/held-out stay real), rerun the
base Shenzhen complement setup. A pipeline that isn't leaking should collapse
to ~0.5. It does, but the naive "best-checkpoint" AUC (0.751) is *not* the
right number to report here: early-stopping's "save whichever epoch has the
highest val AUC" is exactly the wrong summary statistic on pure noise —
across 23 epochs, taking the max is a multiple-comparisons draw that inflates
away from 0.5 by construction. The actual per-epoch trajectory is:

```
0.53 0.24 0.49 0.47 0.44 0.48 0.67 0.72 0.58 0.65 0.67 0.52 0.75
0.60 0.59 0.59 0.58 0.47 0.50 0.55 0.54 0.58 0.59
```

No stable convergence, oscillating 0.24–0.75, mean ≈0.56 (last-10-epoch mean
≈0.56) — noise on a 99-image validation set, not the clean, stable climb to
0.93+ every real result in this section shows. **The control passes: no
leakage.** (This is also a caution about every "Best val AUC" reported
elsewhere in this file — legitimate there because those curves converge and
plateau rather than oscillate, but "best-of-N-epochs" is a statistic worth
distrusting whenever a run doesn't show that shape.)

**Single-patch test** — does a small, *fixed-location* 96×96 patch (same
pixel coordinates for every image, no per-image segmentation) predict TB
status? Tests whether region decomposition was measuring the same pervasive
signature five times rather than five distinct mechanisms (`patch_extract.py`
+ `train_diagnostic.py --variant patch_<loc>_96 --image-size 96`).

| Fixed 96×96 corner patch, Shenzhen-only | Val AUC |
|---|---|
| Top-left | **0.913** |
| Top-right (catches the "L pa" marker) | **0.911** |
| Bottom-left | **0.878** |

All three land in the same range as the whole-region results. Caveat worth
being explicit about: these corners aren't perfectly anatomy-free — depending
on patient positioning some frames show a sliver of shoulder/arm silhouette
at the edge of the crop — so this doesn't cleanly isolate "zero anatomy," but
it's a much smaller, fixed, mostly-background window than any labeled organ
region, and it still lands at 0.88–0.91.

**High-pass residual** — subtract a heavy Gaussian blur (radius 20) from the
*original* image before any masking (so the residual isn't dominated by the
mask rectangle's own hard edge), then blank the lung region in the residual
(`highpass_complement.py`). Anatomy mostly disappears in a high-pass
residual; acquisition-level texture (sharpening kernel, compression,
detector noise) survives.

| Shenzhen-only | Val AUC |
|---|---|
| High-pass residual, lung blanked | **0.936** |

Nearly matches the full complement (0.933–0.951) despite anatomy being
mostly filtered out, with a clean stable trajectory (not noise like the
shuffle control). Together with the patch test, this is the strongest
evidence for the pervasive-texture explanation: a bare corner patch and a
denoised/anatomy-flattened residual both carry most of the signal the full
region decomposition found distributed everywhere.

**Montgomery replication** — same inverse-mask ablation, independent source:

| Montgomery-only, in-domain | Val AUC |
|---|---|
| Full frame (baseline) | 0.940 |
| **Lung field blanked out (complement only)** | **0.810** |

**Replicates.** The gap is larger here (0.13 vs. Shenzhen's 0.03) — Montgomery's
val split is only ~21 images, so this point estimate is noisier than
Shenzhen's, but 0.81 is nowhere near the ~0.5 the label-shuffle control
lands at. This is no longer a Shenzhen-specific quirk: it holds, with
different magnitude, in a second independently-collected dataset.
**Not yet run:** the same check on TBX11K (deliberately deprioritized —
different collection design, much larger, and two independent replications
already establish the pattern more cheaply than a third would).

**Projection probe** (`eval.projection_probe`) — a different, narrower
question on the pooled Shenzhen+TBX11K checkpoint: does the encoder's
penultimate feature space have a single linear direction encoding *which
source dataset* an image came from that's also load-bearing for its TB
prediction? Fitting a source-classifier probe on frozen features, projecting
that one direction out, and refitting the TB-label probe:

| | In-domain val AUC |
|---|---|
| Before projecting out source direction | 0.9906 |
| After projecting out source direction | 0.9890 |

Only a 0.0017 drop — the model's TB prediction does not hinge on a simple
single linear "which dataset is this" feature, despite the source-classifier
probe itself being 94.5% accurate on the same features. **This does not
contradict the inverse-mask result above** — they test different confounds.
The inverse-mask ablation is about position/border/marker cues correlated
with the label *within* Shenzhen; the projection probe is about *cross-source
identity* being linearly encoded and load-bearing. A single projected-out
direction is also a conservative test — it can't catch a multi-dimensional or
nonlinear source confound, so a small drop here is weaker evidence of "no
problem" than a large drop would be evidence of "problem confirmed."

**Bottom line — reframed per the ceiling/inverse-mask evidence, not just
"does it clear TPP":** discrimination, not calibration, is what's failing
cross-source (ceiling analysis), lung-cropping recovers ranking quality but
not threshold transfer (lung-crop ablation), and a large share of even the
in-domain numbers is inflated by an acquisition-correlated confound that
needs no cross-source comparison to see (inverse-mask ablation). The
defensible claim this pipeline currently supports is: **cross-source
threshold transfer fails catastrophically on these public TB CXR sets even
when ranking partially survives, lung-cropping fixes the ranking without
fixing the calibration, and a large share of in-domain performance is
explained by acquisition cues rather than thoracic signal.** Per the Phase 3
gate (§12 of the proposal), that blocks starting Head B until Head A closes
this gap — candidate next steps, roughly in order of expected value per
GPU-hour and **not yet run**:
1. Per-image intensity normalization (CLAHE or z-scoring *inside* the lung
   mask) — attacks the exposure/contrast shift directly, and is a
   preprocessing change, not a training change.
2. Aggressive photometric augmentation (gamma, contrast, brightness, blur,
   resolution jitter, additive noise) — same confound, from the training side.
3. Multi-source training with each source held out in turn, rather than the
   two directions tested so far.
4. Per-deployment threshold calibration (see the proposal's §5
   pre-registration) — legitimate for the calibration gap the ceiling
   analysis still shows on the TBX11K direction, but does **not** address the
   discrimination gap the ceiling analysis shows on Montgomery.

**Updated after the full validity-check sequence (dilation, region
decomposition, label-shuffle control, single-patch test, high-pass residual,
Montgomery replication):** the inverse-mask finding survives every way it
could plausibly have been an artifact. It's not tight-mask under-coverage of
the apices/costophrenic angles (dilating +25px strengthened it). It's not a
pipeline/split leak (the label-shuffle control collapses to ~0.5, once you
use the epoch trajectory instead of the misleading best-checkpoint stat).
It's not concentrated in one attributable region — it's distributed across
every non-lung region tested, a bare fixed-location 96×96 corner patch, and
an anatomy-flattened high-pass residual all land in the same 0.87–0.95 band,
which is the signature of a pervasive acquisition/processing texture rather
than a single fixable bug like corner-marker leakage. And it's not a
Shenzhen-specific quirk: it replicates, at a different magnitude, in the
independently-collected Montgomery set. TBX11K replication was deliberately
not run — two independent replications already establish the pattern.

The accurate framing of this project's current deliverable is a **confound
audit for public TB CXR benchmarks** — sens@spec70 ceilings, cross-source
threshold-transfer failure, and a convergent inverse-mask/region-decomposition/
patch/high-pass/replication evidence chain — rather than a TB screening tool.
That is a coherent, defensible piece of work on its own terms, and the honest
one to report even though it isn't the deliverable the proposal originally
set out to build. It still needs an actual literature check (§ above) before
any claim of novelty, and the label-shuffle control's own reporting caveat
is worth carrying forward: report epoch trajectories, not just best-checkpoint
numbers, whenever a result needs to be trusted rather than just glanced at.

---

## Robustness interventions (2026-08-25)

Two candidate fixes tried against the confound, ranked by expected value per
GPU-hour. **In-domain val AUC is a known-inflated target metric now** — any
tuning loop optimizing it will preferentially select models that read
acquisition texture, since that's the easiest available signal. The
in-domain complement-AUC (lungs blanked, `eval/complement_monitor.py`) is
tracked as a standing diagnostic instead: if it stays near the no-intervention
baseline (Shenzhen: 0.933) while the headline number climbs, nothing has
actually been fixed.

**Montgomery is confirmatory only — direction of effect, never magnitude.**
With ~58 positives, its bootstrap CIs run ±13-15 points; every Montgomery
comparison below is consistent with "no detectable difference" even where
the point estimates move a lot. **TBX11K (n=8260, CIs ~±3 points) is the
decision metric** for everything in this section.

First pass (Steps 1-2) had a methodology bug worth naming rather than
burying: Step 2 was layered on top of Step 1, so "texture aug: 73.2%" was
actually *CLAHE + texture aug*, measured against a preprocessing choice
already shown to cost ~8 points on TBX11K. That comparison was confounded.
Corrected below.

| Configuration | TBX11K ceiling sens@70 (95% CI) | Shenzhen in-domain complement-AUC |
|---|---|---|
| No intervention (full frame) | 78.5% [75.4, 81.6] | 0.933 |
| Lung-crop + in-mask CLAHE (confounded Step 1) | 70.9% [67.3, 74.5] — regression | not tested |
| **Lung-crop only, no CLAHE (clean reference)** | **84.4% [81.0, 87.1]** | n/a — crop removes the complement concept |
| Lung-crop + texture-aug, **mild** | **92.6% [90.3, 94.6]** | 0.893* |
| Lung-crop + texture-aug, medium | 89.5% [86.8, 91.8] | not tested |
| Lung-crop + texture-aug, aggressive | 88.3% [85.6, 90.9] | not tested |

*The 0.893 complement-AUC in the mild row is texture-aug **alone** (no
crop) — the number that was already available when this table was built.
The complement-AUC for the actual winning config (crop **+** mild-aug
combined) required a new run and is reported separately below, with a
caveat the aug-alone number doesn't need.

**Clean lung-crop alone is a real, substantial win on TBX11K** (78.5% →
84.4%) — confirming the earlier CLAHE result wasn't just "crop doesn't help
here," it was specifically CLAHE actively hurting. **Mild texture
augmentation on top pushes the ceiling to 92.6% [90.3, 94.6] — the entire CI
meets the WHO TPP triage sensitivity/specificity targets on held-out TBX11K
(retrospective research split; no prospective or prevalence-adjusted
evaluation), for the first time this session,** with zero overlap against
the clean baseline's CI. This is not Montgomery-sized noise; at this N the
separation is real. (This is one direction, on one held-out source, selected
by a sweep run on that same held-out set — see Limitations below.)

**Frozen-threshold operating point at the winning config (`eval/frozen_threshold_check.py`,
same checkpoint as the 92.6% ceiling, no retraining).** The 92.6% number
above is a *ceiling* — best achievable sensitivity at spec≥70% under a
threshold chosen on TBX11K itself. Deriving the threshold the honest way
instead — frozen from validation (Shenzhen+Montgomery, val AUC 0.951),
never touching TBX11K until scoring — and applying it to held-out TBX11K:

| | sens | spec | meets WHO TPP minimum (spec≥70%)? |
|---|---|---|---|
| Ceiling (per-site threshold, TBX11K-derived) | 92.6% [90.3, 94.6] | ≥70% by construction | — |
| **Frozen threshold (validation-derived)** | **98.2% [97.1, 99.2]** | **44.0% [42.8, 45.1]** | **✗ no** |

**This does not meet the WHO TPP bar out of the box.** A global threshold
frozen from Shenzhen+Montgomery validation data massively over-triages on
TBX11K — 98.2% sensitivity but only 44.0% specificity, far below the 70%
floor, essentially the same over-triage failure mode already documented for
the pre-lung-crop baseline (line ~342: 95.3%/30.0%). Lung-crop + mild-aug
fixed *discrimination* (the ceiling moved from 78.5%→92.6%) but did **not**
fix *threshold transfer* — the val→TBX11K calibration gap survives the fix
essentially untouched. The 92.6% headline is real as a statement about what
this encoder can discriminate, but reading it as "deploys and meets WHO TPP"
without a per-site recalibration step is not supported by this checkpoint.
Per-deployment threshold calibration (proposal §5) is not optional here —
it is the difference between 44% and 70%+ specificity at this operating
point. This reproduces the same qualitative finding as the pre-lung-crop
baseline table above and the Montgomery-direction table below (val-derived
thresholds transfer sensitivity far better than specificity across sources)
— it is a property of frozen cross-source thresholds on these datasets, not
something lung-crop + mild-aug was ever positioned to fix.

**Mechanism (`eval/score_distribution_diagnosis.py`, same checkpoint, no
retraining) — not a uniform prevalence shift, specifically a negative-class
spread problem.** Score histograms, split by label, val vs. held-out:

![Score distributions: validation vs. held-out TBX11K, split by label](figures/tbx11k_winning_score_histograms.png)

Quantified via per-class median offset and IQR ratio (held-out IQR / val
IQR):

| | median offset (held − val) | IQR ratio (spread) |
|---|---|---|
| Negative | +0.126 | **7.94×** |
| Positive | +0.004 | 0.36× |

If this were a "wholesale" shift from prevalence/class-balance differences,
both classes would shift by a similar amount and keep roughly their shape.
That is not what happens: **positives barely move and if anything get
tighter** (offset ≈0, IQR shrinks to 0.36×) — the model is *more* consistent
on TBX11K positives than on validation positives, consistent with the very
high, stable frozen-threshold sensitivity (98.2%). **Negatives are the
problem**: val negatives are tightly clustered near 0 (median 0.047, IQR
0.085) — an easy population, mostly Shenzhen/Montgomery normals — while
TBX11K negatives spread across nearly the full [0, 1] range (median 0.173,
IQR 0.675, visible as the long right tail in the lower panel above). A
threshold calibrated against the tight validation-negative cluster sits
well inside that tail, misclassifying a large fraction of TBX11K negatives
as positive — exactly the 44% specificity collapse. The composition check
below tests, and confirms, the specific source.

**Negative-class composition (`eval/negative_composition_probe.py`, same
checkpoint, no retraining) — this is a training-data gap, not just
acquisition texture.** TBX11K's raw annotations distinguish `healthy` from
`sick_but_non-tb` within what this codebase (like most TB CXR benchmarks)
collapses into a single negative class — Shenzhen and Montgomery's
negatives, by contrast, are almost entirely `healthy`. If the model's
training negatives never included sick-but-not-TB cases, it never learned
where to place them, and they should land unpredictably high. Splitting
TBX11K's held-out negatives by that raw tag and scoring each subgroup
separately:

![TBX11K negative-class scores by subgroup: healthy vs. sick_but_non-tb](figures/tbx11k_winning_negative_composition.png)

| Subgroup | n | median score | IQR | % scoring > 0.5 |
|---|---|---|---|---|
| healthy | 3,800 | 0.055 | 0.069 | 1.6% |
| sick_but_non-tb | 3,800 | **0.722** | 0.530 | **66.5%** |

**Confirmed, and more sharply than expected.** `healthy` reproduces the
tight validation-negative distribution almost exactly (median 0.055 vs.
validation's 0.047). `sick_but_non-tb` is a different population entirely —
median score 0.722, meaning the *typical* sick-but-not-TB case is scored
as TB-positive, not just occasionally confused. Of the 2,588 TBX11K
negatives scoring above 0.5 (the long tail visible in the earlier
histogram), **97.6% are `sick_but_non-tb`** against a 50% base rate in the
negative pool — the tail is almost entirely this one subgroup. The model
learned "sick lung tissue" where it needed "TB-specific lung tissue,"
because its training negatives (Shenzhen/Montgomery, overwhelmingly
healthy) never taught it the difference. This is a known failure mode in
real deployed TB CAD, not just this pipeline: an independent field study of
a commercial CAD system (CAD4TB) in a TB prevalence survey found 46.7%
(245/525) of CXRs it scored highly without confirmed TB had identifiable
non-TB abnormalities (pleural disease, cardiomegaly, non-TB pneumonia,
nodules) — Ngosa, Moonga, Shanaube et al., *BMC Infectious Diseases*, 2023.
That study doesn't attribute its finding to training-data composition (it
wasn't investigating that), so it corroborates the *phenomenon* — non-TB
abnormal cases drawing high TB scores — not this document's specific
*causal* claim about negative-class composition; the causal claim here
rests on the subgroup-split experiment above, which is this codebase's own
result.

**Unlike the acquisition-texture confound, this is fixable at the source,
not just calibratable around.** Per-site threshold recalibration (below)
compensates for it operationally; the direct fix is including non-TB
pathology in the training negative class so the model learns the
distinction it's currently missing entirely. Worth trying before further
calibration-side work, since it would compress the tail itself rather than
work around it — not yet run.

**Deployment spec: how many labeled *negatives* does a new site need
(`eval/score_distribution_diagnosis.py`'s calibration-set-size sweep, same
checkpoint, no retraining)?** Positives contribute nothing to placing a
spec-target threshold — it's a percentile of the negative score
distribution alone — so the sweep is parametrized directly by negative
count, not total calibration-set size. For n ∈ {25, 50, 100, 150, 200, 300,
400, 500} negatives: draw n negatives at random from held-out TBX11K, set
the threshold at their 70th percentile, apply it to the rest of the
held-out set (both classes), and record the achieved specificity there.
Repeated 500× per size:

![Achieved specificity vs. calibration set size, with 10th-90th percentile band](figures/tbx11k_winning_calibration_size_sweep.png)

| n (labeled negatives) | mean achieved spec | 10th–90th pct | P(within ±5pt of 70%) |
|---|---|---|---|
| 25 | 68.4% | [56.1%, 79.5%] | 41.6% |
| 50 | 69.3% | [61.2%, 77.6%] | 57.0% |
| 100 | 69.7% | [63.6%, 75.5%] | 72.0% |
| 150 | 69.7% | [64.6%, 74.3%] | 80.2% |
| 200 | 69.9% | [65.8%, 73.9%] | 87.2% |
| 300 | 70.0% | [66.2%, 73.5%] | 93.4% |
| 400 | 70.0% | [66.7%, 72.8%] | 96.4% |
| 500 | 70.1% | [67.3%, 73.1%] | 98.8% |

(One canonical run shown, matching the saved plot; the held-out `DataLoader`
isn't seed-pinned for shuffling, so a rerun lands within ~1-2 points of
these per size — the trend and the ≈250-300 conclusion below are stable
across reruns, only the last digit moves. TBX11K's own negative fraction
happens to be 92.0% (7,600/8,260), so on *this* dataset a random
total-case draw is already negative-dominated and the total-vs-negative
distinction barely moves the numbers — but the negative-count framing is
the correct one to report, since it is what is actually binding, and a
real deployment site's case mix won't necessarily match TBX11K's curated
composition.) **The mean is essentially unbiased even at n=25** — small
calibration sets aren't systematically wrong on average — but the *spread*
is what makes small sets unusable: at n=25 there's roughly a 1-in-2 chance
of landing more than 5 points from target in either direction, which given
the confidence-interval width could mean shipping a site at 56% specificity
while believing it's at 70%.

**Why the spread is this large: the threshold sits in a sparse valley
between two subpopulations, not just "a tail."** The pooled held-out
negative distribution's 70th percentile — the value a calibration set is
trying to estimate — is **0.607**. Given the composition split above,
that value sits almost exactly between the `healthy` cluster (98.4% of
mass below 0.5) and the `sick_but_non-tb` cluster (median 0.722, the
majority above 0.5): a bimodal mixture with a low-density valley in
between. Order-statistic theory says a sample quantile's variance scales
inversely with the local density at that quantile — flat, low-density
regions make small-sample quantile estimates noisy even when the mean of
the same data would be stable, which is exactly the asymmetry observed
here (unbiased means, wide spreads, even at n=25). The bimodality found in
the composition check isn't just an explanation for *why* the frozen
threshold fails — it's also why *calibrating* that threshold from a small
sample is intrinsically harder than a unimodal negative distribution would
be: the 70th percentile falls precisely in the region a small sample is
least likely to characterize well.

**Concrete deployment spec: ≈250–300 labeled negatives, drawn from the
target site's own population (not rebalanced), to land within ±5 points of
the 70% specificity target with ≥90% confidence.** Below ~100, per-site
calibration is closer to noise than correction. This is now grounded in
both an empirical sweep and the mechanism that explains its shape, not
just an empirical curve read off a plot.

**Where this leaves the headline claim — three parts, each now evidenced
separately, not asserted together:**
1. Discrimination meets the WHO TPP triage bar on held-out TBX11K, under a
   per-site-recalibrated threshold (the 92.6% ceiling).
2. A frozen global threshold does not transfer (44.0% specificity), and the
   mechanism is a training-data gap in negative-class composition — TBX11K
   includes sick-but-non-TB negatives the training sources never did — not
   a uniform distributional offset.
3. Deployment therefore requires roughly 250–300 site-specific labeled
   negatives to recalibrate reliably; below ~100, calibration doesn't help.

Reading any one part as the whole story overclaims; reading all three
together is what this pipeline currently supports.

**Retraining experiment: fix the training-data gap at the source
(`train_diagnostic.py --held-out tbx11k-val`, new).** TBX11K already has
the labels needed to test this directly — no new data collection. Added a
new held-out mode: TBX11K's **train** split (6,496 images, including its
sick_but_non-tb negatives) joins the training pool instead of being fully
withheld; only TBX11K's **val** split (1,764 images, disjoint from that
train split) is held out. Same recipe otherwise (lung-crop + mild-aug),
same validation source for the frozen threshold (Shenzhen+Montgomery).
Four falsifiable predictions were made before running this:

1. The healthy/sick_but_non-tb bimodality collapses to unimodal.
2. Frozen-threshold specificity recovers well above 44%.
3. The calibration-set-size requirement drops from ≈250–300 toward ≈50–100.
4. sens@spec70 probably *falls* below 92.6%, because distinguishing TB from
   other lung pathology is a genuinely harder task than distinguishing TB
   from healthy — and if it doesn't fall, that's worth explaining, not
   quietly accepting.

**Important caveat before the results: this is not a fair cross-source
test, by construction.** Letting TBX11K's train split join training also
eliminates TBX11K as a naive, never-seen source — the held-out val split
now shares scanner/protocol/population with its own training split in a
way Montgomery and Shenzhen never did with each other. Any number below
should be read as "does the training-data fix work," not as a new,
higher headline replacing 92.6% — that comparison is confounded and this
document says so explicitly rather than letting a bigger number stand in
for a fixed cross-source claim.

Matched comparison — same 1,764 held-out images scored by both checkpoints
(old = no TBX11K in training at all; new = TBX11K's train split added):

| | Held-out AUC | Frozen-threshold sens/spec | Ceiling sens@spec70 (95% CI) |
|---|---|---|---|
| Old checkpoint (`step3_mild_lungcrop_tbx11k`) | 0.894 | 97.6% / **43.8%** ✗ | 92.7% [87.4, 96.4] |
| **New checkpoint (`step5_negcomp_tbx11k-val`)** | **0.998** | 97.6% / **97.6%** ✓✓ | **100.0%** [100.0, 100.0] |

**Predictions 1 and 2: confirmed, sharply.**

![TBX11K negative-class scores by subgroup after retraining: healthy vs. sick_but_non-tb, held-out val split only](figures/tbx11k_negcomp_retrain_negative_composition.png)

| Subgroup (held-out val only) | median score, before | median score, after |
|---|---|---|
| healthy | 0.055 | 0.255 |
| sick_but_non-tb | 0.722 | **0.270** |

The two subgroups are now nearly indistinguishable (IQR 0.028 vs. 0.042) —
the bimodality is gone. The tail didn't vanish entirely (46/800
sick_but_non-tb, 5.8%, still score >0.5 vs. 4/800, 0.5%, for healthy — an
~12× relative rate, down from ~42× before) but it went from *the dominant
pattern in the data* to a small residual. Frozen-threshold specificity went
from 43.8% to 97.6% on the identical images — the training-data fix is
real, not just directionally suggestive.

**Prediction 3: not confirmed — and the reason why corrects an earlier
claim in this document.**

![Calibration-set-size sweep after retraining, unimodal negative distribution](figures/tbx11k_negcomp_retrain_calibration_size_sweep.png)

| n (negatives) | P(within ±5pt of 70%), before (bimodal) | after (unimodal) |
|---|---|---|
| 100 | 72.0% | 71.0% |
| 200 | 87.2% | 84.4% |
| 300 | 93.4% | 90.0% |

**Essentially unchanged**, despite the negative distribution going from
sharply bimodal to tightly unimodal. This falsifies the specific mechanism
claimed earlier in this section — that the calibration curve's shape comes
from the 70th-percentile threshold sitting in a low-density valley between
two modes. If that were the driver, removing the valley should have made
small-sample calibration much more reliable, and it didn't. **The better
explanation, consistent with both curves:** treating a calibration
threshold's *achieved coverage on new data* as a rank statistic (the
probability-integral transform maps any continuous distribution's order
statistics onto Uniform(0,1) order statistics) makes its sampling variance
≈ p(1−p)/n **regardless of the underlying distribution's shape** — at
p=0.70, that predicts std ≈ 0.046 at n=100 and ≈ 0.027 at n=300, matching
both the bimodal and unimodal curves' observed std (0.046–0.048 and
0.027–0.028) closely. This is the same distribution-free reasoning behind
Riley et al.'s external-validation sample-size formulas (Literature check,
below) — it explains why *this* number didn't move even though the
mechanism story from before did. Worth stating plainly: the earlier
"sparse valley" explanation was a plausible-sounding mechanism that this
follow-up experiment was positioned to test almost by accident, and it
didn't survive the test — left in this document rather than quietly
corrected, because the correction is itself informative about which
explanation actually generalizes.

**Prediction 4: not cleanly testable with this design — say so rather than
force a reading.** The ceiling rose to 100%, the opposite of the predicted
direction. But this experiment changed two things at once (training
negative-class composition *and* cross-source purity), and the
in-distribution effect plausibly dominates: a model that has seen TBX11K's
own train split should ace TBX11K's val split regardless of how hard the
negative class is, simply because train/val here share a source in a way
no other comparison in this document does. This design cannot isolate
"is TB-vs-other-pathology harder than TB-vs-healthy" from "is in-source
val easier than cross-source val" — both point the same direction here, so
a clean answer needs a different experiment (e.g. sick_but_non-tb-labeled
negatives from a source disjoint from the held-out evaluation, which no
dataset in this codebase currently provides). Flagged as open, not resolved.

**Retrospective corroboration, in data already collected before this
experiment was designed.** The Montgomery-direction winning-config run
(`stage1_montgomery_winning`, reported earlier in this section) trained on
Shenzhen+TBX11K — which already put TBX11K's train-split sick_but_non-tb
cases in its training negatives, for the unrelated reason that Montgomery
was the held-out source that run. Its frozen-threshold result was 53.4%
sens / **97.5%** spec — the *only* frozen-threshold result in this entire
document that clears WHO TPP before this retraining experiment was run.
That's consistent with the causal claim here, sitting unnoticed in results
already reported earlier, not cherry-picked after the fact to fit it.

The strength sweep confirms the exact mechanism predicted: TB findings
(cavitation, miliary nodules, reticulonodular infiltrate) are themselves
high-frequency texture, so aggressive randomization of sharpening/noise/
compression destroys diagnostic and confound texture together. **Mild beats
aggressive by 4.3 points** (92.6% vs. 88.3%) despite touching far less of the
image. The confirmatory complement-AUC check on the mild-strength checkpoint
(same in-domain Shenzhen methodology, `eval/complement_monitor.py`) reads
0.893 — a real but smaller drop from the 0.933 baseline than aggressive's
0.848, i.e. mild removes *less* raw confound reliance yet delivers *far
more* held-out sensitivity. That's the signature of an augmentation strength
sitting closer to the point that trades away shortcut-texture without
trading away diagnostic-texture, exactly the dial the sweep was built to find.

**Live hypothesis worth stating, not resolving:** every complement-AUC drop
in this section could be read two ways — "the intervention removed shortcut
reliance" (good) or "the intervention removed real skill along with the
shortcut, and the resulting held-out number is a less-inflated but still
partial estimate" (less good, but not actually a failure — it would mean the
baseline's apparent skill was itself partly confound). Both readings predict
a complement-AUC drop; they disagree about how much of the *held-out* gain
is trustworthy. Mild aug's result — complement-AUC barely moved (0.933→0.893)
while held-out ceiling jumped 8+ points — favors the first reading here,
but this should be treated as evidence, not proof, until replicated.

**Montgomery direction at the winning config (confirmatory).** The 92.6%
result above is one direction only (Shenzhen+Montgomery trained → held-out
TBX11K). Running the identical recipe (lung-crop + mild-aug) in the reverse
direction, Shenzhen+TBX11K trained → held-out Montgomery:

| | Val AUC | Held-out AUC | Frozen-threshold sens/spec | Ceiling sens@70 (95% CI) |
|---|---|---|---|---|
| No intervention (baseline) | 0.997 | 0.580 | 37.9% / 82.5% | 44.8% [31.5, 61.5] |
| **Winning config (crop + mild-aug)** | 0.994 | 0.796 | 53.4% / 97.5% | **75.9% [65.4, 87.0]** |

Same direction of effect as TBX11K, and — despite Montgomery's usual
resolution limits — this particular jump (44.8→75.9) is large enough that
the two CIs don't actually overlap (baseline tops out at 61.5, winning
config starts at 65.4). Read as confirmatory: the fix helps in both
directions, magnitude uncertain in this one but the effect itself is not
plausibly just noise here.

**Complement-AUC for the actual winning config (crop + mild-aug combined,
not aug alone).** Training Shenzhen-only with the full winning recipe
(`--lung-crop --texture-aug --aug-strength mild`) and scoring it against the
+25px-dilated complement (`eval/complement_monitor.py
--complement-variant lungcomplement_d25`): **0.634** — the lowest
complement-AUC measured all session, well below aug-alone's 0.893 and much
closer to the label-shuffle control's ~0.5 floor than to the 0.933 baseline.

**Caveat this number needs that the others didn't:** this checkpoint was
*trained* on lung-crop images (tight, zoomed lung-only crops) and *scored*
on full-frame complement images (wide field of view, lungs blanked) — a
real distribution shift in framing and scale, not just content. Some of the
0.634 is very likely the model simply not knowing what to do with an
input shape it never trained on, stacked on top of any genuine confound
reduction. This number is a *ceiling* on how much confound-reliance could
be claimed as removed, not a clean isolated measurement of it the way the
full-frame-vs-full-frame comparisons (0.933→0.893, aug alone) are. Treat it
as suggestive, not as directly comparable to the other rows in this section.

### CXR-pretrained encoder (DenseNet-121, torchxrayvision)

Swapped `XRVDenseNetEncoder` (`--backbone xrv_densenet121-res224-all`) in on
top of the winning config (lung-crop + mild aug, held-out=tbx11k):

| Encoder | TBX11K ceiling sens@70 (95% CI) | Val AUC → held-out AUC (gap) |
|---|---|---|
| EfficientNet-B0 (ImageNet) | 92.6% [90.3, 94.6] | 0.951 → 0.889 (−0.061) |
| **DenseNet-121 (CXR-pretrained)** | **92.0% [89.9, 94.0]** | **0.920 → 0.909 (−0.011)** |

**Statistically indistinguishable ceilings** — the CIs overlap almost
entirely, so the CXR-pretrained encoder did not deliver an additional win on
top of what crop + mild-aug already captured. But the *shape* of the result
differs in a way worth keeping: DenseNet reached its number with almost no
val→held-out gap (0.011, essentially no cross-source degradation), while
EfficientNet's held-out number is a real 0.061 drop from a higher
in-domain fit. Same destination, different route — DenseNet's features seem
inherently less tied to source-specific texture (consistent with the
mechanistic case for CXR- over ImageNet-pretraining), it just didn't have
further ground to make up once lung-crop + mild-aug had already done most of
the work. Not swapped in as the default given no measured gain, but worth
keeping in mind if a future intervention needs a smaller val→held-out gap
specifically (e.g. if per-site calibration set size is very small, a smaller
gap means the frozen-threshold number is closer to the ceiling to begin with).

### AdaBN — test-time BatchNorm adaptation

`eval/adabn.py`, on top of the mild-aug lung-crop winner. Free: no labels, no
backward pass, one forward pass per target image to recompute BatchNorm
running statistics on the held-out source's own (unlabeled) images before
scoring on those same images — the standard AdaBN protocol; there is no
separate adaptation split by construction, since the method exists
specifically for the case where a deployment site has only its own unlabeled
images.

**Hit and fixed a real bug in the reference implementation first** (present
in the original `robust_tb.py` module this whole intervention plan grew out
of): the naive pattern — set BN layers to `.train()`, then call `.eval()` on
every *other* module — doesn't work, because `nn.Module.eval()`/`.train()`
recurse into all children. Calling `.eval()` on a parent container silently
resets any BN leaf underneath it back to eval mode too, if that parent is
visited after the BN leaf during iteration. First symptom was a full
`reset_running_stats()` going numerically unstable (activations to 10^7
within a few batches, NaN predictions after adaptation) — switching to
blending from the trained stats with a small momentum (0.05) instead of a
full reset fixed that, but then the delta came back as *exactly* zero
(AUC −0.0000), which was the real tell: adaptation was silently a no-op the
whole time. Fix: call `model.eval()` once first, then set `bn.training = True`
directly on each BN leaf afterward — never re-run `.eval()`/`.train()` on
any container after that point.

| | AUC | sens@spec70 (95% CI) |
|---|---|---|
| Before AdaBN | 0.8894 | 92.6% [90.3, 94.6] |
| After AdaBN | 0.9057 | 92.4% [90.2, 94.5] |
| Delta | +0.0164 | −0.0015 (noise) |

A real, small AUC gain; no measurable movement on the decision metric — the
CIs are essentially identical. Free and harmless, worth keeping at
deployment time, but not a meaningful additional lever on top of what
lung-crop + mild-aug already captured.

**Not yet run:** source-adversarial training (DANN) — deliberately last, per
the plan, since it's unstable and easy to fool without the complement-AUC
check already in place to verify it isn't just satisfied superficially.
**Revised priority after the frozen-threshold finding above:** Steps 1-4
fixed discrimination (the 92.6% ceiling) but left threshold transfer
essentially broken (44.0% frozen-threshold specificity, well under the 70%
floor) — that is precisely the failure mode DANN-style domain-invariant
training targets, so its expected marginal value is higher than originally
assessed here, not lower. Per-deployment threshold calibration (proposal
§5) is the cheaper fix to try first since it needs no retraining; DANN is
the next lever if calibration alone can't close the gap. Still optional for
now, but no longer for the reason originally given.

---

## Limitations

The headline result — lung-crop + mild texture augmentation gets held-out
TBX11K discrimination to the point where a per-site-recalibrated threshold
meets the WHO TPP triage sensitivity/specificity targets (retrospective
research split; no prospective or prevalence-adjusted evaluation) — is real
but narrower than it can sound quoted alone:

- **The 92.6% figure is a ceiling, not a deployable operating point.** A
  threshold frozen from validation and applied to TBX11K without
  recalibration (`eval/frozen_threshold_check.py`) gives 98.2% sensitivity
  but only 44.0% [42.8, 45.1] specificity — well below the 70% WHO TPP
  floor. Lung-crop + mild-aug fixed discrimination (the ceiling), not
  threshold transfer (the calibration gap is essentially unchanged from the
  pre-lung-crop baseline). Quoting 92.6% as "this model meets WHO TPP"
  without the per-site-recalibration caveat overstates what was
  demonstrated; see the Phase 2 validation results section for the full
  frozen-threshold-vs-ceiling table. **The mechanism is a training-data
  gap, not just calibration:** TBX11K's negative class includes
  sick-but-non-TB cases that Shenzhen/Montgomery's (overwhelmingly healthy)
  training negatives never taught the model to place — confirmed directly
  by splitting held-out negatives on their raw annotation (median score
  0.722 for sick-but-non-TB vs. 0.055 for healthy, `eval/negative_composition_probe.py`).
  Per-site recalibration compensates for this but doesn't fix it. **The
  direct fix was tried** (`--held-out tbx11k-val`, adding TBX11K's train
  split to training): on the identical held-out images, frozen-threshold
  specificity went from 43.8% to 97.6% and the healthy/sick_but_non-tb
  bimodality collapsed — but this specific test isn't a fair cross-source
  comparison (TBX11K informed training, so its own val split is no longer
  a naive source), so it demonstrates the mechanism is real and fixable,
  not a new validated headline number. It also falsified this document's
  own earlier explanation for *why* the calibration-set-size curve is
  wide — that curve barely moved even with the bimodality gone, pointing
  instead to a distribution-free order-statistic effect (≈p(1−p)/n) rather
  than the bimodal-valley story originally proposed; see the retraining
  subsection above. The recalibration workaround still has a concrete
  cost when the training-data fix isn't available: ≈250–300 labeled
  *negatives* from the target site's own population to land within ±5
  points of the 70% specificity target with ≥90% confidence
  (`eval/score_distribution_diagnosis.py`); below ~100, per-site
  calibration is closer to noise than correction.
- **Single-direction strength.** The 92.6% [90.3, 94.6] result is one
  direction only — Shenzhen + Montgomery trained, TBX11K held out. The
  reverse direction (Shenzhen + TBX11K trained, Montgomery held out) is
  reported separately above with much wider CIs; it is confirmatory (same
  direction of effect) but structurally cannot match this precision. Treat
  the strong claim as belonging to one direction, not to the pipeline in
  general.
- **No untouched holdout remains.** All three sources (Shenzhen, Montgomery,
  TBX11K) informed tuning decisions somewhere in this process — which source
  to hold out, the augmentation-strength sweep, the lung-crop margin/dilation
  — even though the *final* reported number for each configuration was
  scored on data that configuration's own training never saw. There is no
  fourth, fully naive source left to confirm the winning recipe generalizes
  to a TB CXR set nobody has looked at yet.
- **The augmentation-strength sweep selected mild on the same TBX11K
  held-out set it's being reported against.** Three configurations (mild/
  medium/aggressive) were compared on TBX11K sens@spec70 and the best was
  kept — that TBX11K number is a selection-optimized result, not an
  independent confirmation. The complement-AUC checks (measured on Shenzhen,
  not TBX11K) are the closer-to-independent evidence that mild's mechanism
  is real rather than a lucky draw among three.
- **Montgomery structurally cannot resolve differences below ~15 points.**
  With ~58 positives, its bootstrap CIs run ±13-15 points — most Montgomery
  comparisons in this document are consistent with "no detectable
  difference" even where point estimates move substantially, and should be
  read for direction of effect only, never magnitude. The one exception is
  the winning-config comparison (44.8% [31.5, 61.5] → 75.9% [65.4, 87.0]),
  where the jump happens to be large enough that the CIs don't overlap —
  still treat this as a smaller-N result than the TBX11K number, not as
  equally precise.
- **Residual complement AUC.** The winning configuration's complement-AUC
  (in-domain, Shenzhen, lung-crop + mild-aug combined, scored against the
  +25px-dilated complement) is **0.634** — real progress from the 0.933
  baseline, but nowhere near the label-shuffle control's ~0.5 floor. Some
  acquisition-texture reliance survives the fix; the honest claim is
  "substantially reduced," not "eliminated." This particular number also
  carries a distribution-shift caveat (crop-trained model scored on
  full-frame input) not shared by the other complement-AUC figures in this
  document — see the Phase 2 validation results section for the full
  caveat before quoting 0.634 on its own.

---

## Literature check

A targeted (not exhaustive — a handful of searches, not a systematic
review) check for prior work on Shenzhen/Montgomery confounding, shortcut
learning in CXR classifiers, and cross-institution TB screening
generalization, done before any writeup claims novelty.

**The phenomenon is known, not novel:**
- **FairALM** (Lokhande, Akash, Ravi & Singh, ECCV 2020, [arXiv:2004.01355](https://arxiv.org/abs/2004.01355))
  trains a TB classifier on exactly Shenzhen + Montgomery and explicitly
  treats *site* (which hospital) as a nuisance variable to decorrelate from
  the label, for precisely the reason this project's audit chain found: "a
  model may cheat and use site-specific (rather than disease-specific)
  artifacts in the images for prediction." Same two datasets, same
  underlying concern, five years earlier — the confound's *existence* is
  prior art.
- **DeGrave et al.**, "AI for radiographic COVID-19 detection selects
  shortcuts over signal," *Nature Machine Intelligence* 2021
  ([PMC7523163](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7523163/)) —
  the landmark CXR shortcut-learning paper. Different disease (COVID, not
  TB) and a different method (saliency maps / Expected Gradients, not a
  direct ablation), but the same structural finding: models reading
  non-lung regions (lead markers) rather than pathology, appearing accurate
  in-distribution and failing at new sites.
- **Rafferty & Rajan**, "Limitations of Public Chest Radiography Datasets
  for Artificial Intelligence," arXiv 2509.15107 (2026) — a recent survey
  that names border/corner/marker/scanner-artifact shortcuts as a known,
  general limitation of public CXR datasets.
- **Ngosa, Moonga, Shanaube et al.**, *BMC Infectious Diseases*, 2023
  ([PMC10408069](https://pmc.ncbi.nlm.nih.gov/articles/PMC10408069/)) — in a
  real TB prevalence survey, 46.7% (245/525) of CXRs a deployed commercial
  CAD system (CAD4TB) scored highly without confirmed TB had identifiable
  non-TB abnormalities (pleural disease, cardiomegaly, non-TB pneumonia,
  nodules). This corroborates the *phenomenon* behind this document's
  negative-class-composition finding (non-TB abnormal cases drawing high TB
  scores) in an independent, real-world, non-training-focused study — but
  the paper doesn't investigate or attribute a training-data-composition
  cause, so it supports the phenomenon, not this document's specific causal
  claim (that Shenzhen/Montgomery's near-exclusively-healthy training
  negatives are why); that causal claim rests on this document's own
  subgroup-split experiment (`eval/negative_composition_probe.py`,
  Robustness interventions section) and would need its own literature check
  before treating it as itself prior art.
- **Riley, Debray, Collins, Archer, Ensor, van Smeden & Snell**,
  "Minimum sample size for external validation of a clinical prediction
  model with a binary outcome," *Statistics in Medicine* 40(19), 2021
  ([DOI: 10.1002/sim.9025](https://onlinelibrary.wiley.com/doi/10.1002/sim.9025))
  — the standard methodological reference for exactly the question this
  document's calibration-set-size sweep asks empirically: how many
  external-validation cases are needed to precisely estimate calibration.
  Their finding that the calibration-slope criterion typically drives the
  largest sample-size requirement (far more than a bare "100 events" rule
  of thumb) is the same qualitative conclusion this document reaches by
  resampling — that small calibration sets are unbiased on average but too
  imprecise to trust — reached instead via closed-form/simulation sample-
  size formulas rather than bootstrap resampling of an existing model's
  scores. Not TB- or CXR-specific and not previously connected to this
  project's calibration-set-size result; worth citing as the general
  statistical grounding for that section, not discovered independently by
  it — the empirical result here should be read as a case study consistent
  with, not a replacement for, this more general methodology.
- One more lead surfaced but **not verified**: a medRxiv preprint titled
  "Geographic Domain Shift Precipitates Divergent Failure Modes in Deep
  Learning–Based Tuberculosis Screening: A Multi-National External
  Validation Study" looks directly on-point (TB screening + cross-site
  failure modes) but its PDF wouldn't parse through the fetch tool used
  here — worth tracking down properly before citing, not worth citing on a
  title alone.

**What doesn't appear to have direct precedent** (in what this check
found — a real literature review before submission would need to confirm
this more thoroughly than a handful of searches can): a direct causal
ablation that trains and scores a model with the *entire* lung field
removed and reports the surviving AUC as a number, cross-checked with a
mask-dilation robustness control, a region-by-region decomposition
(shoulders/spine/diaphragm/corners via actual organ segmentation), a
fixed-location patch test, a high-pass residual test, and a label-shuffle
negative control, specifically on Shenzhen/Montgomery/TBX11K. FairALM
addresses the confound by training around it without quantifying its raw
magnitude; DeGrave's saliency-based method is indirect (saliency maps have
well-documented reliability problems of their own) rather than a direct
"how much survives with the diagnostic region completely gone" measurement.

**Honest positioning:** known confound, quantified more precisely and with
a more direct, convergent evidence chain than the prior work found here —
not a novel discovery. Write it up as replication-plus-quantification-
plus-fix, cite FairALM and DeGrave as the closest prior art, and track down
the medRxiv lead properly before final submission.

---

## Phase 3 extension point

When adding the reconstruction head, modify `TBDiagnosticModel` to:
1. Accept a `ReconHead` alongside `DiagnosticHead`
2. Pass the **full feature list** from `SharedEncoder` to `ReconHead`
3. Keep `DiagnosticHead` consuming only `features[-1]` — diagnosis path is unchanged
