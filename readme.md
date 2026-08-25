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
  clearing the WHO TPP bar for the first time, with a tight, non-overlapping
  CI against every earlier configuration. An augmentation-strength sweep
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

**Phase 3 — reconstruction head: not started; being scoped now.** DiffDRR
pair generation + a back-projection reconstruction head + shape induction
need a real CT dataset and cloud GPU (3D training exceeds the local 4GB
card), neither available in this environment. Plan is to build and
locally-smoke-test the pipeline (architecture, DiffDRR integration, training
script, NRRD/NIfTI export, and the code-level guarantee from the proposal's
§11a that Head B never feeds Head A's decision) and hand off actual training
to a Colab notebook, mirroring `code/tb_diagnostic_colab.ipynb`'s existing
pattern for Phase 1/2. No trained reconstruction exists yet.

**Nothing is committed to git.** `code/` and `datasets/` both have
substantial uncommitted changes from this session — ask before committing.

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
│   ├── projection_probe.py   — Does in-domain AUC ride on a linear source direction?
│   ├── histogram_probe.py    — Spatially-blind intensity-histogram probe
│   ├── shortcut_baseline.py  — Source-classifier shortcut-detector baseline
│   └── complement_monitor.py — Standing in-domain confound diagnostic for any intervention
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
TBX11K — the other cross-source direction) or `none` (train on all sources,
no held-out evaluation).

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
lung removal, plus the region breakdown), not as a novel discovery, pending
an actual literature check before any writeup.

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
| Lung-crop + texture-aug, **mild** | **92.6% [90.3, 94.6]** | **0.893** |
| Lung-crop + texture-aug, medium | 89.5% [86.8, 91.8] | not tested |
| Lung-crop + texture-aug, aggressive | 88.3% [85.6, 90.9] | not tested |

**Clean lung-crop alone is a real, substantial win on TBX11K** (78.5% →
84.4%) — confirming the earlier CLAHE result wasn't just "crop doesn't help
here," it was specifically CLAHE actively hurting. **Mild texture
augmentation on top pushes the ceiling to 92.6% [90.3, 94.6] — the entire CI
clears the 90% WHO-TPP sensitivity target for the first time this session,**
with zero overlap against the clean baseline's CI. This is not Montgomery-
sized noise; at this N the separation is real.

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
check already in place to verify it isn't just satisfied superficially. Given
Steps 1-4 already found a configuration that clears the 90% ceiling with
tight CIs, DANN's expected marginal value from here is low; worth treating as
optional rather than required to complete this intervention plan.

---

## Phase 3 extension point

When adding the reconstruction head, modify `TBDiagnosticModel` to:
1. Accept a `ReconHead` alongside `DiagnosticHead`
2. Pass the **full feature list** from `SharedEncoder` to `ReconHead`
3. Keep `DiagnosticHead` consuming only `features[-1]` — diagnosis path is unchanged
