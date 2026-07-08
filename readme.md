# TB CXR Diagnostic — Phase 1 & 2

Shared encoder + diagnostic head for TB detection from chest X-rays.
Targets the WHO TPP triage specification (sensitivity ≥ 90%, specificity ≥ 70%).

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
│   └── metrics.py        — AUC, WHO TPP operating point selection
└── utils/
    └── gradcam.py        — Grad-CAM + heatmap overlay
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

`--held-out none` trains on all sources with no held-out evaluation.

---

## Expected outputs

```
outputs/diagnostic/
└── best_model.pt    — checkpoint at best val AUC (model weights + epoch + metrics)
```

Console reports AUC and WHO TPP operating point after each epoch, and a final
held-out source evaluation showing the cross-source generalisation gap.

---

## Phase 3 extension point

When adding the reconstruction head, modify `TBDiagnosticModel` to:
1. Accept a `ReconHead` alongside `DiagnosticHead`
2. Pass the **full feature list** from `SharedEncoder` to `ReconHead`
3. Keep `DiagnosticHead` consuming only `features[-1]` — diagnosis path is unchanged
