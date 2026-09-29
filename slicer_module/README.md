# TB Reconstruction — 3D Slicer module

Loads the outputs of `recon/export_for_slicer.py` into 3D Slicer: the
synthesized thoracic volume, Head A's back-projected localization, the TB
probability, and a threshold slider over the attention.

## Install

Slicer → Edit → Application Settings → Modules → **Additional module paths** →
add this directory → restart Slicer. The module appears under **Chest Imaging →
TB CXR 3D Reconstruction**.

## Use

1. Produce a case:

   ```
   python -m recon.export_for_slicer \
       --diagnostic-checkpoint outputs/step3_mild_lungcrop_tbx11k/best_model.pt \
       --recon-checkpoint outputs/phase3_recon_run7_frozen/latest.pt \
       --image <lung-cropped CXR>.png --out-dir outputs/<case>
   ```

2. In Slicer, point the module at `outputs/<case>` and press **Load case**.

## Design

Thin by intent (proposal §7). Inference runs outside Slicer, so Slicer's Python
needs no torch and this module only loads, displays and thresholds. Everything
that is not UI lives in `tb_recon_logic.py`, which imports nothing from Slicer
and is covered by `tests/test_slicer_logic.py` in the main test run.

The banner and the per-file `SYNTHESIZED` tag both state the limitation, so it
survives a UI that forgets to show it and a file that outlives this module.

The threshold slider works on the **continuous** attention volume
(`attention.nrrd`); `attention_mask.nrrd` is a pre-thresholded convenience,
since re-thresholding a binary mask cannot recover what binarization discarded.
