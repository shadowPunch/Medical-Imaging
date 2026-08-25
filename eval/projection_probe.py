"""
Projection probe: how much of a checkpoint's in-domain AUC rides on
source-correlated features, rather than TB-specific ones?

Extracts the encoder's penultimate (post-GAP) features for the pooled
train/val split a checkpoint was trained on, fits a linear source-classifier
probe on frozen train features, and measures in-domain val AUC for a TB-label
linear probe before vs. after projecting the source direction out of the
features. A large AUC drop after projection means the original in-domain AUC
was substantially riding on source-identity rather than TB-specific signal.
No retraining of the encoder — everything after feature extraction is a fast
sklearn linear fit. See code/readme.md's Phase 2 validation results.

Example:
    python -m eval.projection_probe --checkpoint outputs/diagnostic_montgomery/best_model.pt \
        --shenzhen ../datasets/tb-shenzen --montgomery ../datasets/tb-montgomery --tbx11k ../datasets/tbx11k \
        --held-out montgomery
"""
import argparse

import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import DataConfig, TrainConfig
from data.transforms import get_val_transforms
from models.tb_model import build_model
from train_diagnostic import build_splits, make_loader


def _source_of(path) -> str:
    name = str(path)
    if "CHNCXR" in name:
        return "shenzhen"
    if "MCUCXR" in name:
        return "montgomery"
    return "tbx11k"


def _extract_features(model, samples, image_size, batch_size, device):
    """Returns (pooled_features [N, C], tb_labels [N], source_labels [N])."""
    loader, _ = make_loader(samples, get_val_transforms(image_size), TrainConfig(batch_size=batch_size))
    sources = [_source_of(p) for p, _ in samples]

    feats, labels = [], []
    with torch.no_grad():
        for imgs, y in loader:
            with torch.amp.autocast(device.type):
                f = model.encoder(imgs.to(device))[-1]
                pooled = F.adaptive_avg_pool2d(f, 1).flatten(1)
            feats.append(pooled.float().cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels), np.array(sources)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--shenzhen",   type=str, default=None)
    parser.add_argument("--montgomery", type=str, default=None)
    parser.add_argument("--tbx11k",     type=str, default=None)
    parser.add_argument("--held-out",   type=str, required=True,
                        choices=["shenzhen", "montgomery", "tbx11k"])
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    train_s, val_s, _ = build_splits(args, DataConfig(), variant="")
    print(f"train: {len(train_s)}  val: {len(val_s)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("efficientnet_b0", pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}  (val AUC at train time={ckpt['metrics']['auc']:.4f})")

    Xtr, ytr, str_ = _extract_features(model, train_s, args.image_size, args.batch_size, device)
    Xval, yval, sval = _extract_features(model, val_s, args.image_size, args.batch_size, device)

    sources_present = sorted(set(str_) | set(sval))
    if len(sources_present) < 2:
        raise ValueError(f"Only one source ({sources_present}) in this split — "
                          "the source-direction probe needs at least two.")
    print(f"Sources in this split: {sources_present}")

    # Standardize features (logistic regression + projection are scale-sensitive)
    mu, sigma = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    Xtr_n, Xval_n = (Xtr - mu) / sigma, (Xval - mu) / sigma

    # --- Source-classifier direction, fit on train features ---
    src_bin_tr = (str_ == sources_present[0]).astype(int)
    src_clf = LogisticRegression(max_iter=2000).fit(Xtr_n, src_bin_tr)
    src_acc = src_clf.score((Xval_n), (sval == sources_present[0]).astype(int))
    w_src = src_clf.coef_[0]
    w_src_hat = w_src / (np.linalg.norm(w_src) + 1e-8)
    print(f"Source-direction probe accuracy (val, {sources_present[0]} vs rest): {src_acc:.4f}")

    # --- TB-label probe, before projection ---
    tb_clf_before = LogisticRegression(max_iter=2000).fit(Xtr_n, ytr)
    auc_before = roc_auc_score(yval, tb_clf_before.predict_proba(Xval_n)[:, 1])

    # --- Project the source direction out of both splits, refit TB-label probe ---
    def project_out(X, direction):
        return X - np.outer(X @ direction, direction)

    Xtr_proj = project_out(Xtr_n, w_src_hat)
    Xval_proj = project_out(Xval_n, w_src_hat)
    tb_clf_after = LogisticRegression(max_iter=2000).fit(Xtr_proj, ytr)
    auc_after = roc_auc_score(yval, tb_clf_after.predict_proba(Xval_proj)[:, 1])

    print(f"\nIn-domain val AUC (linear probe on frozen features):")
    print(f"  before projecting out source direction: {auc_before:.4f}")
    print(f"  after  projecting out source direction: {auc_after:.4f}")
    print(f"  drop: {auc_before - auc_after:+.4f}")
    if auc_before - auc_after > 0.05:
        print("  -> Substantial drop: a meaningful share of the original in-domain AUC "
              "was riding on source-correlated features, not TB-specific ones.")
    else:
        print("  -> Small drop: the in-domain AUC does not appear to depend much on "
              "the linear source direction in these features.")


if __name__ == "__main__":
    main()
