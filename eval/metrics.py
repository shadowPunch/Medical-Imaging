import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob))


def _sens_spec_curve(y_true, y_prob):
    """Returns parallel arrays: (thresholds, sensitivity, specificity)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return thresholds, tpr, 1.0 - fpr


def find_who_tpp_point(y_true: np.ndarray, y_prob: np.ndarray) -> dict | None:
    """
    WHO TPP triage bar: sensitivity ≥ 90%.
    Among all thresholds meeting that floor, picks the one with highest specificity.
    Returns None if the model never reaches 90% sensitivity.
    """
    thresholds, sens, spec = _sens_spec_curve(y_true, y_prob)
    eligible = np.where(sens >= 0.90)[0]
    if len(eligible) == 0:
        return None

    best = eligible[np.argmax(spec[eligible])]
    return {
        "threshold":         float(thresholds[best]),
        "sensitivity":       float(sens[best]),
        "specificity":       float(spec[best]),
        # WHO minimum: spec ≥ 70%; optimal: spec ≥ 80%
        "meets_minimum_tpp": float(spec[best]) >= 0.70,
        "meets_optimal_tpp": float(spec[best]) >= 0.80,
    }


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "auc":     compute_auc(y_true, y_prob),
        "who_tpp": find_who_tpp_point(y_true, y_prob),
    }


def print_report(metrics: dict) -> None:
    print(f"  AUC : {metrics['auc']:.4f}")
    t = metrics["who_tpp"]
    if t is None:
        print("  WHO TPP : sensitivity floor (90%) not reached")
        return
    min_ok = "✓" if t["meets_minimum_tpp"] else "✗"
    opt_ok = "✓" if t["meets_optimal_tpp"] else "✗"
    print(
        f"  WHO TPP (sens≥90%) : thr={t['threshold']:.3f}  "
        f"sens={t['sensitivity']:.3f}  spec={t['specificity']:.3f}  "
        f"minimum(spec≥70%)={min_ok}  optimal(spec≥80%)={opt_ok}"
    )


def apply_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """
    Sensitivity/specificity at a *given* threshold — does not re-derive the
    operating point from (y_true, y_prob). Use this to score a held-out set
    against a threshold chosen on a separate validation set; scoring and
    threshold-selection must never share data or the reported numbers leak.
    """
    pred = y_prob >= threshold
    tp = int(np.sum(pred & (y_true == 1)))
    fn = int(np.sum(~pred & (y_true == 1)))
    tn = int(np.sum(~pred & (y_true == 0)))
    fp = int(np.sum(pred & (y_true == 0)))
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "threshold":         float(threshold),
        "sensitivity":       float(sensitivity),
        "specificity":       float(specificity),
        "meets_minimum_tpp": specificity >= 0.70,
        "meets_optimal_tpp": specificity >= 0.80,
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Percentile bootstrap 95% CI (default alpha=0.05) for AUC, sensitivity, and
    specificity at a fixed threshold. Resamples (y_true, y_prob) pairs with
    replacement — the right way to report a metric computed on a small
    held-out set (e.g. Montgomery, ~138 images) instead of a bare point
    estimate that hides how little the sample size can actually resolve.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs, sens, spec = [], [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue  # AUC undefined for a resample with only one class
        aucs.append(compute_auc(yt, yp))
        point = apply_threshold(yt, yp, threshold)
        sens.append(point["sensitivity"])
        spec.append(point["specificity"])

    def _ci(values: list[float]) -> tuple[float, float]:
        lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(lo), float(hi)

    return {
        "n_boot_used":    len(aucs),
        "auc_ci":         _ci(aucs),
        "sensitivity_ci": _ci(sens),
        "specificity_ci": _ci(spec),
    }


def sens_at_spec(y_true: np.ndarray, y_prob: np.ndarray, target_spec: float = 0.70) -> dict:
    """
    Best achievable sensitivity while keeping specificity >= target_spec, read
    directly off the ROC curve on (y_true, y_prob). This is a *ceiling* under
    an unconstrained, per-set operating point — the ranking-quality question
    ("can this encoder discriminate at all here") decoupled from whether any
    single frozen threshold happens to transfer. Complements apply_threshold,
    which answers the opposite question: how a specific externally-chosen
    threshold performs. A low ceiling means discrimination itself is the
    problem; a high ceiling with a low apply_threshold score means the
    problem is calibration/threshold transfer, not discrimination.
    """
    thresholds, sens, spec = _sens_spec_curve(y_true, y_prob)
    eligible = np.where(spec >= target_spec)[0]
    if len(eligible) == 0:
        return {"threshold": None, "sensitivity": 0.0, "specificity": None, "target_spec": target_spec}
    best = eligible[np.argmax(sens[eligible])]
    return {
        "threshold":   float(thresholds[best]),
        "sensitivity": float(sens[best]),
        "specificity": float(spec[best]),
        "target_spec": target_spec,
    }


def bootstrap_sens_at_spec_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_spec: float = 0.70,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap 95% CI for sens_at_spec's sensitivity ceiling."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    sens_vals = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        point = sens_at_spec(yt, yp, target_spec)
        if point["threshold"] is not None:
            sens_vals.append(point["sensitivity"])

    lo, hi = np.percentile(sens_vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"n_boot_used": len(sens_vals), "sensitivity_ci": (float(lo), float(hi))}


def print_frozen_report(point: dict, ci: dict | None = None) -> None:
    min_ok = "✓" if point["meets_minimum_tpp"] else "✗"
    opt_ok = "✓" if point["meets_optimal_tpp"] else "✗"
    print(
        f"  Frozen threshold={point['threshold']:.3f}  "
        f"sens={point['sensitivity']:.3f}  spec={point['specificity']:.3f}  "
        f"minimum(spec≥70%)={min_ok}  optimal(spec≥80%)={opt_ok}"
    )
    if ci is not None:
        auc_lo, auc_hi = ci["auc_ci"]
        sens_lo, sens_hi = ci["sensitivity_ci"]
        spec_lo, spec_hi = ci["specificity_ci"]
        print(
            f"  95% CI (n_boot={ci['n_boot_used']})  "
            f"AUC=[{auc_lo:.3f}, {auc_hi:.3f}]  "
            f"sens=[{sens_lo:.3f}, {sens_hi:.3f}]  "
            f"spec=[{spec_lo:.3f}, {spec_hi:.3f}]"
        )
