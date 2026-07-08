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
