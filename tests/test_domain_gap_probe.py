import numpy as np

from recon.domain_gap_probe import count_outliers, separation_ratio


def _two_domains(seed: int = 0, shift: float = 2.0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (300, 16)), rng.normal(shift, 1, (300, 16))


def test_separation_ratio_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    assert separation_ratio(rng.normal(0, 1, (300, 16)), rng.normal(0, 1, (300, 16))) < 0.1


def test_separation_ratio_not_faked_small_by_a_few_exploding_samples():
    a, b = _two_domains()
    clean = separation_ratio(a, b)
    b[:3] *= 1000  # 1% of inputs blow up, as seen on a realism-only checkpoint
    assert separation_ratio(a, b) > 0.8 * clean


def test_count_outliers_flags_only_far_volumes():
    ct = {"mean": {"mean": 0.1, "std": 0.02}}
    rows = [[0.5, 0.1, 0, 0, 0], [0.5, 0.15, 0, 0, 0], [0.5, 52.0, 0, 0, 0]]
    assert count_outliers(rows, ct) == 1
