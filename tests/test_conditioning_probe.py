import numpy as np

from recon.conditioning_probe import cross_patient_correlation


def test_identical_patients_correlate_at_one():
    means = np.tile(np.linspace(0, 1, 50), (4, 1))
    assert cross_patient_correlation(means) > 0.99


def test_unrelated_patients_correlate_near_zero():
    rng = np.random.default_rng(0)
    assert abs(cross_patient_correlation(rng.normal(size=(8, 500)))) < 0.2
