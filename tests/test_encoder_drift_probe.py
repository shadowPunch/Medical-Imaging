import pytest
import torch

from eval.encoder_drift_probe import graft_encoder


def _sd(prefix_val: float, head_val: float) -> dict:
    return {"encoder.blocks.0.weight": torch.full((2, 2), prefix_val),
            "diagnostic_head.fc.weight": torch.full((1, 2), head_val)}


def test_graft_takes_encoder_from_donor_and_head_from_base():
    out = graft_encoder(_sd(1.0, 5.0), _sd(2.0, 9.0))
    assert torch.equal(out["encoder.blocks.0.weight"], torch.full((2, 2), 2.0))
    assert torch.equal(out["diagnostic_head.fc.weight"], torch.full((1, 2), 5.0))


def test_graft_does_not_mutate_the_base_state_dict():
    base = _sd(1.0, 5.0)
    graft_encoder(base, _sd(2.0, 9.0))
    assert torch.equal(base["encoder.blocks.0.weight"], torch.full((2, 2), 1.0))


def test_graft_rejects_a_donor_with_no_matching_encoder_keys():
    with pytest.raises(ValueError, match="no encoder"):
        graft_encoder(_sd(1.0, 5.0), {"recon_head.to_density.bias": torch.zeros(1)})


def test_graft_rejects_a_shape_mismatch_rather_than_loading_silently():
    donor = {"encoder.blocks.0.weight": torch.zeros(3, 3)}
    with pytest.raises(ValueError, match="shape"):
        graft_encoder(_sd(1.0, 5.0), donor)


def test_linear_head_separates_separable_features():
    import numpy as np

    from eval.encoder_drift_probe import fit_linear_head
    from eval.metrics import compute_auc

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (100, 8)), rng.normal(4, 1, (100, 8))])
    y = np.array([0] * 100 + [1] * 100)
    assert compute_auc(y, fit_linear_head(X, y, X)) > 0.99
