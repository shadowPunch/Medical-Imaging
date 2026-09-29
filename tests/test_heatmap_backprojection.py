import numpy as np
import torch

from recon.heatmap import attention_volume, backproject, infer_projection_axis


def _hot_corner(size: int = 16) -> torch.Tensor:
    """2D attention map with one bright quadrant."""
    hm = torch.zeros(size, size)
    hm[:size // 2, :size // 2] = 1.0
    return hm


def test_backprojection_fills_the_volume_along_the_projection_axis():
    vol = backproject(_hot_corner(), volume_size=8, axis=0)
    assert vol.shape == (8, 8, 8)
    # every slice along the projection axis is the same map: a ray carries no depth info
    assert torch.allclose(vol[0], vol[-1])


def test_backprojection_preserves_where_the_attention_is():
    vol = backproject(_hot_corner(), volume_size=8, axis=0)
    hot = vol[:, :4, :4].mean()
    cold = vol[:, 4:, 4:].mean()
    assert hot > 0.9 and cold < 0.1


def test_backprojection_along_each_axis_is_constant_on_that_axis():
    for axis in (0, 1, 2):
        vol = backproject(_hot_corner(), volume_size=8, axis=axis)
        assert vol.shape == (8, 8, 8)
        spread = vol.std(dim=axis).max()
        assert spread < 1e-6, f"axis {axis} varies along the projection direction"


def test_attention_is_suppressed_where_the_volume_predicts_air():
    """Attention floating in air is meaningless — it should be masked by predicted
    tissue, which is the point of showing it on the volume at all."""
    density = torch.zeros(8, 8, 8)
    density[:, :4, :4] = 1.0  # tissue only under the hot quadrant
    vol = attention_volume(_hot_corner(), density, axis=0)
    assert vol[:, :4, :4].mean() > 0.5
    assert vol[:, 4:, 4:].max() < 1e-6


def test_attention_volume_is_normalized_and_finite():
    density = torch.rand(8, 8, 8)
    vol = attention_volume(_hot_corner(), density, axis=0)
    assert torch.isfinite(vol).all()
    assert vol.min() >= 0.0 and vol.max() <= 1.0


def test_projection_axis_is_measured_not_assumed():
    """The predicted volume's axis order is a property of the training target, so
    the projection axis is found by matching against the DRR rather than guessed."""
    rng = np.random.default_rng(0)
    vol = torch.from_numpy(rng.random((8, 8, 8))).float()
    for axis in (0, 1, 2):
        drr = vol.mean(dim=axis)
        assert infer_projection_axis(vol, drr)[0] == axis


def test_choose_axis_prefers_the_geometry_default_when_image_evidence_is_weak():
    from recon.heatmap import choose_axis

    axis, warning = choose_axis(default=2, inferred=0, correlation=0.05)
    assert axis == 2 and "weak" in warning.lower()


def test_choose_axis_reports_disagreement_with_strong_evidence():
    from recon.heatmap import choose_axis

    axis, warning = choose_axis(default=2, inferred=1, correlation=0.8)
    assert axis == 2 and "disagree" in warning.lower()


def test_choose_axis_is_quiet_when_evidence_agrees():
    from recon.heatmap import choose_axis

    axis, warning = choose_axis(default=2, inferred=2, correlation=0.6)
    assert axis == 2 and warning == ""


def test_export_tag_states_the_limitation_not_just_the_word_synthesized():
    from recon.export import SYNTHESIZED_TAG

    tag = SYNTHESIZED_TAG.lower()
    assert "not for measurement" in tag
    assert "unvalidated" in tag        # says what is not established
    assert "single radiograph" in tag  # says where the volume came from
