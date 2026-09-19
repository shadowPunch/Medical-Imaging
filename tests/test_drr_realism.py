import pytest
import torch

from recon.drr_realism import crop_to_body, realistic_drr


def _synthetic_drr(batch: int = 2, size: int = 64) -> torch.Tensor:
    """Small bright 'body' block on a black background, like a normalized DRR."""
    x = torch.zeros(batch, 1, size, size)
    x[:, :, 24:40, 22:42] = torch.linspace(0.3, 0.9, 20).view(1, 1, 1, 20)
    return x


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_off_is_identity():
    x = _synthetic_drr()
    assert torch.equal(realistic_drr(x, "off"), x)


@pytest.mark.parametrize("strength", ["mild", "strong"])
def test_preserves_shape_and_range(strength):
    x = _synthetic_drr()
    y = realistic_drr(x, strength, generator=_gen(0))
    assert y.shape == x.shape
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_same_seed_same_output_different_seed_different_output():
    x = _synthetic_drr()
    a = realistic_drr(x, "mild", generator=_gen(1))
    b = realistic_drr(x, "mild", generator=_gen(1))
    c = realistic_drr(x, "mild", generator=_gen(2))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_samples_in_batch_get_independent_randomization():
    x = _synthetic_drr(batch=2)
    y = realistic_drr(x, "mild", generator=_gen(3))
    assert not torch.equal(y[0], y[1])


def test_prob_zero_is_identity():
    x = _synthetic_drr()
    assert torch.equal(realistic_drr(x, "mild", generator=_gen(0), prob=0.0), x)


def test_partial_prob_leaves_some_samples_clean_and_changes_others():
    x = _synthetic_drr(batch=32)
    y = realistic_drr(x, "mild", generator=_gen(5), prob=0.5)
    unchanged = sum(torch.equal(x[i], y[i]) for i in range(32))
    assert 0 < unchanged < 32


def test_crop_to_body_makes_body_fill_more_of_the_frame():
    x = _synthetic_drr(batch=1)
    before = (x > 0.05).float().mean()
    y = crop_to_body(x, margin=torch.tensor([0.05]))
    after = (y > 0.05).float().mean()
    assert after > 2 * before


def test_negative_margin_crops_inside_body_so_frame_is_filled():
    # Real films have anatomy running off the frame edge; a negative margin
    # crops inside the CT slab to remove its hard truncated outline.
    x = torch.zeros(1, 1, 64, 64)
    x[:, :, 12:52, 12:52] = 0.6
    y = crop_to_body(x, margin=torch.tensor([-0.1]))
    assert (y > 0.05).float().mean() > 0.99


def test_crop_to_body_leaves_empty_image_unchanged():
    x = torch.zeros(1, 1, 32, 32)
    assert torch.equal(crop_to_body(x, margin=torch.tensor([0.1])), x)


def test_realism_brightens_mid_tones_toward_real_film_statistics():
    # Real CXRs measured at mean 0.54-0.67 vs 0.23 for a raw DRR; the tone
    # curve plus body crop should move the mean up, on average across draws.
    x = _synthetic_drr(batch=16)
    y = realistic_drr(x, "mild", generator=_gen(4))
    assert y.mean() > x.mean()
