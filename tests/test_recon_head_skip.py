import torch

from models.recon_head import ReconHead

DEEP_CH, SKIP_CH = 320, 112


def _features(seed: int = 0, skip_seed: int | None = None):
    """EfficientNet-B0-shaped feature list for a 320px input."""
    g = torch.Generator().manual_seed(seed)
    gs = torch.Generator().manual_seed(seed if skip_seed is None else skip_seed)
    return [torch.randn(1, 16, 160, 160, generator=g),
            torch.randn(1, 24, 80, 80, generator=g),
            torch.randn(1, 40, 40, 40, generator=g),
            torch.randn(1, SKIP_CH, 20, 20, generator=gs),
            torch.randn(1, DEEP_CH, 10, 10, generator=g)]


def _head(use_skip: bool) -> ReconHead:
    torch.manual_seed(0)
    return ReconHead(DEEP_CH, volume_size=32, use_skip=use_skip, skip_channels=SKIP_CH).eval()


def test_output_shape_and_non_negative_with_skip():
    out = _head(True)(_features())
    assert out.shape == (1, 1, 32, 32, 32)
    assert (out >= 0).all()


def test_without_skip_the_finer_feature_map_is_ignored():
    """Documents the bottleneck the skip connection exists to remove."""
    head = _head(False)
    with torch.no_grad():
        a = head(_features(skip_seed=1))
        b = head(_features(skip_seed=2))
    assert torch.equal(a, b)


def test_with_skip_the_finer_feature_map_changes_the_volume():
    head = _head(True)
    with torch.no_grad():
        a = head(_features(skip_seed=1))
        b = head(_features(skip_seed=2))
    assert not torch.equal(a, b)


def test_skip_parameters_are_additive_so_old_checkpoints_still_load():
    plain, skipped = _head(False).state_dict(), _head(True).state_dict()
    assert set(plain).issubset(set(skipped))
    missing, unexpected = _head(False).load_state_dict(plain, strict=True)
    assert not missing and not unexpected
