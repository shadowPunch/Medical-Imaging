import torch

from models.tb_model import build_model
from recon.train_recon import setup_trainable


def _model():
    return build_model("efficientnet_b0", pretrained=False, with_recon=True, volume_size=16)


def test_joint_mode_trains_encoder_and_recon_head():
    model = _model()
    params = setup_trainable(model, freeze_encoder=False)
    ids = {id(p) for p in params}
    assert any(id(p) in ids for p in model.encoder.parameters())
    assert any(id(p) in ids for p in model.recon_head.parameters())
    assert all(p.requires_grad for p in model.encoder.parameters())


def test_frozen_mode_excludes_every_encoder_parameter():
    model = _model()
    params = setup_trainable(model, freeze_encoder=True)
    ids = {id(p) for p in params}
    assert not any(id(p) in ids for p in model.encoder.parameters())
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(id(p) in ids for p in model.recon_head.parameters())


def test_frozen_encoder_weights_do_not_move_under_a_backward_pass():
    model = _model()
    params = setup_trainable(model, freeze_encoder=True)
    before = next(model.encoder.parameters()).detach().clone()
    opt = torch.optim.AdamW(params, lr=1e-2)
    model.forward_recon(torch.randn(1, 3, 64, 64)).sum().backward()
    opt.step()
    assert torch.equal(next(model.encoder.parameters()), before)


def test_frozen_encoder_stays_in_eval_mode_after_model_train():
    """BatchNorm running stats would otherwise drift on DRR inputs even with
    the weights frozen, which changes Head A's behaviour just as much."""
    model = _model()
    setup_trainable(model, freeze_encoder=True)
    model.train()
    assert not model.encoder.training
    assert model.recon_head.training
