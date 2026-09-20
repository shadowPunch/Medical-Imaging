from models.tb_model import build_model
from recon.checkpoint import has_skip


def _sd(recon_skip: bool) -> dict:
    return build_model("efficientnet_b0", pretrained=False, with_recon=True,
                       volume_size=16, recon_skip=recon_skip).state_dict()


def test_detects_a_skip_checkpoint():
    assert has_skip(_sd(True))


def test_detects_a_pre_skip_checkpoint():
    assert not has_skip(_sd(False))
