"""Loading a Phase 3 checkpoint into the right architecture.

Whether a checkpoint's ReconHead has the features[-2] skip connection is
recorded only by the presence of its parameters, so it is detected here rather
than passed around as a flag every eval script would have to thread through
(and get wrong for older checkpoints).
"""
import torch

from models.tb_model import build_model

SKIP_PREFIX = "recon_head.skip_"


def has_skip(state_dict: dict) -> bool:
    return any(k.startswith(SKIP_PREFIX) for k in state_dict)


def load_recon_model(checkpoint: str, volume_size: int, device: torch.device,
                     backbone: str = "efficientnet_b0"):
    sd = torch.load(checkpoint, map_location=device, weights_only=False)["model"]
    model = build_model(backbone, pretrained=False, with_recon=True,
                        volume_size=volume_size, recon_skip=has_skip(sd)).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model
