import torch
import torch.nn as nn

from .diagnostic_head import DiagnosticHead
from .encoder import SharedEncoder, XRVDenseNetEncoder
from .recon_head import ReconHead


class TBDiagnosticModel(nn.Module):
    """
    Shared encoder + diagnostic head, with an optional reconstruction head
    (Phase 3, Head B) sharing the same encoder.

    Isolation guarantee (TB_CXR_Diagnostic_3D_Proposal.md §11a): Head B's
    output must never enter any path that affects Head A's probability or
    threshold. Enforced structurally here, not just by convention —
    `forward()` (the diagnosis path every Phase 1/2 script calls) only ever
    touches `self.encoder` and `self.head`; it has no reference to
    `self.recon_head` at all. Reconstruction runs only through the separate
    `forward_recon()` method, which Phase 1/2 code never calls. See
    recon/firewall_test.py for the automated check of this property.
    """

    def __init__(self, encoder: SharedEncoder, head: DiagnosticHead,
                recon_head: ReconHead | None = None):
        super().__init__()
        self.encoder    = encoder
        self.head       = head
        self.recon_head = recon_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))  # (B, 1) logit — diagnosis path, unchanged by Head B

    def forward_recon(self, x: torch.Tensor) -> torch.Tensor:
        """Head B only. Raises if this model was built without a recon_head."""
        if self.recon_head is None:
            raise RuntimeError("This model has no recon_head — build with build_model(with_recon=True)")
        return self.recon_head(self.encoder(x))  # (B, 1, D, D, D) density volume


def build_model(
    backbone:    str  = "efficientnet_b0",
    pretrained:  bool = True,
    dropout:     float = 0.3,
    with_recon:  bool = False,
    volume_size: int = 128,
    recon_skip:  bool = False,
) -> TBDiagnosticModel:
    if backbone.startswith("xrv_"):
        # e.g. "xrv_densenet121-res224-all" -> torchxrayvision weights id "densenet121-res224-all"
        encoder: nn.Module = XRVDenseNetEncoder(weights=backbone[len("xrv_"):])
    else:
        encoder = SharedEncoder(backbone, pretrained)
    head = DiagnosticHead(encoder.out_channels, dropout)
    recon_head = ReconHead(encoder.out_channels, volume_size=volume_size,
                           use_skip=recon_skip,
                           skip_channels=getattr(encoder, "skip_channels", None)) if with_recon else None
    return TBDiagnosticModel(encoder, head, recon_head)
