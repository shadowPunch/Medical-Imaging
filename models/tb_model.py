import torch
import torch.nn as nn

from .encoder import SharedEncoder, XRVDenseNetEncoder
from .diagnostic_head import DiagnosticHead


class TBDiagnosticModel(nn.Module):
    """Shared encoder + diagnostic head. The reconstruction head slots in here in Phase 3."""

    def __init__(self, encoder: SharedEncoder, head: DiagnosticHead):
        super().__init__()
        self.encoder = encoder
        self.head    = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))  # (B, 1) logit


def build_model(
    backbone:   str  = "efficientnet_b0",
    pretrained: bool = True,
    dropout:    float = 0.3,
) -> TBDiagnosticModel:
    if backbone.startswith("xrv_"):
        # e.g. "xrv_densenet121-res224-all" -> torchxrayvision weights id "densenet121-res224-all"
        encoder: nn.Module = XRVDenseNetEncoder(weights=backbone[len("xrv_"):])
    else:
        encoder = SharedEncoder(backbone, pretrained)
    head = DiagnosticHead(encoder.out_channels, dropout)
    return TBDiagnosticModel(encoder, head)
