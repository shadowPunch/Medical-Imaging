import torch
import torch.nn as nn

from .encoder import SharedEncoder
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
    encoder = SharedEncoder(backbone, pretrained)
    head    = DiagnosticHead(encoder.out_channels, dropout)
    return TBDiagnosticModel(encoder, head)
