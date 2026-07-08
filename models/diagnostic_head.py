import torch
import torch.nn as nn


class DiagnosticHead(nn.Module):
    """
    Head A: binary TB classifier on top of the encoder's deepest feature map.

    Output is a raw logit — use BCEWithLogitsLoss during training,
    torch.sigmoid() at inference to get a probability in [0, 1].
    The pre-GAP feature map (features[-1]) is the natural Grad-CAM target.
    """

    def __init__(self, in_channels: int, dropout: float = 0.3):
        super().__init__()
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(in_channels, 1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features[-1]               # (B, C, H, W)
        x = self.gap(x).flatten(1)    # (B, C)
        return self.fc(self.drop(x))  # (B, 1) logit
