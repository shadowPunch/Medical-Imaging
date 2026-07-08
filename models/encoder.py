import timm
import torch
import torch.nn as nn


class SharedEncoder(nn.Module):
    """
    Multi-scale 2D backbone shared between the diagnostic head (Phase 1)
    and the reconstruction head (Phase 3).

    Uses timm's features_only mode to expose feature maps at 5 scales.
    Head A consumes features[-1] (deepest, most semantic).
    Head B (future) will consume all scales for back-projection.
    """

    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        self.net = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        # e.g. EfficientNet-B0: channels = [16, 24, 40, 112, 320]
        self._channels: list[int] = self.net.feature_info.channels()

    @property
    def out_channels(self) -> int:
        """Channels of the deepest feature map — what Head A connects to."""
        return self._channels[-1]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns a list of 5 feature maps, coarse-to-fine in channel depth."""
        return self.net(x)
