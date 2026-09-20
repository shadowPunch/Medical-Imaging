import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    @property
    def skip_channels(self) -> int | None:
        """Channels of the next-finer map — what Head B's skip connection uses."""
        return self._channels[-2] if len(self._channels) > 1 else None

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns a list of 5 feature maps, coarse-to-fine in channel depth."""
        return self.net(x)


class XRVDenseNetEncoder(nn.Module):
    """
    CXR-pretrained encoder (torchxrayvision DenseNet-121, trained across
    several independent chest-X-ray corpora — CheXpert, MIMIC, NIH,
    PadChest, etc.) instead of ImageNet-pretrained EfficientNet. ImageNet
    features are texture-biased by construction — natural-image statistics
    reward exactly the kind of high-frequency acquisition cue the confound
    audit in code/readme.md found this project's default encoder riding.
    Features learned across several independent CXR acquisition pipelines
    should be less tied to any single source's texture signature.

    Same forward interface as SharedEncoder (`.out_channels`, `forward(x) ->
    list[Tensor]` with one spatial feature map) — a drop-in swap in
    build_model(), no changes needed to DiagnosticHead or the training loop.
    """

    # Re-derives the raw [0, 1] grayscale value from the ImageNet-normalized
    # RGB tensor the rest of the pipeline already produces, then applies the
    # [-1024, 1024] normalization torchxrayvision's models are trained on —
    # keeps this encoder a drop-in without a second dataset/transform path.
    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, weights: str = "densenet121-res224-all"):
        super().__init__()
        import torchxrayvision as xrv
        self.net = xrv.models.DenseNet(weights=weights)
        self._out_channels = 1024

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        mean = self._IMAGENET_MEAN.to(x.device)
        std = self._IMAGENET_STD.to(x.device)
        gray01 = (x * std + mean).mean(dim=1, keepdim=True).clamp(0, 1)  # (B, 1, H, W)
        xrv_in = gray01 * 2048 - 1024  # torchxrayvision's [-1024, 1024] convention

        feat = self.net.features(xrv_in)
        feat = F.relu(feat, inplace=False)
        return [feat]
