import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping over a specified target layer.

    Usage:
        cam = GradCAM(model, target_layer=model.encoder.net.blocks[-1])
        heatmap = cam(image_tensor)          # (H, W) float in [0, 1]
        overlay = overlay_heatmap(img, heatmap)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output):
        # output shape: (B, C, H, W)
        self._activations = output.detach()

    def _save_gradient(self, _, __, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """
        x: (1, C, H, W) — single image tensor on the same device as model.
        Returns a (H, W) heatmap normalized to [0, 1].
        """
        self.model.eval()
        logit = self.model(x)
        self.model.zero_grad()
        logit.backward()

        # Global-average the gradients over spatial dims → channel importance weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input resolution
        h, w = x.shape[-2:]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam


def overlay_heatmap(
    image_np: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap over a CXR image for visualization.
    image_np : (H, W, 3) uint8 RGB
    cam      : (H, W) float in [0, 1]
    Returns    (H, W, 3) uint8 RGB
    """
    colormap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    return (alpha * colormap + (1 - alpha) * image_np).astype(np.uint8)
