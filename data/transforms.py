import numpy as np
from torchvision import transforms

# ImageNet mean/std — EfficientNet pretrained on ImageNet expects this normalisation
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 512):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        # Mild brightness/contrast jitter simulates different X-ray exposure settings
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def get_val_transforms(image_size: int = 512):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


class _AlbumentationsToPIL:
    """Bridges an albumentations pipeline (numpy uint8 in/out) into a
    torchvision transforms.Compose (PIL in/out) so it can sit ahead of the
    existing ToTensor/Normalize step without duplicating that logic."""

    def __init__(self, album_transform):
        self._t = album_transform

    def __call__(self, img):
        from PIL import Image
        out = self._t(image=np.array(img.convert("RGB")))["image"]
        return Image.fromarray(out)


def get_texture_aug_transforms(image_size: int = 320, strength: str = "aggressive"):
    """
    Randomizes the acquisition-level texture signature (resolution round-trip,
    sharpen/blur jitter, noise floor, JPEG compression, gamma/contrast curve)
    on top of the standard geometric augmentation. Targets the confound
    quantified in docs/investigation-log.md's Phase 2 validation results — the
    high-pass-residual and single-patch tests show the shortcut lives in
    exactly this kind of high-frequency processing signature.

    TB findings (cavitation, miliary nodules, reticulonodular infiltrate) are
    *also* high-frequency texture, so this is a dial, not a switch: too
    aggressive and it degrades diagnostic texture along with confound texture.
    `strength` sweeps that dial —
      "mild":       gamma/brightness/contrast + light blur-sharpen jitter only.
                    No resolution round-trip, no compression, no noise.
      "medium":     mild + a single mild Downscale round-trip (0.6-0.9).
      "aggressive": the original, unconstrained version — two Downscale
                    tiers (down to 0.35), heavier noise, JPEG compression.
    """
    import albumentations as A

    blur_sharpen = A.OneOf([
        A.Sharpen(alpha=(0.1, 0.5), lightness=(0.8, 1.2), p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.UnsharpMask(blur_limit=(3, 7), alpha=(0.2, 0.7), p=1.0),
    ], p=0.6)
    gamma = A.RandomGamma(gamma_limit=(70, 140), p=0.7)

    if strength == "mild":
        steps = [blur_sharpen, gamma]
    elif strength == "medium":
        steps = [A.Downscale(scale_range=(0.6, 0.9), p=0.5), blur_sharpen, gamma]
    elif strength == "aggressive":
        steps = [
            A.OneOf([
                A.Downscale(scale_range=(0.5, 0.9), p=1.0),
                A.Downscale(scale_range=(0.35, 0.6), p=1.0),
            ], p=0.5),
            blur_sharpen,
            A.OneOf([
                A.GaussNoise(std_range=(0.01, 0.08), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.92, 1.08), elementwise=True, p=1.0),
            ], p=0.5),
            A.ImageCompression(quality_range=(45, 100), p=0.4),
            gamma,
        ]
    else:
        raise ValueError(f"Unknown strength: {strength!r} (mild/medium/aggressive)")

    album = A.Compose(steps)

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        _AlbumentationsToPIL(album),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
