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
