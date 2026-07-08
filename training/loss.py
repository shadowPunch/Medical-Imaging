import torch
import torch.nn as nn

from data.dataset import TBCXRDataset


def compute_pos_weight(dataset: TBCXRDataset) -> torch.Tensor:
    """
    BCEWithLogitsLoss pos_weight = num_negative / num_positive.
    Downweights the majority class; used on top of WeightedRandomSampler
    as a secondary correction for residual imbalance.
    """
    neg, pos = dataset.class_counts()
    return torch.tensor([neg / pos], dtype=torch.float32)


class LabelSmoothBCE(nn.Module):
    """
    BCE with logits + label smoothing.
    Smoothing prevents overconfident predictions and improves calibration,
    which matters here because we select operating points by threshold.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        # Shift hard 0/1 labels toward 0.5 by the smoothing factor
        targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits.squeeze(1), targets)
