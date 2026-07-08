from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    # 320px keeps peak training memory ~1.5GB on a 4GB GPU (measured empirically
    # with sanity_check.py); 512px only fits at batch_size<=8, which hurts
    # BatchNorm stability. Raise this if training on a card with more headroom.
    image_size: int = 320
    # TBX11K: latent_tb cases are radiographically ambiguous — excluded by default
    tbx11k_latent_as_positive: bool = False


@dataclass
class ModelConfig:
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    dropout: float = 0.3


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    patience: int = 10  # early-stop if val AUC doesn't improve


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: Path = Path("outputs/diagnostic")
    device: str = "cuda"
    seed: int = 42
