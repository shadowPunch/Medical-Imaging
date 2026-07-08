"""Quick end-to-end check: data loading → model forward → loss → backward."""
import sys, time
sys.path.insert(0, ".")

# Python 3.14 changed the default multiprocessing start method on Linux
# from 'fork' to 'forkserver'. Setting it back to 'fork' here avoids the
# need to guard every DataLoader call with if __name__ == '__main__'.
import torch.multiprocessing as _mp
_mp.set_start_method("fork", force=True)
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

from data.dataset import load_shenzhen, load_montgomery, load_tbx11k, TBCXRDataset
from data.transforms import get_train_transforms
from models.tb_model import build_model
from training.loss import LabelSmoothBCE, compute_pos_weight

DS = Path("../datasets")

# 1. Loaders
sz    = load_shenzhen(DS / "tb-shenzen")
mg    = load_montgomery(DS / "tb-montgomery")
tb_tr = load_tbx11k(DS / "tbx11k", split="train")
tb_val= load_tbx11k(DS / "tbx11k", split="val")

train_samples = sz + tb_tr
print(f"[data] train={len(train_samples)}  val={len(tb_val)}  held-out(mg)={len(mg)}")

# 2. Find the largest (image_size, batch_size) that fits in 4GB, training mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pos_weight_dummy = torch.tensor([1.0]).to(device)
criterion = LabelSmoothBCE(pos_weight=pos_weight_dummy, smoothing=0.1)

def try_config(image_size: int, batch_size: int) -> float | None:
    """Returns peak GB if it fits, None if OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model("efficientnet_b0", pretrained=False, dropout=0.3).to(device)
    scaler = torch.amp.GradScaler(device.type)
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    y = torch.randint(0, 2, (batch_size,), device=device)
    try:
        model.train()
        with torch.amp.autocast(device.type):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        peak = torch.cuda.max_memory_allocated() / 1e9
        del model, x, y, loss
        torch.cuda.empty_cache()
        return peak
    except torch.OutOfMemoryError:
        del model, x, y
        torch.cuda.empty_cache()
        return None

print("[gpu] probing image_size × batch_size that fits in 4GB VRAM...")
for image_size in [512, 384, 320, 256]:
    for batch_size in [32, 16, 8, 4]:
        peak = try_config(image_size, batch_size)
        status = f"{peak:.2f} GB" if peak is not None else "OOM"
        print(f"  size={image_size:3d}  batch={batch_size:2d}  → {status}")
    print()

# 3. End-to-end check with a config known to be safe: 320px, batch 8
dataset = TBCXRDataset(train_samples, get_train_transforms(320))
labels  = np.array(dataset.labels)
counts  = np.bincount(labels)
sampler = WeightedRandomSampler((1.0 / counts)[labels], len(labels))
loader  = DataLoader(dataset, batch_size=8, sampler=sampler, num_workers=2, pin_memory=True)

t0 = time.time()
imgs, lbls = next(iter(loader))
print(f"[data] batch: {tuple(imgs.shape)}  load_time={time.time()-t0:.2f}s  "
      f"pos_in_batch={lbls.sum().item()}/8")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
model = build_model("efficientnet_b0", pretrained=False, dropout=0.3).to(device)
imgs_d, lbls_d = imgs.to(device), lbls.to(device)
pos_weight = compute_pos_weight(dataset).to(device)
criterion  = LabelSmoothBCE(pos_weight=pos_weight, smoothing=0.1)
scaler = torch.amp.GradScaler(device.type)

model.train()
with torch.amp.autocast(device.type):
    loss = criterion(model(imgs_d), lbls_d)
scaler.scale(loss).backward()
grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"[loss] {loss.item():.4f}  pos_weight={pos_weight.item():.2f}x  grad_norm={grad_norm:.2f}")
print(f"[gpu] peak memory: {peak_gb:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print("\nAll checks passed.")
