from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval.metrics import evaluate, print_report


class Trainer:
    """
    Handles the train / validate / checkpoint cycle.
    Mixed-precision (AMP) is always on; GradScaler is a no-op on CPU.
    """

    def __init__(
        self,
        model:      torch.nn.Module,
        optimizer:  torch.optim.Optimizer,
        criterion:  torch.nn.Module,
        scheduler,                         # any LRScheduler
        device:     torch.device,
        output_dir: Path,
    ):
        self.model      = model
        self.optimizer  = optimizer
        self.criterion  = criterion
        self.scheduler  = scheduler
        self.device     = device
        self.output_dir = output_dir
        self.scaler     = torch.amp.GradScaler(device.type)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _run_epoch(
        self, loader: DataLoader, train: bool
    ) -> tuple[float, np.ndarray, np.ndarray]:
        self.model.train(train)
        total_loss, all_probs, all_labels = 0.0, [], []

        with torch.set_grad_enabled(train):
            for imgs, labels in tqdm(loader, desc="train" if train else "val", leave=False):
                imgs   = imgs.to(self.device)
                labels = labels.to(self.device)

                with torch.amp.autocast(self.device.type):
                    logits = self.model(imgs)
                    loss   = self.criterion(logits, labels)

                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                total_loss += loss.item() * len(labels)
                probs = torch.sigmoid(logits).squeeze(1).float().detach().cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())

        n = len(loader.dataset)
        return total_loss / n, np.concatenate(all_probs), np.concatenate(all_labels)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int, patience: int = 10) -> None:
        best_auc, no_improve = 0.0, 0

        for epoch in range(1, epochs + 1):
            tr_loss, _, _              = self._run_epoch(train_loader, train=True)
            val_loss, val_probs, val_y = self._run_epoch(val_loader,   train=False)
            self.scheduler.step()

            metrics = evaluate(val_y, val_probs)
            print(f"\nEpoch {epoch:3d}  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")
            print_report(metrics)

            if metrics["auc"] > best_auc:
                best_auc   = metrics["auc"]
                no_improve = 0
                self._save_checkpoint(epoch, metrics)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\nEarly stop at epoch {epoch} — no AUC gain for {patience} epochs.")
                    break

        print(f"\nBest val AUC: {best_auc:.4f}")

    def evaluate_loader(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on a loader; returns (probabilities, labels)."""
        _, probs, labels = self._run_epoch(loader, train=False)
        return probs, labels

    def load_best(self) -> None:
        """Load the best saved checkpoint back into self.model."""
        # weights_only=False needed because checkpoint includes optimizer state + metrics dict
        ckpt = torch.load(self.output_dir / "best_model.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        print(f"Loaded best model — epoch {ckpt['epoch']}  AUC={ckpt['metrics']['auc']:.4f}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, metrics: dict) -> None:
        torch.save(
            {"epoch": epoch, "model": self.model.state_dict(),
             "optimizer": self.optimizer.state_dict(), "metrics": metrics},
            self.output_dir / "best_model.pt",
        )
        print(f"  → checkpoint saved  (AUC={metrics['auc']:.4f})")
