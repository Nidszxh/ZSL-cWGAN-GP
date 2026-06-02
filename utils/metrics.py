from pathlib import Path
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter


class MetricsTracker:
    def __init__(self, log_dir: str = "results", writer: Optional[SummaryWriter] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.g_losses: list[float] = []
        self.d_losses: list[float] = []
        self.w_distances: list[float] = []
        self.gp_values: list[float] = []
        self.metrics_history: list[tuple[int, dict]] = []

        self.best_fid = float("inf")
        self.epochs_without_improv = 0
        self.writer = writer

    def update(self, g_loss: float, d_loss: float, w_dist: float, gp: float):
        self.g_losses.append(g_loss)
        self.d_losses.append(d_loss)
        self.w_distances.append(w_dist)
        self.gp_values.append(gp)

    def update_metrics(self, metrics: dict, epoch: int, netG: torch.nn.Module, netD: torch.nn.Module, checkpoints_dir: str = "checkpoints") -> bool:
        fid_score = metrics["fid"]
        self.metrics_history.append((epoch, metrics))

        if self.writer is not None:
            self.writer.add_scalar("Metrics/FID", fid_score, epoch)
            self.writer.add_scalar("Metrics/IS_mean", metrics.get("is_mean", 0), epoch)
            self.writer.add_scalar("Metrics/KID", metrics.get("kid_mean", 0), epoch)

        if fid_score < self.best_fid:
            self.best_fid = fid_score
            self.epochs_without_improv = 0

            ckpt_dir = Path(checkpoints_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "generator": netG.state_dict(),
                    "discriminator": netD.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                },
                ckpt_dir / "best_model.pth",
            )
            return True
        else:
            self.epochs_without_improv += 1
            return False

    def should_stop_early(self, patience: int) -> bool:
        return self.epochs_without_improv >= patience
