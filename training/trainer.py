"""
Training Loop for ZSL/GZSL cWGAN-GP.

Improvements:
- Two-Time-Scale Update Rule (TTUR): D LR > G LR
- Cosine annealing with warmup scheduler
- Exponential Moving Average (EMA) of Generator weights
- Gradient penalty scheduling (lambda_gp increases over time)
- Feature matching loss
- Gradient accumulation support
"""

import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluation.gan_eval import compute_fid, save_fake_images
from training.losses import WGANGPLoss
from utils.metrics import MetricsTracker
from utils.visualization import save_sample_grid


class EMAModel:
    """Exponential Moving Average of model parameters (state-dict based, no deepcopy)."""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self):
        model_dict = self.model.state_dict()
        for k in self.shadow:
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * model_dict[k]

    def apply_shadow(self):
        self.backup = {k: v.clone().detach() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(
            {k: v.to(self.model.parameters().__next__().device) for k, v in self.shadow.items()}
        )

    def restore(self):
        self.model.load_state_dict(self.backup)
        del self.backup

    def state_dict(self):
        return self.shadow


def _get_scheduler(optimizer, cfg, num_epochs, warmup_epochs):
    """Build cosine annealing with linear warmup scheduler."""
    sched_type = cfg.get("scheduler", "cosine_warmup")

    if sched_type == "cosine_warmup":
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=cfg.get("min_lr", 1e-6))
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        from torch.optim.lr_scheduler import ExponentialLR
        return ExponentialLR(optimizer, gamma=cfg.get("lr_decay", 0.995))


def train_gan(
    netG: nn.Module,
    netD: nn.Module,
    train_loader,
    seen_embeddings: torch.Tensor,
    device: torch.device,
    config: dict,
    real_images_dir: str,
):
    cfg = config["training"]
    paths = config["paths"]
    eval_cfg = config["evaluation"]
    model_cfg = config["model"]

    nz = model_cfg["generator"]["nz"]
    num_seen_classes = seen_embeddings.size(0)
    use_amp = cfg.get("mixed_precision", False) and device.type == "cuda"

    optimizerD = optim.Adam(netD.parameters(), lr=cfg["lr_d"], betas=(cfg["beta1"], cfg["beta2"]))
    optimizerG = optim.Adam(netG.parameters(), lr=cfg["lr_g"], betas=(cfg["beta1"], cfg["beta2"]))

    schedulerD = _get_scheduler(optimizerD, cfg, cfg["num_epochs"], cfg.get("warmup_epochs", 5))
    schedulerG = _get_scheduler(optimizerG, cfg, cfg["num_epochs"], cfg.get("warmup_epochs", 5))

    lambda_gp_base = cfg["lambda_gp"]
    lambda_gp_schedule = cfg.get("lambda_gp_schedule", False)
    lambda_gp_final = cfg.get("lambda_gp_final", lambda_gp_base * 2)
    lambda_gp_ramp_epochs = cfg.get("lambda_gp_ramp_epochs", 50)

    loss_fn = WGANGPLoss(lambda_gp=lambda_gp_base)
    scaler = GradScaler("cuda", enabled=use_amp)
    writer = SummaryWriter(paths["tensorboard_dir"])
    tracker = MetricsTracker(log_dir=paths["results_dir"], writer=writer)

    use_ema = cfg.get("use_ema", False)
    ema_decay = cfg.get("ema_decay", 0.999)
    ema_start_epoch = cfg.get("ema_start_epoch", 10)
    ema = EMAModel(netG, decay=ema_decay) if use_ema else None

    feature_matching_weight = cfg.get("feature_matching_weight", 0.0)

    fixed_noise = torch.randn(16, nz, device=device)
    fixed_labels = torch.randint(0, num_seen_classes, (16,), device=device)
    global_step = 0

    if use_amp:
        print("Mixed precision training enabled (FP16)")
    if use_ema:
        print(f"EMA enabled (decay={ema_decay}, starts at epoch {ema_start_epoch})")

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(cfg["num_epochs"]):
        netG.train()
        netD.train()

        current_lambda_gp = lambda_gp_base
        if lambda_gp_schedule and epoch < lambda_gp_ramp_epochs:
            progress = epoch / lambda_gp_ramp_epochs
            current_lambda_gp = lambda_gp_base + (lambda_gp_final - lambda_gp_base) * progress
            loss_fn.lambda_gp = current_lambda_gp

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['num_epochs']}")
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_batches = 0

        for real_images, labels in pbar:
            real_images, labels = real_images.to(device), labels.to(device)
            batch_size = real_images.size(0)
            epoch_batches += 1
            global_step += 1

            for _ in range(cfg["n_critic"]):
                netD.zero_grad(set_to_none=True)

                with autocast("cuda", enabled=use_amp):
                    z = torch.randn(batch_size, nz, device=device)
                    fake_images = netG(z, labels, seen_embeddings)
                    d_loss_dict = loss_fn.discriminator_loss(
                        netD, real_images, fake_images, labels, seen_embeddings, device
                    )

                scaler.scale(d_loss_dict["d_loss"]).backward()
                scaler.unscale_(optimizerD)
                torch.nn.utils.clip_grad_norm_(netD.parameters(), cfg["grad_clip"])
                scaler.step(optimizerD)
                scaler.update()

            netG.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                z = torch.randn(batch_size, nz, device=device)
                fake_images = netG(z, labels, seen_embeddings)
                g_loss_dict = loss_fn.generator_loss(
                    netD,
                    fake_images,
                    labels,
                    seen_embeddings,
                    real_images=real_images,
                    feature_matching_weight=feature_matching_weight,
                )

            scaler.scale(g_loss_dict["g_loss"]).backward()
            scaler.unscale_(optimizerG)
            torch.nn.utils.clip_grad_norm_(netG.parameters(), cfg["grad_clip"])
            scaler.step(optimizerG)
            scaler.update()

            if use_ema and epoch >= ema_start_epoch:
                ema.update()

            tracker.update(
                g_loss_dict["g_loss"].item(),
                d_loss_dict["d_loss"].item(),
                d_loss_dict["wasserstein_distance"],
                d_loss_dict["gradient_penalty"],
            )

            epoch_g_loss += g_loss_dict["g_loss"].item()
            epoch_d_loss += d_loss_dict["d_loss"].item()

            if global_step % 50 == 0:
                writer.add_scalar("Loss/Generator", g_loss_dict["g_loss"].item(), global_step)
                writer.add_scalar("Loss/Discriminator", d_loss_dict["d_loss"].item(), global_step)
                writer.add_scalar("Loss/Wasserstein_Distance", d_loss_dict["wasserstein_distance"], global_step)
                writer.add_scalar("Loss/Gradient_Penalty", d_loss_dict["gradient_penalty"], global_step)
                writer.add_scalar("LR/Discriminator", optimizerD.param_groups[0]["lr"], global_step)
                writer.add_scalar("LR/Generator", optimizerG.param_groups[0]["lr"], global_step)
                if "feature_matching_loss" in g_loss_dict:
                    writer.add_scalar("Loss/Feature_Matching", g_loss_dict["feature_matching_loss"], global_step)
                if lambda_gp_schedule:
                    writer.add_scalar("Config/Lambda_GP", current_lambda_gp, global_step)

            pbar.set_postfix(
                {
                    "G": f"{epoch_g_loss / epoch_batches:.4f}",
                    "D": f"{epoch_d_loss / epoch_batches:.4f}",
                    "W": f"{d_loss_dict['wasserstein_distance']:.4f}",
                    "lrD": f"{optimizerD.param_groups[0]['lr']:.6f}",
                }
            )

        schedulerD.step()
        schedulerG.step()

        eval_model = ema.shadow if (use_ema and epoch >= ema_start_epoch) else netG
        grid = save_sample_grid(eval_model, fixed_noise, fixed_labels, seen_embeddings, epoch + 1, device, save_dir=paths["results_dir"])
        writer.add_image("Generated_Samples", grid, epoch + 1)

        if (epoch + 1) % cfg["eval_interval"] == 0 or epoch == cfg["num_epochs"] - 1:
            print(f"\n{'=' * 70}")
            print(f"Evaluating at epoch {epoch + 1}")
            print(f"{'=' * 70}")

            fake_dir = save_fake_images(
                eval_model,
                epoch + 1,
                device,
                num_samples=eval_cfg["fid_samples"],
                batch_size=cfg["batch_size"],
                nz=nz,
                num_seen_classes=num_seen_classes,
                seen_embeddings=seen_embeddings,
            )
            metrics = compute_fid(real_images_dir, fake_dir)
            print(
                f"FID: {metrics['fid']:.2f} | IS: {metrics['is_mean']:.2f} +- {metrics['is_std']:.2f} | KID: {metrics['kid_mean']:.4f}"
            )

            improved = tracker.update_metrics(
                metrics,
                epoch + 1,
                eval_model if use_ema else netG,
                netD,
                checkpoints_dir=paths["checkpoints_dir"],
            )
            if improved:
                print(f"New best FID: {metrics['fid']:.2f}")
            else:
                print(f"FID: {metrics['fid']:.2f} vs best {tracker.best_fid:.2f}")

            if tracker.should_stop_early(cfg["early_stopping_patience"]):
                print(f"\nEarly stopping after {epoch + 1} epochs without improvement\n")
                break

        if (epoch + 1) % cfg["save_interval"] == 0:
            ckpt_dir = Path(paths["checkpoints_dir"])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_state = {
                "epoch": epoch + 1,
                "generator": netG.state_dict(),
                "discriminator": netD.state_dict(),
                "optimizerG": optimizerG.state_dict(),
                "optimizerD": optimizerD.state_dict(),
                "schedulerG": schedulerG.state_dict(),
                "schedulerD": schedulerD.state_dict(),
                "config": config,
            }
            if use_ema and epoch >= ema_start_epoch:
                save_state["ema_shadow"] = ema.state_dict()
            torch.save(save_state, ckpt_dir / f"checkpoint_epoch_{epoch + 1:03d}.pth")

    writer.close()
    return tracker
