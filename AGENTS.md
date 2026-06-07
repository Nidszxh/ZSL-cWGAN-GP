# ZSL-cWGAN-GP

Zero-Shot Learning with Conditional WGAN-GP on CIFAR-100 using PyTorch. Supports both standard ZSL and **Generalized Zero-Shot Learning (GZSL)** with calibrated stacking.

## Entrypoints

| Command | Purpose |
|---|---|
| `python main.py` | Full training pipeline (CLIP, config-driven, `configs/config.yaml`) |
| `python ZSLcWGAN-GP.py` | Original monolithic GloVe-based training (legacy) |
| `python test_training.py` | Quick 5-epoch sanity check (CLIP, no AMP, batch_size=64) |
| `python test_clip.py` | Standalone CLIP embedding quality tests |
| `python app.py` | Gradio web demo (requires `checkpoints/best_zsl_classifier.pth`) |

## Key architecture

Modular codebase under `models/`, `utils/`, `training/`, `evaluation/`. The monolithic `ZSLcWGAN-GP.py` (GloVe 300d) is legacy. See `ARCHITECTURE.md` for full data flow and component wiring.

- **Generator** (`models/generator.py`): SAGAN self-attention at configurable resolutions (8x8, 16x16), spectral norm, orthogonal init. Semantic projection MLP has spectral norm for stable conditioning. Input: noise + semantic embedding indexed by label.
- **Discriminator** (`models/discriminator.py`): Projection-based conditional critic, spectral norm. Returns features for feature matching loss when `return_features=True`.
- **ZSL classifier** (`models/zsl_classifier.py`): Pluggable backbone — ResNet-18 (default, pretrained), EfficientNet-B0, or custom 4-conv CNN. Classifier head: Dropout(0.5) → Linear(512) → ReLU → Dropout(0.3) → Linear(256→num_classes).
- **GZSL classifier** (`evaluation/gzsl_eval.py`): Trains on real seen + synthetic unseen jointly. `CalibratedClassifier` wrapper learns temperature scaling to reduce seen-class bias. Reports seen/unseen accuracy and harmonic mean.
- **Mixed precision**: `torch.amp.GradScaler("cuda")` + `autocast` — enabled via `config['training']['mixed_precision']`.
- **EMA**: State-dict-based exponential moving average of Generator weights for stable evaluation.
- **TTUR**: Discriminator LR (0.0004) is 4x Generator LR (0.0001) for faster convergence.

## Config

- `configs/config.yaml` drives the modular code. Nested key access: `config['training']['lr_d']`.
- `ZSLcWGAN-GP.py` has its own flat `config` dict: `config['lr_d']`.
- Class split (80 seen / 20 unseen) persisted to `cache/class_split.json`.
- Setting `embeddings.type` to `"clip"` (768d ViT-L/14) or `"clip_ensemble"` or `"glove"` (300d); model `semantic_dim` must match.

## Key gotchas

- **FID evaluation is slow**: generates 20K images per eval via `torch-fidelity` every `eval_interval` epochs.
- **Checkpoints**: best model (lowest FID) → `checkpoints/best_model.pth`; ZSL classifier → `checkpoints/best_zsl_classifier.pth`; GZSL classifier → `checkpoints/best_gzsl_classifier.pth`; periodic → `checkpoint_epoch_XXX.pth`.
- **Early stopping**: 20 epochs without FID improvement (`config['training']['early_stopping_patience']`).
- **GPU required** for practical training; CPU will be extremely slow.
- **Batch size 128 + mixed precision** fits 8GB VRAM (RTX 4060, ~501 img/s). Use `test_training.py` (batch 64, no AMP) for quick GPU smoke tests.
- **GradScaler**: `scaler.update()` called inside the n_critic loop after each D step (required for AMP state machine with multiple optimizer steps per batch).
- **`cache/` directory**: auto-generated class split + embeddings pickle; safe to delete for fresh split/embeddings.
- **`data/`**: auto-downloaded CIFAR-100; `cache/clip/`: auto-downloaded CLIP model weights.

## Useful commands

```bash
tensorboard --logdir=runs
python test_clip.py          # quick CLIP integration check
python test_training.py      # 5-epoch pipeline sanity check
pip install -r requirements.txt
```

---

## Pipeline overview

`main.py` orchestrates:
1. Load `configs/config.yaml` → set seeds, create dirs
2. Split 100 CIFAR-100 classes → 80 seen + 20 unseen (cached in `cache/class_split.json`)
3. Load CLIP/GloVe embeddings via `EmbeddingManager` (cached in `cache/embeddings_clip.pkl`)
4. Build `Generator` + `Discriminator` (orthogonal init, spectral norm, SN on semantic proj)
5. Train GAN: n_critic D steps per G step, WGAN-GP loss, AMP if enabled, feature matching loss, FID eval every N epochs, early stopping. EMA shadow of G used for eval.
6. Generate synthetic unseen-class images (2000 per class)
7. Train `ZSLClassifier` (ResNet-18 pretrained backbone) on synthetic data (AdamW, 50 epochs, ReduceLROnPlateau, early stopping patience=10)
8. Train `GZSLClassifier` on real seen + synthetic unseen jointly with `CalibratedClassifier` (learned temperature scaling)
9. Evaluate on real unseen-class test set → ZSL Top-1, Top-5, mean class accuracy, confusion matrix
10. Evaluate on seen + unseen test set → GZSL Seen/Unseen accuracy, Harmonic Mean (H)
11. Save plots (`training_curves.png`, `metrics_progress.png`, `zsl_confusion_matrix.png`, `gzsl_results.png`, `experiment_summary.png`)

## Training loop details

**WGAN-GP**: D loss = `E[D(fake)] - E[D(real)] + λ·GP` where GP penalizes deviation of gradient norm from 1 on interpolates. G loss = `-E[D(fake)]` + `feature_matching_weight * FM`. Adam betas=(0.0, 0.9) — β₁=0.0 is standard for WGAN.

**TTUR**: D gets 4× higher learning rate (0.0004) than G (0.0001) for faster critic convergence. n_critic=5.

**Feature matching loss**: When `feature_matching_weight > 0`, G also minimizes L2 distance between D's intermediate feature maps for real vs fake images.

**Cosine annealing with warmup**: LR schedule uses linear warmup for first N epochs, then cosine annealing to `min_lr`.

**EMA**: Exponential moving average of G weights (decay=0.999). Shadow weights used for eval/sample generation after `ema_start_epoch`.

**Gradient penalty scheduling** (optional): λ_gp ramps from 10 → 20 over first 50 epochs for stricter Lipschitz enforcement over time.

**AMP flow** (when `mixed_precision: true`):
```python
with autocast("cuda", enabled=use_amp):
    ...forward passes...
scaler.scale(loss).backward()
scaler.unscale_(optimizer)          # pass optimizer, not model params
torch.nn.utils.clip_grad_norm_(...)
scaler.step(optimizer)
scaler.update()                     # inside n_critic loop, after each D step
```
`scaler.update()` inside the inner n_critic loop is required because the AMP state machine must reset after each optimizer step when multiple steps are taken per batch.

## Data transforms

**Training**: `RandomCrop(32,4)` + `RandomHorizontalFlip` + `ColorJitter(0.1)` → `ToTensor()` → `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))` → output in [-1, 1].
**Validation/Test**: `ToTensor()` → `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`.

Generator outputs via Tanh → [-1, 1] range. When saving to PNG, `(x+1)/2` maps to [0, 1] for `vutils.save_image`. ZSL classifier gets the [-1, 1] tensor directly (matches real data normalization).

## Generator architecture

```
z [128] ──┐
           ├─ concat → Linear(384→2048) → reshape(512×4×4)
sem_proj ──┘          ↑ 2-layer MLP(768→256→256, SN+LeakyReLU+Dropout)
→ Up1: nearest×2, SN Conv(512→256, 3×3), BN, LeakyReLU    [4→8]
→ SelfAttention(256 channels) at 8×8                        [8→8]
→ Up2: nearest×2, SN Conv(256→128, 3×3), BN, LeakyReLU     [8→16]
→ SelfAttention(128 channels) at 16×16                      [16→16]
→ Up3: nearest×2, SN Conv(128→3, 3×3), Tanh                 [16→32]
```

~5.0M params. Orthogonal init. Self-attention starts as identity (`gamma=0`). Semantic projection MLP has spectral norm for stable conditioning.

## Discriminator architecture

```
Image [3×32×32]
→ SN Conv(3→64, 3×3), LReLU
→ SN Conv(64→64, 4×4, stride 2), LReLU                   [32→16]
→ SN Conv(64→128, 3×3), LReLU
→ SN Conv(128→128, 4×4, stride 2), LReLU                  [16→8]
→ SN Conv(128→256, 3×3), LReLU
→ SN Conv(256→256, 4×4, stride 2), LReLU                  [8→4]
→ SN Conv(256→512, 3×3), LReLU
→ SN Conv(512→1, 4×4) → unconditional score
→ adaptive_avg_pool + embed_output(256→512) · h_pooled → conditional score
```

~3.3M params. Final score = unconditional + conditional (projection trick). Semantic projection: SN Linear(768→256) + LReLU + SN Linear(256→256) + LReLU.

## ZSL classifier architecture

Pluggable backbone via `model.classifier.backbone`:

**ResNet-18 (default)**: Pretrained on ImageNet, ~11.4M params. Head: Dropout(0.5) → Linear(512→512) → ReLU → Dropout(0.3) → Linear(512→num_classes).

**EfficientNet-B0**: Pretrained on ImageNet, ~4.7M params. Same head architecture.

**Custom CNN (legacy)**: 4-layer conv (64→128→256→512) + AdaptiveAvgPool, ~3.0M params.

Trained on synthetic data: 2000 samples/class × 20 unseen classes = 40K total (80/20 train/val split). Data augmentation: `RandomCrop(32,4)` + `RandomHorizontalFlip`. Optimizer: AdamW (lr=0.001, wd=1e-4). Scheduler: ReduceLROnPlateau(mode="max", factor=0.5, patience=3) on validation accuracy. Early stopping: patience=10.

**ZSL is true ZSL**: The classifier is trained exclusively on generated samples and evaluated on real unseen-class images.

## GZSL classifier architecture

Same backbone as ZSL classifier but outputs `num_seen + num_unseen` (100) classes. Trained on:
- Real seen-class images from training set
- Synthetic unseen-class images from generator

**CalibratedClassifier** wrapper: Applies learned temperature scaling to separate seen/unseen logits, reducing the bias toward seen classes. Two learnable parameters: `temperature_s` (seen), `temperature_u` (unseen), and `bias_shift`.

Evaluates: Seen accuracy, Unseen accuracy, Harmonic mean (H) = `2·S·U / (S + U)`.

## Evaluation metrics

- **FID**: via `torch-fidelity` comparing real vs fake PNG directories (20K images each). Saved under `results/real/` and `results/fake_epochXXX/`.
- **IS / KID**: also via `torch-fidelity` alongside FID.
- **ZSL Top-1 / Top-5**: percentage of correctly classified real unseen-class test images.
- **ZSL mean class accuracy**: average of per-class accuracies.
- **GZSL Seen accuracy**: classification accuracy on real seen-class images (100-class output).
- **GZSL Unseen accuracy**: classification accuracy on real unseen-class images (100-class output).
- **GZSL Harmonic Mean**: `2·seen·unseen / (seen + unseen)` — the primary GZSL metric.

## Directory structure

```
configs/config.yaml          ← single source of truth
models/                      ← generator.py, discriminator.py, zsl_classifier.py
training/                    ← trainer.py (loop + EMA), losses.py (WGAN-GP + feature matching)
evaluation/                  ← gan_eval.py (FID/IS/KID), zsl_eval.py (ZSL), gzsl_eval.py (GZSL)
utils/                       ← embeddings.py (CLIP/ensemble/GloVe), data_loader.py, metrics.py, visualization.py
main.py                      ← orchestrator
app.py                       ← Gradio demo
ZSLcWGAN-GP.py               ← legacy GloVe (ignore for new work)
test_training.py             ← 5-epoch sanity
test_clip.py                 ← CLIP test suite
cache/                       ← auto-gen: class_split.json, embeddings_*.pkl
checkpoints/                 ← best_model.pth, best_zsl_classifier.pth, best_gzsl_classifier.pth
results/                     ← real/, fake_epoch*/, unseen_synthetic/, plots
runs/                        ← TensorBoard logs
```

## Dependencies (from requirements.txt)

Core: torch, torchvision, numpy, scipy, Pillow. Training: tqdm, torch-fidelity, scikit-learn. CLIP: transformers, ftfy, regex. Viz: matplotlib, seaborn, opencv-python. Logging: tensorboard, pyyaml, wandb (optional). Demo: gradio. Quality: pytest, black, flake8, mypy (optional).
