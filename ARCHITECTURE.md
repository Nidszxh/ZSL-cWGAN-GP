# ZSL-cWGAN-GP Architecture

## Overview

ZSL-cWGAN-GP is a conditional generative adversarial network that synthesizes CIFAR-100 images conditioned on semantic text embeddings (CLIP or GloVe). The trained generator produces images for unseen classes, which are used to train a zero-shot classifier. An optional Generalized Zero-Shot Learning (GZSL) branch additionally classifies seen classes with calibrated temperature scaling to reduce seen-bias.

---

## Data Flow

```
CIFAR-100 (100 classes)
    │
    ├── 80 seen classes ──► Generator ──► Synthetic unseen images
    │                          │                  │
    │                          ▼                  ▼
    │                    Discriminator      ZSL Classifier (unseen only)
    │                    (real vs fake)     GZSL Classifier (seen+unseen)
    │
    └── 20 unseen classes ──► Held out during GAN training
                              Used only for ZSL/GZSL evaluation
```

---

## Component Pipeline

### 1. Embeddings (`src/utils/embeddings.py`)

```
CIFAR-100 class names
    │
    ├── CLIPTextEmbedder: "openai/clip-vit-large-patch14" → 768-dim
    │   - Uses 5 prompt templates per class, averages them
    │   - L2-normalizes output
    │   - Optional ensemble: multiple CLIP models averaged
    │   - "both" mode concatenates CLIP + GloVe embeddings
    │
    └── GloVeEmbedder: glove.6B.300d.txt → 300-dim
        - Falls back to word averaging, then random if missing
```

Both produce a tensor `[num_classes, embedding_dim]` that the GAN indexes by label at runtime.

---

### 2. Generator (`src/models/generator.py`)

```
z (noise) [B, 128] ───┐
                       ├── concat ──► Linear ──► reshape ──► Conv2D ──► Attn ──► Conv2D ──► Conv2D ──► Tanh ──► [B, 3, 32, 32]
semantic_emb[labels]   │            8*4*4           4x4      4→8      8x8    8→16      16→32        [-1, 1]
[B, 512] ───► MLP ────┘
             256-256
            (LeakyReLU+Dropout)
```

**Semantic projection**: 2-layer MLP (768→256→256 for CLIP, 300→256→256 for GloVe) with LeakyReLU(0.2), Dropout(0.2), and spectral norm.

**Self-attention**: SAGAN-style block at configurable resolutions (default 8x8, optional 16x16) with spectral norm Q/K/V projections. Initialized with `gamma=0` so training starts as identity and gradually learns global dependencies.

**Generation**: Starts at 4×4, nearest-neighbor upsamples to 8×8 → 16×16 → 32×32 with spectral norm conv layers and BatchNorm.

**Weight init**: Orthogonal for all conv/linear layers.

---

### 3. Discriminator (`src/models/discriminator.py`)

```
image [B, 3, 32, 32]
    │
    └──► Conv2D layers (spectral norm, LeakyReLU) ──► conv 4×4 ──► score [B]
         32→64→128→256→512                                      (unconditional)
                        │
semantic_emb[labels]    │
    └──► MLP (SN) ──► Linear ──► * ──► sum ──► final score
         256-256       512        │         [B]
                                  │
                    adaptive_avg_pool2d ────► flatten
```

**Projection discriminator**: The conditional score is the inner product between the projected embedding and the pooled feature map. This is the standard cGAN projection trick — no explicit label injection into the feature layers.

Uses `view(-1)` instead of `squeeze()` for batch_size=1 safety.

---

### 4. WGAN-GP Training Loop

```
for each batch:
    # Critic (n_critic=5 steps per generator step)
    for _ in range(n_critic):
        fake ← G(z, labels)
        D_loss = E[D(fake)] - E[D(real)] + λ · GP
        GP = E[(||∇D(interpolated)||₂ - 1)²]
        update D with Adam(β₁=0.0, β₂=0.9)

    # Generator
    fake ← G(z, labels)
    G_loss = -E[D(fake)] + feature_matching_weight · FM
    update G with Adam(β₁=0.0, β₂=0.9)
```

**Key parameters** (from `configs/config.yaml`):

| Parameter | Value | Purpose |
|---|---|---|
| `λ_gp` | 10 | Gradient penalty weight |
| `n_critic` | 5 | D updates per G update |
| `β₁` | 0.0 | Adam momentum (standard for WGAN) |
| `β₂` | 0.9 | Adam second moment |
| `lr_D` | 0.0004 | Discriminator learning rate (TTUR: 4x G) |
| `lr_G` | 0.0001 | Generator learning rate |
| `grad_clip` | 1.0 | Global gradient norm clipping |

**Mixed precision**: Forward passes run in FP16 via `torch.amp.autocast("cuda")`; `GradScaler("cuda")` prevents underflow. `scaler.update()` is called after each D step inside the n_critic loop (required for AMP state machine with multiple optimizer steps per batch).

**Regularization**:
- Spectral normalization on all conv/linear layers in both G and D
- Gradient penalty on interpolates between real and fake
- Dropout (0.2) in G's semantic projection
- BatchNorm in G's upsampling layers
- Feature matching loss (configurable weight) from D's intermediate features

**EMA**: Exponential moving average of generator weights (decay=0.999) applied in-place before evaluation and restored after.

**Checkpoint resume**: `--resume` CLI flag restores G, D, optimizers, schedulers, EMA shadow, and global_step from a checkpoint file.

---

### 5. Zero-Shot Classifier (`models/zsl_classifier.py`)

A pluggable backbone classifier trained on synthetic images from the generator:

```
input [3, 32, 32]
    └──► Backbone (ResNet-18 / EfficientNet-B0 / Custom CNN)
    └──► Classifier Head:
           Dropout(0.5) → Linear(feat_dim→512) → ReLU → Dropout(0.3) → Linear(512→num_classes)
```

**Backbone options**:
- **ResNet-18** (default): Pretrained on ImageNet, ~11.4M params, 512-dim features
- **EfficientNet-B0**: Pretrained on ImageNet, ~4.7M params, 1280-dim features
- **Custom CNN** (legacy): 4× Conv2D (64→128→256→512) + AdaptiveAvgPool2d, ~3.0M params

Strong dropout (0.5 on stem, 0.3 on head) prevents overfitting on limited synthetic data.

Trained for up to 50 epochs on synthetic unseen data (2000 images/class) with:
- **Data augmentation**: RandomCrop(32,4) + RandomHorizontalFlip applied at train time
- **Optimizer**: AdamW (lr=0.001, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau on validation accuracy
- **Early stopping**: patience=10 on validation accuracy
- **Mixup** (configurable): alpha=0.2, 50% probability per batch
- **Label smoothing**: 0.1

**ZSL is true ZSL**: The classifier is trained exclusively on generated samples and evaluated on real unseen-class images. It has never seen real images of those classes during training.

---

### 6. GZSL Classifier (`evaluation/gzsl_eval.py`)

Generalized Zero-Shot Learning extends ZSL to also classify seen classes:

- Same backbone as ZSL classifier but outputs `num_seen + num_unseen` (100) classes
- Trained on real seen-class images (from training set) + synthetic unseen-class images
- **CalibratedClassifier** wrapper applies learned temperature scaling:
  - `temperature_s` for seen-class logits
  - `temperature_u` for unseen-class logits
  - `bias_shift` for overall unseen bias adjustment
- Uses `ConcatDataset` + `_LabelShiftedDataset` (module-level, pickle-safe) for efficient joint training
- Evaluated on held-out seen validation set (not training set) and unseen test set
- Reports: Seen accuracy, Unseen accuracy, Harmonic Mean (H)

---

### 7. Evaluation

| Metric | How |
|---|---|
| **FID** | `torch-fidelity` between saved real/fake PNGs (20K samples per eval) |
| **IS** | Inception Score via torch-fidelity |
| **KID** | Kernel ID via torch-fidelity |
| **ZSL Top-1** | % correct on real unseen test set |
| **ZSL Top-5** | % of top-5 containing correct label |
| **ZSL Mean Class** | Per-class accuracy averaged |
| **GZSL Seen** | % correct on held-out seen validation set |
| **GZSL Unseen** | % correct on real unseen test set (labels shifted by num_seen) |
| **GZSL H** | `2·seen·unseen / (seen + unseen)` — primary GZSL metric |

---

## File Map

```
configs/config.yaml          ← single source of truth for all hyperparameters
.pre-commit-config.yaml      ← code quality hooks (black, flake8, mypy)
models/generator.py          ← G(z, labels, embeddings) → images
models/discriminator.py      ← D(images, labels, embeddings) → score
models/zsl_classifier.py     ← CNN trained on synthetic unseen data
training/losses.py           ← WGAN-GP loss + gradient penalty + feature matching
training/trainer.py          ← epoch loop, EMA, checkpointing, AMP, resume
evaluation/gan_eval.py       ← FID/IS/KID via torch-fidelity
evaluation/zsl_eval.py       ← synthetic dataset, classifier training, ZSL metrics
evaluation/gzsl_eval.py      ← GZSL training, calibrated stacking, seen+unseen eval
utils/embeddings.py          ← CLIP / CLIP ensemble / GloVe embedding extraction ("both" mode)
utils/data_loader.py         ← CIFAR-100 with seen/unseen split + aligned train/val transforms
utils/metrics.py             ← MetricsTracker (losses, FID history)
utils/visualization.py       ← all plotting (curves, grids, confusion matrix, experiment summary)
main.py                      ← orchestrates everything (config validation, --resume)
app.py                       ← Gradio demo (config-driven backbone)
```

## Config Validation

`validate_config(config)` in `main.py` checks all required config keys at startup and exits with clear error messages if any are missing. Keys validated include:

- `paths`: data_root, results_dir, checkpoints_dir, cache_dir
- `dataset`: num_classes, seen_classes, unseen_classes
- `embeddings`: type (and clip_model if type is clip/clip_ensemble)
- `model.generator`: nz, ngf, nc, semantic_proj_dim
- `model.discriminator`: ndf, nc, semantic_proj_dim
- `model.classifier`: backbone
- `training`: num_epochs, batch_size, lr_g, lr_d, n_critic, lambda_gp
