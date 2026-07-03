# ZSL-cWGAN-GP Architecture

## Overview

ZSL-cWGAN-GP is a conditional generative adversarial network that synthesizes CIFAR-100 images conditioned on CLIP semantic text embeddings. The trained generator produces images for unseen classes, which are used to train a zero-shot classifier. An optional Generalized Zero-Shot Learning (GZSL) branch additionally classifies seen classes with calibrated temperature scaling to reduce seen-bias.

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
    └── CLIPTextEmbedder: "openai/clip-vit-large-patch14" → 768-dim
        - Uses 5 prompt templates per class, averages them
        - L2-normalizes output
        - Cached to cache/embeddings_clip.pkl
```

This produces a tensor `[num_classes, embedding_dim]` that the GAN indexes by label at runtime.

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

**Semantic projection**: 2-layer MLP (768→256→256 for CLIP) with LeakyReLU(0.2), Dropout(0.2), and spectral norm.

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

**Bias-free critic (L1-F18)**: all convs and linears are `bias=False`. Spectral norm and the gradient penalty constrain weights and gradients, but neither constrains the critic's *absolute* output — learnable biases (additive constants invisible to both constraints) compounded through the LeakyReLU stack and inflated D(real)≈D(fake) to +1.7M with W frozen at ~-10, until float32 overflow collapsed training (validated: final G -1.67M → -9.8 after the fix). Bias-free pins D(0)=0, so |D(x)| ≤ ‖x‖·Lipschitz is bounded for the whole run.

**Lipschitz mechanisms (L1-F11 note)**: every conv, the output conv, and `embed_output` are spectral-normed, AND the loss adds a WGAN-GP penalty — two redundant Lipschitz constraints. This is a deliberate, non-standard choice vs. the plain-conv f-CLSWGAN baseline (which relies on the GP alone). The two interact: SN keeps each layer's operator norm at 1 (bounded scores), the GP enforces the global 1-Lipschitz constraint on the interpolated line. Keep them together — removing either one changes the Lipschitz regime and requires re-tuning (LR, λ_gp, grad_clip) and is out of scope for a paper ablation unless that re-tuning is reported.

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

**Key parameters** (from `src/configs/config.yaml`):

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

### 5. Zero-Shot Classifier (`src/models/zsl_classifier.py`)

A pluggable backbone classifier trained on synthetic images from the generator:

```
input [3, 32, 32]
    └──► Backbone (ResNet-18 / EfficientNet-B0 / Custom CNN)
    └──► Classifier Head:
           Dropout(0.5) → Linear(feat_dim→512) → ReLU → Dropout(0.3) → Linear(512→num_classes)
```

**Backbone options** (randomly initialized by default — `model.classifier.pretrained: false`, since classifier inputs are [-1, 1]-normalized while ImageNet weights expect (0.485, 0.229, 0.225); set `true` only if you also apply ImageNet normalization to classifier inputs):
- **ResNet-18** (default): ~11.4M params, 512-dim features
- **EfficientNet-B0**: ~4.7M params, 1280-dim features
- **Custom CNN**: 4× Conv2D (64→128→256→512) + AdaptiveAvgPool2d, ~3.0M params

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

### 6. GZSL Classifier (`src/evaluation/gzsl_eval.py`)

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
| **FID** | `torch-fidelity` between saved real/fake PNGs (4K samples per eval, equal counts) |
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
src/configs/config.yaml        ← single source of truth for all hyperparameters
.pre-commit-config.yaml        ← code quality hooks (black, flake8, mypy)
src/models/generator.py        ← G(z, labels, embeddings) → images
src/models/discriminator.py    ← D(images, labels, embeddings) → score
src/models/zsl_classifier.py   ← CNN trained on synthetic unseen data
src/training/losses.py         ← WGAN-GP loss + gradient penalty + feature matching
src/training/trainer.py        ← epoch loop, EMA, checkpointing, AMP, resume
src/evaluation/gan_eval.py     ← FID/IS/KID via torch-fidelity
src/evaluation/zsl_eval.py     ← synthetic dataset, classifier training, ZSL metrics
src/evaluation/gzsl_eval.py    ← GZSL training, calibrated stacking, seen+unseen eval
src/utils/embeddings.py        ← CLIP text embedding extraction (cached)
src/utils/data_loader.py       ← CIFAR-100 with seen/unseen split + aligned train/val transforms
src/utils/metrics.py           ← MetricsTracker (losses, FID history)
src/utils/visualization.py     ← all plotting (curves, grids, confusion matrix, experiment summary)
src/main.py                    ← orchestrates everything (config validation, --resume)
src/app.py                     ← Gradio demo (config-driven backbone)
test/test_training.py          ← 5-epoch sanity check
test/test_clip.py              ← CLIP embedding test suite
```

## Config Validation

`validate_config(config)` in `src/main.py` checks all required config keys at startup and exits with clear error messages if any are missing. Keys validated include:

- `paths`: data_root, results_dir, checkpoints_dir, cache_dir
- `dataset`: num_classes, seen_classes
- `embeddings`: type (must be "clip") and clip_model
- `model.generator`: nz, ngf, nc, semantic_proj_dim
- `model.discriminator`: ndf, nc, semantic_proj_dim
- `model.classifier`: backbone
- `training`: num_epochs, batch_size, lr_g, lr_d, n_critic, lambda_gp
