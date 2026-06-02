# ZSL-cWGAN-GP Architecture

## Overview

ZSL-cWGAN-GP is a conditional generative adversarial network that synthesizes CIFAR-100 images conditioned on semantic text embeddings (CLIP or GloVe). The trained generator produces images for unseen classes, which are used to train a zero-shot classifier.

---

## Data Flow

```
CIFAR-100 (100 classes)
    │
    ├── 80 seen classes ──► Generator ──► Synthetic unseen images
    │                          │                  │
    │                          ▼                  ▼
    │                    Discriminator      ZSL Classifier
    │                    (real vs fake)     (trained on synthetic,
    │                                       evaluated on real unseen)
    │
    └── 20 unseen classes ──► Held out during GAN training
                              Used only for ZSL evaluation
```

---

## Component Pipeline

### 1. Embeddings (`utils/embeddings.py`)

```
CIFAR-100 class names
    │
    ├── CLIPTextEmbedder: "openai/clip-vit-base-patch32" → 512-dim
    │   - Uses 5 prompt templates per class, averages them
    │   - L2-normalizes output
    │
    └── GloVeEmbedder: glove.6B.300d.txt → 300-dim
        - Falls back to word averaging, then random if missing
```

Both produce a tensor `[num_classes, embedding_dim]` that the GAN indexes by label at runtime.

---

### 2. Generator (`models/generator.py`)

```
z (noise) [B, 128] ───┐
                       ├── concat ──► Linear ──► reshape ──► Conv2D ──► Attn ──► Conv2D ──► Conv2D ──► Tanh ──► [B, 3, 32, 32]
semantic_emb[labels]   │            8*4*4           4x4         4→8      8x8    8→16      16→32        [-1, 1]
[B, 512] ───► MLP ────┘
            256-256
           (LeakyReLU+Dropout)
```

**Semantic projection**: 2-layer MLP (512→256→256) with LeakyReLU(0.2) and Dropout(0.2).

**Self-attention**: SAGAN-style block at 8x8 resolution with spectral norm Q/K/V projections. Initialized with `gamma=0` so training starts as identity and gradually learns global dependencies.

**Generation**: Starts at 4×4, nearest-neighbor upsamples to 8×8 → 16×16 → 32×32 with spectral norm conv layers and BatchNorm.

**Weight init**: Orthogonal for all conv/linear layers.

---

### 3. Discriminator (`models/discriminator.py`)

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
    G_loss = -E[D(fake)]
    update G with Adam(β₁=0.0, β₂=0.9)
```

**Key parameters** (from `configs/config.yaml`):

| Parameter | Value | Purpose |
|---|---|---|
| `λ_gp` | 10 | Gradient penalty weight |
| `n_critic` | 5 | D updates per G update |
| `β₁` | 0.0 | Adam momentum (standard for WGAN) |
| `β₂` | 0.9 | Adam second moment |
| `lr_D` | 0.0001 | Discriminator learning rate |
| `lr_G` | 0.0001 | Generator learning rate |
| `grad_clip` | 1.0 | Global gradient norm clipping |

**Mixed precision**: Forward passes run in FP16 via `torch.amp.autocast("cuda")`; `GradScaler("cuda")` prevents underflow. `scaler.update()` is called after each D step inside the n_critic loop (required for AMP state machine with multiple optimizer steps per batch).

**Regularization**:
- Spectral normalization on all conv/linear layers in both G and D
- Gradient penalty on interpolates between real and fake
- Dropout (0.2) in G's semantic projection
- BatchNorm in G's upsampling layers

---

### 5. Zero-Shot Classifier (`models/zsl_classifier.py`)

A lightweight CNN trained on synthetic images from the generator:

```
input [3, 32, 32]
    └──► 4× Conv2D (64→128→256→512) + BatchNorm + ReLU
         Downsampled via stride-2 convolutions (32→16→8→4→1)
    └──► AdaptiveAvgPool2d(1)
    └──► MLP: 512 → 256 → num_unseen_classes
         (Dropout(0.5) + ReLU + Dropout(0.3))
```

Strong dropout (0.5 on stem, 0.3 on head) prevents overfitting on limited synthetic data.

Trained for up to 50 epochs on synthetic unseen data (2000 images/class) with:
- **Data augmentation**: RandomCrop(32,4) + RandomHorizontalFlip applied at train time
- **Optimizer**: AdamW (lr=0.001, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau on validation accuracy
- **Early stopping**: patience=10 on validation accuracy

**ZSL is true ZSL**: The classifier is trained exclusively on generated samples and evaluated on real unseen-class images. It has never seen real images of those classes during training.

---

### 6. Evaluation

| Metric | How |
|---|---|
| **FID** | `torch-fidelity` between saved real/fake PNGs (20K samples per eval) |
| **IS** | Inception Score via torch-fidelity |
| **KID** | Kernel ID via torch-fidelity |
| **ZSL Top-1** | % correct on real unseen test set |
| **ZSL Top-5** | % of top-5 containing correct label |
| **ZSL Mean Class** | Per-class accuracy averaged |

---

## File Map

```
configs/config.yaml        ← single source of truth for all hyperparameters
models/generator.py        ← G(z, labels, embeddings) → images
models/discriminator.py    ← D(images, labels, embeddings) → score
models/zsl_classifier.py   ← CNN trained on synthetic unseen data
training/losses.py         ← WGAN-GP loss + gradient penalty
training/trainer.py        ← epoch loop, checkpointing, AMP
evaluation/gan_eval.py     ← FID/IS/KID via torch-fidelity
evaluation/zsl_eval.py     ← synthetic dataset, classifier training, metrics
utils/embeddings.py        ← CLIP / GloVe embedding extraction
utils/data_loader.py       ← CIFAR-100 with seen/unseen split + caching
utils/metrics.py           ← MetricsTracker (losses, FID history)
utils/visualization.py     ← all plotting (curves, grids, confusion matrix)
main.py                    ← orchestrates everything
app.py                     ← Gradio demo
```
