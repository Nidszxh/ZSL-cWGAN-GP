# Zero-Shot Learning with Conditional WGAN-GP (ZSL-cWGAN-GP)

A PyTorch implementation of **Zero-Shot Learning (ZSL)** using a **Conditional Wasserstein GAN with Gradient Penalty (cWGAN-GP)** on **CIFAR-100** with CLIP text embeddings.

The model learns to generate realistic images for **unseen classes** by conditioning a WGAN-GP on **semantic embeddings** (CLIP text embeddings or GloVe). A classifier trained on these synthetic samples can then recognize categories it was never trained on.

---

## Key Features

- **Semantic Conditioning:** CLIP text embeddings (512-dim) or GloVe (300-dim)
- **Conditional WGAN-GP:** Stable training with gradient penalty and spectral normalization
- **Zero-Shot Classifier:** Trained on synthetically generated unseen-class images
- **FID / IS / KID Evaluation:** Monitors generation quality during training
- **Interactive Demo:** Gradio web interface for ZSL predictions
- **TensorBoard Logging:** Loss curves, metrics, and generated samples

---

## Installation

```bash
pip install -r requirements.txt
```

> **GPU is strongly recommended** — CPU training will be extremely slow.

---

## Quick Start

### Train (CLIP-based, 150 epochs)
```bash
python main.py
```

### View training curves
```bash
tensorboard --logdir=runs
```

### Launch interactive demo
```bash
python app.py
```

### Quick sanity check (5 epochs)
```bash
python test_training.py
```

---

## Usage

### Full Training Pipeline

`python main.py` drives the modular pipeline:

1. Reads `configs/config.yaml`
2. Downloads CIFAR-100 and CLIP model (on first run)
3. Splits 100 classes into 80 seen / 20 unseen (cached)
4. Trains the conditional WGAN-GP with periodic FID evaluation
5. Generates synthetic images for unseen classes
6. Trains and evaluates a Zero-Shot classifier
7. Saves visualizations and experiment summary

**Config** is controlled via `configs/config.yaml`:
```yaml
embeddings:
  type: "clip"                        # "clip" (512d) or "glove" (300d)
training:
  num_epochs: 150                     # Total training epochs
  batch_size: 128                     # 128 fits 8GB VRAM (501 img/s)
  lr_g: 0.0001
  lr_d: 0.0001
  eval_interval: 10                   # FID evaluation every N epochs
  early_stopping_patience: 20
  n_critic: 5                         # D updates per G update
  lambda_gp: 10                       # Gradient penalty coefficient
```

### Quick Test
```bash
python test_training.py     # 5-epoch CLIP test
```

### CLIP Embedding Tests
```bash
python test_clip.py         # validates CLIP integration
```

### Legacy GloVe Training
```bash
python ZSLcWGAN-GP.py       # original monolithic script
```

### Interactive Demo
After training, launch the Gradio demo:
```bash
python app.py
```

Upload an image from one of the 20 unseen CIFAR-100 classes and get top-5 ZSL predictions.

---

## Project Structure

```
ZSL-cWGAN-GP/
├── main.py                  # Unified entrypoint (CLIP pipeline)
├── app.py                   # Gradio demo
├── ZSLcWGAN-GP.py           # Legacy monolithic script (GloVe)
├── test_training.py         # Quick 5-epoch sanity check
├── test_clip.py             # CLIP embedding tests
├── configs/
│   └── config.yaml          # All hyperparameters
├── models/
│   ├── generator.py         # Conditional generator
│   ├── discriminator.py     # Projection discriminator
│   └── zsl_classifier.py    # ZSL classifier
├── training/
│   ├── trainer.py           # Training loop
│   └── losses.py            # WGAN-GP loss functions
├── evaluation/
│   ├── gan_eval.py          # FID / IS / KID evaluation
│   └── zsl_eval.py          # ZSL classifier training & eval
├── utils/
│   ├── embeddings.py        # CLIP + GloVe embedders
│   ├── data_loader.py       # CIFAR-100 with seen/unseen split
│   ├── metrics.py           # Metrics tracker
│   └── visualization.py     # Plotting utilities
├── data/                    # CIFAR-100 (auto-downloaded)
├── cache/                   # Embedding cache, class split
├── checkpoints/             # Saved model weights
├── results/                 # Output images, plots, logs
│   ├── fake/                # Generated samples for FID
│   ├── real/                # Real samples for FID
│   ├── unseen_synthetic/    # Unseen class generations
│   └── ...
└── runs/                    # TensorBoard logs
```

---

## Outputs

After training, results are saved under `results/`:

| File | Description |
|---|---|
| `training_curves.png` | Generator/discriminator loss curves |
| `metrics_progress.png` | FID and Inception Score over epochs |
| `generated_samples_grid.png` | Grid of generated seen-class images |
| `zsl_confusion_matrix.png` | Confusion matrix on unseen classes |
| `zsl_class_accuracy.png` | Per-class ZSL accuracy |
| `experiment_summary.png` | Combined summary visualization |

Sample console output:
```
Epoch 10/150 - FID: 87.84 ✓ New best FID: 87.84
Epoch 20/150 - FID: 72.54 ✓ New best FID: 72.54
...
Epoch 50/150 - FID: 58.84 ✓ New best FID: 58.84
ZSL Top-1 Accuracy: 4.45%
```

---

## Configuration Reference

Key settings in `configs/config.yaml`:

| Key | Default | Description |
|---|---|---|
| `embeddings.type` | `"clip"` | `"clip"`, `"glove"`, or `"both"` |
| `training.num_epochs` | `150` | Total training epochs |
| `training.batch_size` | `128` | Batch size (benchmarked up to 160 on 8GB) |
| `training.lr_g` | `0.0001` | Generator learning rate |
| `training.lr_d` | `0.0001` | Discriminator learning rate |
| `training.n_critic` | `5` | D updates per G update |
| `training.lambda_gp` | `10` | Gradient penalty coefficient |
| `training.eval_interval` | `10` | FID eval frequency (epochs) |
| `training.early_stopping_patience` | `20` | Patience before early stop |
| `evaluation.fid_samples` | `20000` | Samples for FID calculation |
| `evaluation.synthetic_samples_per_class` | `2000` | Synthetic images per unseen class |

---

## Auto-Downloads

- **CIFAR-100** — via `torchvision.datasets`
- **CLIP model** — `openai/clip-vit-base-patch32` via HuggingFace `transformers`
- **GloVe 6B 300d** — ~822MB zip from Stanford (only when using GloVe)

---

## License

MIT

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Torch-Fidelity](https://github.com/toshas/torch-fidelity)
