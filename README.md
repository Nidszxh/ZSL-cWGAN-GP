# Zero-Shot Learning with Conditional WGAN-GP (ZSL-cWGAN-GP)

A PyTorch implementation of **Zero-Shot Learning (ZSL)** using a **Conditional Wasserstein GAN with Gradient Penalty (cWGAN-GP)**.  
This project learns to generate realistic images for unseen classes by combining **semantic word embeddings (GloVe)** with adversarial image synthesis on **CIFAR-100**.

---

## Overview

This project tackles **Zero-Shot Learning**, where a model must recognize categories it has never seen before.  
By conditioning a **Wasserstein GAN** on **semantic embeddings**, it can generate synthetic samples for *unseen classes*, which are then used to train a classifier.

### Key Features
- **Semantic Conditioning:** Uses GloVe embeddings to integrate semantic meaning into generation.  
- **Conditional WGAN-GP:** Stable training with gradient penalty and spectral normalization.  
- **Zero-Shot Classifier:** Trains a classifier on generated unseen data.  
- **FID Evaluation:** Computes Fréchet Inception Distance to monitor realism.  
- **Automatic Visualization:** Saves grids, FID curves, confusion matrices, and summary images.  

---

## Installation

### Requirements
Ensure Python ≥ 3.9 and the following dependencies:

```bash
pip install torch torchvision numpy matplotlib tqdm requests torch-fidelity
```

> 💡 GPU is strongly recommended for training.

---

## ⚙️ Configuration

All hyperparameters are defined in the `config` dictionary:

```python
config = {
    'batch_size': 64,
    'nz': 128,
    'ngf': 64,
    'ndf': 64,
    'num_classes': 100,
    'lr_g': 0.0001,
    'lr_d': 0.0002,
    'lambda_gp': 10,
    'n_critic': 5,
    'embedding_dim': 300,
    'semantic_proj_dim': 256
}
```

Modify as needed before running.

---

## Usage

### Run Training
```bash
python ZSLcWGAN-GP.py
```

This will:
- Download CIFAR-100 and GloVe (if missing)
- Train on 80 seen classes
- Compute FID periodically
- Save checkpoints and sample outputs automatically

### Evaluate
After training, the script:
- Loads the best checkpoint (lowest FID)
- Generates unseen class images
- Evaluates a Zero-Shot classifier
- Saves results under `results/`

---

## Project Structure

```
ZSLcWGAN-GP/
├── ZSLcWGAN-GP.py              # Main script
├── data/                       # CIFAR-100 dataset (auto-downloaded)
├── checkpoints/                # Saved model weights
├── results/
│   ├── fake/                   # Generated fake samples
│   ├── real/                   # Real samples for FID
│   ├── unseen_synthetic/       # Unseen class generations
│   ├── fid_progress.png        # FID curve
│   ├── training_curves.png     # G/D losses
│   ├── zsl_class_accuracy.png  # Class-level ZSL accuracy
│   └── exp_summary.png         # Summary visualization
└── glove.6B.300d.txt           # GloVe embeddings
```

---

## Outputs

After execution, results are saved in `results/`:

- **Generated Samples:** `generated_samples_grid.png`  
- **FID Curve:** `fid_progress.png`  
- **Zero-Shot Class Accuracy:** `zsl_class_accuracy.png`  
- **Confusion Matrix:** `zsl_confusion_matrix.png`  
- **Experiment Summary:** `exp_summary.png`  

Sample console output:
```
Epoch 5/10 - FID: 45.12 ✓ New best FID: 45.12
Epoch 10/10 - FID: 42.88 × FID did not improve
Final FID: 42.15
ZSL Accuracy: 37.64%
```

---

## Evaluate Separately

If you want to re-run ZSL evaluation:

```python
from ZSLcWGAN_GP import evaluate_zsl_performance
metrics = evaluate_zsl_performance()
print(metrics["accuracy"])
```

---

## License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it with attribution.

---

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [Stanford GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)
- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Torch-Fidelity for FID computation](https://github.com/toshas/torch-fidelity)

