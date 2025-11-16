import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import matplotlib.pyplot as plt
import numpy as np
import os
import string
import requests
import zipfile
from torch_fidelity import calculate_metrics
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
import json
import pickle
import warnings
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import random

warnings.filterwarnings('ignore')

# CONFIGURATION

config = {
    # Model architecture
    'batch_size': 128,
    'nc': 3,
    'nz': 128,
    'ngf': 64,
    'ndf': 64,
    'num_classes': 100,
    
    # Training
    'num_epochs': 50,
    'lr_g': 0.0001,
    'lr_d': 0.0004,
    'beta1': 0.0,
    'beta2': 0.9,
    'lambda_gp': 10,
    'n_critic': 5,
    'grad_clip': 1.0,
    
    # Semantic embeddings
    'embedding_dim': 300,
    'semantic_proj_dim': 256,
    
    # Early stopping
    'early_stopping_patience': 15,
    'fid_eval_interval': 5,
    
    # Paths
    'data_root': './data',
    'glove_path': 'glove.6B.300d.txt',
    'results_dir': 'results',
    'checkpoints_dir': 'checkpoints',
    'cache_dir': 'cache',
    
    # ZSL
    'seen_classes_count': 80,
    'unseen_classes_count': 20,
    'synthetic_samples_per_class': 500,
    
    # Other
    'seed': 42,
    'num_workers': 4,
}

# SEED & DEVICE SETUP

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config['seed'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# DIRECTORY SETUP

for dir_path in [config['results_dir'], config['checkpoints_dir'], config['cache_dir'],
                 f"{config['results_dir']}/fake", f"{config['results_dir']}/real",
                 f"{config['results_dir']}/unseen_synthetic"]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# TensorBoard writer
writer = SummaryWriter(f"{config['results_dir']}/runs")

# DATA PREPARATION

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CLASS SPLIT WITH PERSISTENCE

split_file = Path(config['cache_dir']) / 'class_split.json'

if split_file.exists():
    print("Loading existing class split...")
    with open(split_file, 'r') as f:
        split_data = json.load(f)
        seen_classes = np.array(split_data['seen'])
        unseen_classes = np.array(split_data['unseen'])
else:
    print("Creating new class split...")
    set_seed(config['seed'])
    all_classes = np.arange(config['num_classes'])
    np.random.shuffle(all_classes)
    seen_classes = all_classes[:config['seen_classes_count']]
    unseen_classes = all_classes[config['seen_classes_count']:]
    
    with open(split_file, 'w') as f:
        json.dump({
            'seen': seen_classes.tolist(),
            'unseen': unseen_classes.tolist()
        }, f, indent=2)
    print(f"Class split saved to {split_file}")

print(f"Seen classes: {len(seen_classes)}, Unseen classes: {len(unseen_classes)}")

# FILTERED CIFAR100 DATASET

class FilteredCIFAR100(datasets.CIFAR100):
    def __init__(self, *args, allowed_classes=None, transform=None, **kwargs):
        super().__init__(*args, transform=transform, **kwargs)
        if allowed_classes is not None:
            mask = np.isin(self.targets, allowed_classes)
            indices = np.where(mask)[0]
            self.data = self.data[indices]
            self.targets = [self.targets[i] for i in indices]
            self.org_to_new = {old_cls: i for i, old_cls in enumerate(sorted(set(self.targets)))}
            self.targets = [self.org_to_new[target] for target in self.targets]

# Create datasets
train_dataset = FilteredCIFAR100(
    root=config['data_root'], train=True, download=True,
    transform=transform_train, allowed_classes=seen_classes
)

# Split training data
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# DataLoaders
num_workers = min(config['num_workers'], os.cpu_count() or 2)
train_loader = DataLoader(
    train_subset, batch_size=config['batch_size'], shuffle=True,
    num_workers=num_workers, pin_memory=True, persistent_workers=True
)
val_loader = DataLoader(
    val_subset, batch_size=config['batch_size'], shuffle=False,
    num_workers=num_workers, pin_memory=True, persistent_workers=True
)

# Save class mappings
seen_class_to_new_idx = train_dataset.org_to_new
new_idx_to_seen_class = {v: k for k, v in seen_class_to_new_idx.items()}
num_seen_classes = len(seen_class_to_new_idx)

# GLOVE EMBEDDINGS WITH CACHING

def load_glove_embeddings(glove_file='glove.6B.300d.txt'):
    """Load GloVe embeddings with caching"""
    cache_file = Path(config['cache_dir']) / 'glove_cache.pkl'
    
    if cache_file.exists():
        print(f"Loading cached GloVe embeddings from {cache_file}...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    if not Path(glove_file).exists():
        print("Downloading GloVe embeddings...")
        glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
        r = requests.get(glove_url, stream=True, allow_redirects=True)
        
        with open('glove.6B.zip', 'wb') as f:
            total_size = int(r.headers.get('content-length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        with zipfile.ZipFile('glove.6B.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        
        print("GloVe embeddings downloaded and extracted")
    
    print(f"Loading GloVe embeddings from {glove_file}...")
    embeddings_dict = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading GloVe"):
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embeddings_dict[word] = vector
    
    print(f"Loaded {len(embeddings_dict)} GloVe word vectors")
    
    # Cache for next time
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    print(f"Cached embeddings to {cache_file}")
    
    return embeddings_dict

def clean_label(label):
    """Remove punctuation and convert to lowercase"""
    return label.translate(str.maketrans('', '', string.punctuation)).lower()

def create_semantic_embeddings(class_names, glove_embeddings, embedding_dim=300):
    """Create semantic embeddings for all classes"""
    semantic_embeddings = {}
    
    for class_idx, class_name in enumerate(class_names):
        cleaned = clean_label(class_name)
        
        if cleaned in glove_embeddings:
            semantic_embeddings[class_idx] = glove_embeddings[cleaned]
        else:
            words = cleaned.split()
            found_vectors = [glove_embeddings[word] for word in words if word in glove_embeddings]
            
            if found_vectors:
                semantic_embeddings[class_idx] = np.mean(found_vectors, axis=0)
            else:
                print(f"Class '{class_name}' not found in GloVe, using random embedding")
                rng = np.random.RandomState(42 + class_idx)
                semantic_embeddings[class_idx] = rng.normal(scale=0.6, size=embedding_dim).astype(np.float32)
    
    return semantic_embeddings

# Load GloVe and create embeddings
glove_embeddings = load_glove_embeddings(config['glove_path'])
cifar100_classes = datasets.CIFAR100(root=config['data_root'], download=True).classes

semantic_embeddings = create_semantic_embeddings(
    cifar100_classes, glove_embeddings, embedding_dim=config['embedding_dim']
)

# Convert to tensors
all_semantic_embeddings = torch.tensor(
    [semantic_embeddings[cls] for cls in range(config['num_classes'])],
    dtype=torch.float32, device=device
)

seen_semantic_embeddings = torch.tensor(
    [semantic_embeddings[cls] for cls in seen_classes],
    dtype=torch.float32, device=device
)

unseen_semantic_embeddings = torch.tensor(
    [semantic_embeddings[cls] for cls in unseen_classes],
    dtype=torch.float32, device=device
)

# GENERATOR WITH IMPROVEMENTS

class Generator(nn.Module):
    def __init__(self, nz=128, ngf=64, nc=3, semantic_dim=300, semantic_proj_dim=256):
        super(Generator, self).__init__()
        
        # Semantic projection with dropout
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, semantic_proj_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(semantic_proj_dim, semantic_proj_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        combined_dim = nz + semantic_proj_dim
        
        self.project = nn.Sequential(
            spectral_norm(nn.Linear(combined_dim, ngf * 8 * 4 * 4)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),
            nn.Upsample(scale_factor=2, mode='nearest'),
            spectral_norm(nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1)),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            spectral_norm(nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1)),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            spectral_norm(nn.Conv2d(ngf * 2, nc, 3, 1, 1)),
            nn.Tanh()
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z, labels, semantic_embeddings):
        sem_features = self.semantic_proj(semantic_embeddings[labels])
        x = torch.cat([z, sem_features], dim=1)
        x = self.project(x).view(-1, config['ngf'] * 8, 4, 4)
        return self.main(x)

# DISCRIMINATOR

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, semantic_dim=300):
        super(Discriminator, self).__init__()
        
        self.semantic_proj = nn.Sequential(
            spectral_norm(nn.Linear(semantic_dim, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(256, 256)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(nc, ndf, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.output = spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0))
        self.embed_output = spectral_norm(nn.Linear(256, ndf * 8))
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)) and not isinstance(m, nn.BatchNorm2d):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, labels, semantic_embeddings):
        sem_features = self.semantic_proj(semantic_embeddings[labels])
        h = self.features(x)
        
        output = self.output(h).squeeze()
        
        projection = self.embed_output(sem_features).view(-1, config['ndf'] * 8, 1, 1)
        h_pooled = F.adaptive_avg_pool2d(h, 1).view(-1, config['ndf'] * 8)
        cond_output = torch.sum(projection.squeeze() * h_pooled, dim=1)
        
        return output + cond_output

# GRADIENT PENALTY

def compute_gradient_penalty(D, real, fake, labels, semantic_embeddings):
    """Compute WGAN-GP gradient penalty"""
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    
    d_interpolates = D(interpolates, labels, semantic_embeddings)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

# EVALUATION UTILITIES

def save_real_images_for_eval(dataset, num_samples=5000, save_dir=None):
    """Save real images for FID evaluation"""
    if save_dir is None:
        save_dir = f"{config['results_dir']}/real"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    loader = DataLoader(dataset, batch_size=100, shuffle=False)
    
    count = 0
    print(f"Saving {num_samples} real images for evaluation...")
    
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Saving real images"):
            images = (images + 1) / 2
            
            for img in images:
                if count >= num_samples:
                    break
                vutils.save_image(img, Path(save_dir) / f"real_{count:05d}.png")
                count += 1
                
            if count >= num_samples:
                break
    
    print(f"Saved {count} real images to {save_dir}")

def save_fake_images_for_eval(generator, epoch, num_samples=2000, save_dir=None):
    """Save generated images for FID evaluation"""
    if save_dir is None:
        save_dir = f"{config['results_dir']}/fake_epoch{epoch}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    generator.eval()
    
    print(f"Generating {num_samples} fake images for evaluation...")
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, config['batch_size']), desc="Generating images"):
            curr_batch_size = min(config['batch_size'], num_samples - i)
            if curr_batch_size <= 0:
                break
            
            z = torch.randn(curr_batch_size, config['nz'], device=device)
            labels = torch.randint(0, num_seen_classes, (curr_batch_size,), device=device)
            
            fake_images = generator(z, labels, seen_semantic_embeddings).cpu()
            fake_images = (fake_images + 1) / 2
            
            for j, img in enumerate(fake_images):
                idx = i + j
                vutils.save_image(img, Path(save_dir) / f"fake_{idx:05d}.png")
    
    generator.train()
    print(f"Saved images to {save_dir}")
    return save_dir

def calculate_fid(real_dir, fake_dir):
    """Calculate FID score with additional metrics"""
    print(f"Calculating metrics between {real_dir} and {fake_dir}")
    try:
        metrics = calculate_metrics(
            input1=real_dir,
            input2=fake_dir,
            cuda=torch.cuda.is_available(),
            fid=True,
            isc=True,
            kid=True,
            verbose=False
        )
        return {
            'fid': metrics['frechet_inception_distance'],
            'is_mean': metrics.get('inception_score_mean', 0),
            'is_std': metrics.get('inception_score_std', 0),
            'kid_mean': metrics.get('kernel_inception_distance_mean', 0)
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {'fid': float('inf'), 'is_mean': 0, 'is_std': 0, 'kid_mean': 0}

# TRAINING TRACKER

class TrainingTracker:
    def __init__(self, log_dir=None):
        self.log_dir = log_dir or config['results_dir']
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        self.g_losses = []
        self.d_losses = []
        self.w_distances = []
        self.gp_values = []
        self.metrics_history = []
        
        self.best_fid = float('inf')
        self.epochs_without_improv = 0
    
    def update(self, g_loss, d_loss, w_dist, gp):
        self.g_losses.append(g_loss)
        self.d_losses.append(d_loss)
        self.w_distances.append(w_dist)
        self.gp_values.append(gp)
    
    def update_metrics(self, metrics, epoch, netG, netD):
        """Update metrics and save best model"""
        fid_score = metrics['fid']
        self.metrics_history.append((epoch, metrics))
        
        # Log to TensorBoard
        writer.add_scalar('Metrics/FID', fid_score, epoch)
        writer.add_scalar('Metrics/IS_mean', metrics['is_mean'], epoch)
        writer.add_scalar('Metrics/KID', metrics['kid_mean'], epoch)
        
        if fid_score < self.best_fid:
            self.best_fid = fid_score
            self.epochs_without_improv = 0
            
            torch.save({
                'generator': netG.state_dict(),
                'discriminator': netD.state_dict(),
                'epoch': epoch,
                'metrics': metrics
            }, Path(config['checkpoints_dir']) / 'best_model.pth')
            
            print(f"✓ New best FID: {fid_score:.2f} | IS: {metrics['is_mean']:.2f}±{metrics['is_std']:.2f} - Model saved")
            return True
        else:
            self.epochs_without_improv += 1
            print(f"× FID: {fid_score:.2f} vs best {self.best_fid:.2f}")
            return False
    
    def should_stop_early(self, patience):
        return self.epochs_without_improv >= patience
    
    def plot_losses(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.g_losses, label='G Loss', alpha=0.7)
        ax1.plot(self.d_losses, label='D Loss', alpha=0.7)
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Generator and Discriminator Losses')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.w_distances, label='Wasserstein Distance', alpha=0.7)
        ax2.plot(self.gp_values, label='Gradient Penalty', alpha=0.7)
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.set_title('Wasserstein Distance and Gradient Penalty')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(self.log_dir) / 'training_curves.png', dpi=150)
        plt.close()
    
    def plot_metrics(self):
        if not self.metrics_history:
            return
        
        epochs = [m[0] for m in self.metrics_history]
        fids = [m[1]['fid'] for m in self.metrics_history]
        is_means = [m[1]['is_mean'] for m in self.metrics_history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(epochs, fids, 'o-', label='FID Score', markersize=6)
        ax1.axhline(y=self.best_fid, color='r', linestyle='--', label=f'Best: {self.best_fid:.2f}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('FID Score')
        ax1.set_title('FID Score During Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs, is_means, 'o-', color='green', label='Inception Score', markersize=6)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Inception Score')
        ax2.set_title('Inception Score During Training')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(self.log_dir) / 'metrics_progress.png', dpi=150)
        plt.close()

# SYNTHETIC DATASET FOR ZSL

class SyntheticDataset(Dataset):
    """On-the-fly synthetic image generation"""
    def __init__(self, generator, num_samples, num_classes, semantic_embeddings, nz):
        self.generator = generator
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.semantic_embeddings = semantic_embeddings
        self.nz = nz
        self.generator.eval()
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        with torch.no_grad():
            z = torch.randn(1, self.nz, device=device)
            label = torch.randint(0, self.num_classes, (1,), device=device)
            
            fake_img = self.generator(z, label, self.semantic_embeddings)
            fake_img = (fake_img + 1) / 2
            fake_img = (fake_img - 0.5) / 0.5
            
            return fake_img.squeeze(0).cpu(), label.item()

# ZSL CLASSIFIER

class ZSLClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ZSLClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x).view(x.size(0), -1)
        return self.classifier(features)

# INITIALIZE MODELS

netG = Generator(
    nz=config['nz'],
    ngf=config['ngf'],
    nc=config['nc'],
    semantic_dim=config['embedding_dim'],
    semantic_proj_dim=config['semantic_proj_dim']
).to(device)

netD = Discriminator(
    nc=config['nc'],
    ndf=config['ndf'],
    semantic_dim=config['embedding_dim']
).to(device)

print(f"Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")

# OPTIMIZERS & SCHEDULERS

optimizerD = optim.Adam(netD.parameters(), lr=config['lr_d'], betas=(config['beta1'], config['beta2']))
optimizerG = optim.Adam(netG.parameters(), lr=config['lr_g'], betas=(config['beta1'], config['beta2']))

schedulerD = ExponentialLR(optimizerD, gamma=0.99)
schedulerG = ExponentialLR(optimizerG, gamma=0.99)

# TRAINING SETUP

tracker = TrainingTracker()
fixed_noise = torch.randn(16, config['nz'], device=device)
fixed_labels = torch.randint(0, num_seen_classes, (16,), device=device)

# Save real images once
real_images_dir = f"{config['results_dir']}/real"
if not Path(real_images_dir, 'real_00000.png').exists():
    save_real_images_for_eval(train_subset, save_dir=real_images_dir)

# MAIN TRAINING LOOP

print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)

global_step = 0

for epoch in range(config['num_epochs']):
    netG.train()
    netD.train()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
    epoch_g_loss = 0.0
    epoch_d_loss = 0.0
    epoch_batches = 0
    
    for real_images, labels in pbar:
        real_images, labels = real_images.to(device), labels.to(device)
        batch_size = real_images.size(0)
        epoch_batches += 1
        global_step += 1
        
        # ====================================================================
        # Train Discriminator
        # ====================================================================
        for _ in range(config['n_critic']):
            netD.zero_grad(set_to_none=True)
            
            # Real images
            real_output = netD(real_images, labels, seen_semantic_embeddings)
            
            # Fake images
            z = torch.randn(batch_size, config['nz'], device=device)
            fake_images = netG(z, labels, seen_semantic_embeddings)
            fake_output = netD(fake_images.detach(), labels, seen_semantic_embeddings)
            
            # Gradient penalty
            gp = compute_gradient_penalty(netD, real_images, fake_images, labels, seen_semantic_embeddings)
            
            # Wasserstein loss
            wasserstein_distance = fake_output.mean() - real_output.mean()
            d_loss = wasserstein_distance + config['lambda_gp'] * gp
            
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(netD.parameters(), config['grad_clip'])
            optimizerD.step()
        
        # ====================================================================
        # Train Generator
        # ====================================================================
        netG.zero_grad(set_to_none=True)
        
        z = torch.randn(batch_size, config['nz'], device=device)
        fake_images = netG(z, labels, seen_semantic_embeddings)
        fake_output = netD(fake_images, labels, seen_semantic_embeddings)
        
        g_loss = -fake_output.mean()
        
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(netG.parameters(), config['grad_clip'])
        optimizerG.step()
        
        # ====================================================================
        # Update trackers
        # ====================================================================
        tracker.update(g_loss.item(), d_loss.item(), wasserstein_distance.item(), gp.item())
        
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        
        # Log to TensorBoard
        if global_step % 50 == 0:
            writer.add_scalar('Loss/Generator', g_loss.item(), global_step)
            writer.add_scalar('Loss/Discriminator', d_loss.item(), global_step)
            writer.add_scalar('Loss/Wasserstein_Distance', wasserstein_distance.item(), global_step)
            writer.add_scalar('Loss/Gradient_Penalty', gp.item(), global_step)
        
        pbar.set_postfix({
            'G': f"{epoch_g_loss/epoch_batches:.4f}",
            'D': f"{epoch_d_loss/epoch_batches:.4f}",
            'W': f"{wasserstein_distance.item():.4f}"
        })
    
    # Update learning rate
    schedulerD.step()
    schedulerG.step()
    
    # ========================================================================
    # Generate sample images
    # ========================================================================
    with torch.no_grad():
        fake_samples = netG(fixed_noise, fixed_labels, seen_semantic_embeddings).cpu()
        grid = vutils.make_grid(fake_samples, padding=2, normalize=True)
        
        # Save to file
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title(f"Epoch {epoch+1}")
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.savefig(f"{config['results_dir']}/samples_epoch_{epoch+1:03d}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Log to TensorBoard
        writer.add_image('Generated_Samples', grid, epoch+1)
    
    # ========================================================================
    # Calculate metrics periodically
    # ========================================================================
    if (epoch + 1) % config['fid_eval_interval'] == 0 or epoch == config['num_epochs'] - 1:
        print(f"\n{'='*70}")
        print(f"Evaluating at epoch {epoch+1}")
        print(f"{'='*70}")
        
        fake_dir = save_fake_images_for_eval(netG, epoch+1)
        metrics = calculate_fid(real_images_dir, fake_dir)
        
        print(f"FID: {metrics['fid']:.2f} | IS: {metrics['is_mean']:.2f}±{metrics['is_std']:.2f} | KID: {metrics['kid_mean']:.4f}")
        
        improved = tracker.update_metrics(metrics, epoch+1, netG, netD)
        
        if tracker.should_stop_early(config['early_stopping_patience']):
            print(f"\n{'='*70}")
            print(f"Early stopping after {epoch+1} epochs without improvement")
            print(f"{'='*70}\n")
            break
    
    # ========================================================================
    # Save checkpoint
    # ========================================================================
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch + 1,
            'generator': netG.state_dict(),
            'discriminator': netD.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'schedulerG': schedulerG.state_dict(),
            'schedulerD': schedulerD.state_dict(),
            'config': config
        }, Path(config['checkpoints_dir']) / f'checkpoint_epoch_{epoch+1:03d}.pth')

# PLOT TRAINING CURVES

print("\nGenerating training visualizations...")
tracker.plot_losses()
tracker.plot_metrics()

# LOAD BEST MODEL

print(f"\nLoading best model with FID: {tracker.best_fid:.2f}")
checkpoint = torch.load(Path(config['checkpoints_dir']) / 'best_model.pth')
netG.load_state_dict(checkpoint['generator'])
netD.load_state_dict(checkpoint['discriminator'])

# FINAL EVALUATION

print("\nGenerating final evaluation images...")
final_fake_dir = save_fake_images_for_eval(netG, 'final')
final_metrics = calculate_fid(real_images_dir, final_fake_dir)
print(f"Final FID: {final_metrics['fid']:.2f} | IS: {final_metrics['is_mean']:.2f}±{final_metrics['is_std']:.2f}")

# GENERATE SAMPLE GRIDS

def generate_sample_grid(n_rows=4, n_cols=5):
    """Generate and save a grid of samples"""
    noise = torch.randn(n_rows * n_cols, config['nz'], device=device)
    labels = torch.tensor([i % num_seen_classes for i in range(n_rows * n_cols)], device=device)
    
    with torch.no_grad():
        fake_images = netG(noise, labels, seen_semantic_embeddings).cpu()
        fake_images = (fake_images + 1) / 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
        axes = axes.flatten()
        
        for i, (img, label) in enumerate(zip(fake_images, labels)):
            img_np = img.numpy().transpose(1, 2, 0)
            axes[i].imshow(img_np)
            
            orig_class_idx = new_idx_to_seen_class[label.item()]
            class_name = cifar100_classes[orig_class_idx]
            axes[i].set_title(f"{class_name}", fontsize=8)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{config['results_dir']}/generated_samples_grid.png", dpi=200, bbox_inches='tight')
        plt.close()

print("Generating sample visualization grid...")
generate_sample_grid()

# GENERATE UNSEEN CLASS IMAGES

def generate_unseen_class_images(generator, samples_per_class=None):
    """Generate synthetic images for unseen classes"""
    if samples_per_class is None:
        samples_per_class = config['synthetic_samples_per_class']
    
    save_dir = f"{config['results_dir']}/unseen_synthetic"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    generator.eval()
    
    print(f"\nGenerating {samples_per_class} images per unseen class...")
    
    with torch.no_grad():
        for i, unseen_class_idx in enumerate(tqdm(unseen_classes, desc="Generating unseen classes")):
            # Generate with noise truncation for better quality
            z = torch.randn(samples_per_class, config['nz'], device=device)
            z = torch.clamp(z, -2, 2)  # Truncation trick
            
            labels = torch.full((samples_per_class,), i, device=device)
            
            fake_imgs = generator(z, labels, unseen_semantic_embeddings)
            fake_imgs = (fake_imgs + 1) / 2
            
            class_name = cifar100_classes[unseen_class_idx]
            class_dir = Path(save_dir) / f"{i:02d}_{class_name}"
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for j, img in enumerate(fake_imgs):
                vutils.save_image(img, class_dir / f"{j:04d}.png")
            
            # Save preview grid
            if samples_per_class >= 16:
                grid = vutils.make_grid(fake_imgs[:16], nrow=4, padding=2, normalize=False)
                vutils.save_image(grid, class_dir / "samples_grid.png")
    
    print(f"Saved synthetic images to: {save_dir}")

print("Generating samples for unseen classes...")
generate_unseen_class_images(netG)

# ZERO-SHOT LEARNING EVALUATION

def evaluate_zsl_performance():
    """Train and evaluate zero-shot classifier"""
    print("\n" + "="*70)
    print("ZERO-SHOT LEARNING EVALUATION")
    print("="*70)
    
    # Load test data
    test_dataset = FilteredCIFAR100(
        root=config['data_root'], train=False, download=True,
        transform=transform_test, allowed_classes=unseen_classes
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=100, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Create synthetic training dataset
    print("\nGenerating synthetic training data...")
    num_unseen = len(unseen_classes)
    unseen_class_mapping = test_dataset.org_to_new
    
    synthetic_dataset = SyntheticDataset(
        netG,
        num_samples=config['synthetic_samples_per_class'] * num_unseen,
        num_classes=num_unseen,
        semantic_embeddings=unseen_semantic_embeddings,
        nz=config['nz']
    )
    
    # Split synthetic data
    train_size = int(0.8 * len(synthetic_dataset))
    val_size = len(synthetic_dataset) - train_size
    synth_train, synth_val = random_split(synthetic_dataset, [train_size, val_size])
    
    synth_train_loader = DataLoader(synth_train, batch_size=64, shuffle=True)
    synth_val_loader = DataLoader(synth_val, batch_size=100, shuffle=False)
    
    # Initialize classifier
    classifier = ZSLClassifier(num_unseen).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training loop
    best_val_acc = 0
    patience = 10
    epochs_without_improv = 0
    num_epochs = 30
    
    print("\nTraining ZSL classifier...")
    for epoch in range(num_epochs):
        classifier.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(synth_train_loader, desc=f"ZSL Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f"{train_loss / (pbar.n + 1):.4f}",
                'Acc': f"{100. * correct / total:.2f}%"
            })
        
        # Validation
        classifier.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in synth_val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = classifier(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Validation Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), Path(config['checkpoints_dir']) / 'best_zsl_classifier.pth')
            epochs_without_improv = 0
        else:
            epochs_without_improv += 1
            if epochs_without_improv >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Load best model
    classifier.load_state_dict(torch.load(Path(config['checkpoints_dir']) / 'best_zsl_classifier.pth'))
    
    # Test on real data
    print("\nEvaluating on real unseen class data...")
    classifier.eval()
    correct = 0
    total = 0
    class_correct = [0] * num_unseen
    class_total = [0] * num_unseen
    top5_correct = 0
    
    confusion_matrix = torch.zeros(num_unseen, num_unseen)
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            
            # Top-1 accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
                confusion_matrix[label][predicted[i]] += 1
    
    # Calculate metrics
    top1_accuracy = 100. * correct / total
    top5_accuracy = 100. * top5_correct / total
    per_class_acc = [100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                     for i in range(num_unseen)]
    mean_class_acc = np.mean(per_class_acc)
    
    # Normalize confusion matrix
    for i in range(num_unseen):
        if class_total[i] > 0:
            confusion_matrix[i] = confusion_matrix[i] / class_total[i]
    
    # Print results
    print(f"\n{'='*70}")
    print("ZSL RESULTS")
    print(f"{'='*70}")
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
    print(f"Mean Class Accuracy: {mean_class_acc:.2f}%")
    print(f"{'='*70}\n")
    
    print("Per-class Accuracy:")
    for i in range(num_unseen):
        original_class_idx = unseen_classes[i]
        original_class_name = cifar100_classes[original_class_idx]
        print(f"  {original_class_name:20s}: {per_class_acc[i]:5.2f}%")
    
    # Plot confusion matrix
    plt.figure(figsize=(14, 12))
    plt.imshow(confusion_matrix.cpu().numpy(), interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Zero-Shot Learning Confusion Matrix", fontsize=14, fontweight='bold')
    plt.colorbar()
    
    tick_marks = np.arange(num_unseen)
    class_labels = [cifar100_classes[unseen_classes[i]] for i in range(num_unseen)]
    plt.xticks(tick_marks, class_labels, rotation=90, fontsize=9)
    plt.yticks(tick_marks, class_labels, fontsize=9)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{config['results_dir']}/zsl_confusion_matrix.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot per-class accuracy
    plt.figure(figsize=(16, 6))
    sorted_indices = np.argsort(per_class_acc)
    sorted_acc = [per_class_acc[i] for i in sorted_indices]
    sorted_names = [cifar100_classes[unseen_classes[i]] for i in sorted_indices]
    
    colors = ['#d73027' if acc < mean_class_acc else '#4575b4' for acc in sorted_acc]
    
    plt.bar(range(num_unseen), sorted_acc, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    plt.xticks(range(num_unseen), sorted_names, rotation=90, fontsize=9)
    plt.ylim(0, 100)
    plt.title('Zero-Shot Learning Accuracy by Class', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.axhline(y=mean_class_acc, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_class_acc:.2f}%')
    plt.axhline(y=top1_accuracy, color='green', linestyle='--', linewidth=2,
                label=f'Overall: {top1_accuracy:.2f}%')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{config['results_dir']}/zsl_class_accuracy.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    return {
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "mean_class_accuracy": mean_class_acc,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion_matrix.cpu().numpy()
    }

print("\nEvaluating zero-shot learning performance...")
zsl_metrics = evaluate_zsl_performance()

# CREATE SUMMARY VISUALIZATION

def create_summary_visualization():
    """Create comprehensive summary of the experiment"""
    print("\nCreating experiment summary...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Generated samples
    ax1 = fig.add_subplot(gs[0, :2])
    sample_grid_path = Path(config['results_dir']) / 'generated_samples_grid.png'
    if sample_grid_path.exists():
        img = plt.imread(sample_grid_path)
        ax1.imshow(img)
        ax1.set_title('Generated Samples (Seen Classes)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Training curves
    ax2 = fig.add_subplot(gs[0, 2])
    if tracker.g_losses and tracker.d_losses:
        window = min(500, len(tracker.g_losses))
        ax2.plot(tracker.g_losses[-window:], label='G Loss', alpha=0.7, linewidth=1.5)
        ax2.plot(tracker.d_losses[-window:], label='D Loss', alpha=0.7, linewidth=1.5)
        ax2.set_title('Training Losses (Last 500)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Iterations', fontsize=9)
        ax2.set_ylabel('Loss', fontsize=9)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
    
    # Metrics progression
    ax3 = fig.add_subplot(gs[1, 0])
    if tracker.metrics_history:
        epochs = [m[0] for m in tracker.metrics_history]
        fids = [m[1]['fid'] for m in tracker.metrics_history]
        ax3.plot(epochs, fids, 'o-', color='steelblue', linewidth=2, markersize=6)
        ax3.axhline(y=tracker.best_fid, color='r', linestyle='--', linewidth=2,
                   label=f'Best: {tracker.best_fid:.2f}')
        ax3.set_title('FID Progression', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Epoch', fontsize=9)
        ax3.set_ylabel('FID Score', fontsize=9)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
    
    # Inception Score
    ax4 = fig.add_subplot(gs[1, 1])
    if tracker.metrics_history:
        epochs = [m[0] for m in tracker.metrics_history]
        is_means = [m[1]['is_mean'] for m in tracker.metrics_history]
        ax4.plot(epochs, is_means, 'o-', color='green', linewidth=2, markersize=6)
        ax4.set_title('Inception Score', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Epoch', fontsize=9)
        ax4.set_ylabel('IS', fontsize=9)
        ax4.grid(True, alpha=0.3)
    
    # Wasserstein distance
    ax5 = fig.add_subplot(gs[1, 2])
    if tracker.w_distances:
        window = min(500, len(tracker.w_distances))
        ax5.plot(tracker.w_distances[-window:], color='orange', alpha=0.7, linewidth=1.5)
        ax5.set_title('Wasserstein Distance', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Iterations', fontsize=9)
        ax5.set_ylabel('W-Distance', fontsize=9)
        ax5.grid(True, alpha=0.3)
    
    # ZSL class accuracy
    ax6 = fig.add_subplot(gs[2, :2])
    zsl_acc_path = Path(config['results_dir']) / 'zsl_class_accuracy.png'
    if zsl_acc_path.exists():
        img = plt.imread(zsl_acc_path)
        ax6.imshow(img)
        ax6.set_title('Zero-Shot Learning Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax6.axis('off')
    
    # Summary statistics
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    summary_text = f"""
EXPERIMENT SUMMARY
{'='*30}

GAN Training:
  Best FID: {tracker.best_fid:.2f}
  Final FID: {final_metrics['fid']:.2f}
  Best IS: {tracker.metrics_history[-1][1]['is_mean']:.2f} if tracker.metrics_history else 'N/A'
  
Zero-Shot Learning:
  Top-1 Acc: {zsl_metrics['top1_accuracy']:.2f}%
  Top-5 Acc: {zsl_metrics['top5_accuracy']:.2f}%
  Mean Class: {zsl_metrics['mean_class_accuracy']:.2f}%

Dataset:
  Seen Classes: {len(seen_classes)}
  Unseen Classes: {len(unseen_classes)}
  Training Samples: {len(train_subset)}

Training:
  Epochs: {epoch + 1}
  Batch Size: {config['batch_size']}
  Learning Rate G: {config['lr_g']}
  Learning Rate D: {config['lr_d']}
"""
    
    ax7.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('ZSL-cWGAN-GP Training Summary', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(f"{config['results_dir']}/experiment_summary.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Experiment summary saved to: {config['results_dir']}/experiment_summary.png")

create_summary_visualization()

# FINAL SUMMARY PRINT

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"\nGAN Metrics:")
print(f"  Best FID Score: {tracker.best_fid:.2f}")
print(f"  Final FID Score: {final_metrics['fid']:.2f}")
print(f"  Final IS: {final_metrics['is_mean']:.2f}±{final_metrics['is_std']:.2f}")
print(f"  Final KID: {final_metrics['kid_mean']:.4f}")

print(f"\nZero-Shot Learning:")
print(f"  Top-1 Accuracy: {zsl_metrics['top1_accuracy']:.2f}%")
print(f"  Top-5 Accuracy: {zsl_metrics['top5_accuracy']:.2f}%")
print(f"  Mean Class Accuracy: {zsl_metrics['mean_class_accuracy']:.2f}%")

print(f"\nResults saved in: {config['results_dir']}/")
print(f"Checkpoints saved in: {config['checkpoints_dir']}/")
print(f"TensorBoard logs: {config['results_dir']}/runs/")

print("\n" + "="*70)
print("To view TensorBoard:")
print(f"  tensorboard --logdir={config['results_dir']}/runs")
print("="*70 + "\n")

writer.close()