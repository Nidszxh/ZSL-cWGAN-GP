import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
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
from torch.optim.lr_scheduler import CosineAnnealingLR

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed()

# Hyperparameters
config = {
    'batch_size': 64,
    'nc': 3,
    'nz': 128,
    'ngf': 64,
    'ndf': 64,
    'num_classes': 100,
    'num_epochs': 10,
    'lr_g': 0.0001,
    'lr_d': 0.0002,
    'beta1': 0.0,
    'beta2': 0.9,
    'lambda_gp': 10,
    'n_critic': 5,
    'embedding_dim': 300,
    'semantic_proj_dim': 256,
    'early_stopping_patience': 10
}

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories for results
for dir_path in ['results', 'results/fake', 'results/real', 'results/unseen_synthetic', 'checkpoints']:
    os.makedirs(dir_path, exist_ok=True)

# Transforms with augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Split seen and unseen classes
set_seed()
all_classes = np.arange(config['num_classes'])
np.random.shuffle(all_classes)
seen_classes = all_classes[:80]
unseen_classes = all_classes[80:]

# Filtered CIFAR100 dataset
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
    root='./data', train=True, download=True, transform=transform_train,
    allowed_classes=seen_classes
)

# Split training data
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# Optimize worker count
num_workers = min(4, os.cpu_count() or 2)
train_loader = DataLoader(train_subset, batch_size=config['batch_size'], 
                          shuffle=True, num_workers=num_workers, 
                          pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_subset, batch_size=config['batch_size'], 
                        shuffle=False, num_workers=num_workers, 
                        pin_memory=True, persistent_workers=True)

# Save class mappings
seen_class_to_new_idx = train_dataset.org_to_new
new_idx_to_seen_class = {v: k for k, v in seen_class_to_new_idx.items()}
num_seen_classes = len(seen_class_to_new_idx)

# Load GloVe embeddings
def load_glove_embeddings(glove_file='glove.6B.300d.txt'):
    if not os.path.exists(glove_file):
        print("Downloading GloVe embeddings...")
        glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
        r = requests.get(glove_url, stream=True, allow_redirects=True)
        
        with open('glove.6B.zip', 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        
        with zipfile.ZipFile('glove.6B.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
            
        print("GloVe embeddings downloaded and extracted")
    
    print(f"Loading GloVe embeddings from {glove_file}...")
    embeddings_dict = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embeddings_dict[word] = vector
    print(f"Loaded {len(embeddings_dict)} GloVe word vectors")
    return embeddings_dict

# Load class names and GloVe embeddings
glove_embeddings = load_glove_embeddings('/home/nidszxh/Codes/ZSL-cWGAN-GP/glove.6B.300d.txt')
cifar100_classes = datasets.CIFAR100(root='./data', download=True).classes

def clean_label(label):
    """Remove punctuation and convert to lowercase"""
    return label.translate(str.maketrans('', '', string.punctuation)).lower()

# Semantic embedding creation
def create_semantic_embeddings(class_names, glove_embeddings, embedding_dim=300):
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
                print(f"Class '{class_name}' not found in GloVe")
                rng = np.random.RandomState(42 + class_idx)
                semantic_embeddings[class_idx] = rng.normal(scale=0.6, size=embedding_dim).astype(np.float32)
    
    return semantic_embeddings

# Create semantic embeddings
semantic_embeddings = create_semantic_embeddings(cifar100_classes, glove_embeddings, 
                                               embedding_dim=config['embedding_dim'])

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

# Generator with semantic integration
class Generator(nn.Module):
    def __init__(self, nz=128, ngf=64, nc=3, semantic_dim=300, semantic_proj_dim=256):
        super(Generator, self).__init__()
        
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, semantic_proj_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(semantic_proj_dim, semantic_proj_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        combined_dim = nz + semantic_proj_dim
        
        self.project = nn.Sequential(
            nn.Linear(combined_dim, ngf * 8 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, nc, 3, 1, 1),
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

# Discriminator
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
        
        output = self.output(h).squeeze(1)
        
        projection = self.embed_output(sem_features).view(-1, config['ndf'] * 8, 1, 1)
        h_mean = h.mean([2, 3])
        cond_output = torch.sum(projection.squeeze() * h_mean, dim=1)
        
        return output + cond_output

# Gradient Penalty
def compute_gradient_penalty(D, real, fake, labels, semantic_embeddings):
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

# Save real images for evaluation
def save_real_images_for_eval(dataset, num_samples=5000, save_dir='results/real'):
    os.makedirs(save_dir, exist_ok=True)
    
    loader = DataLoader(dataset, batch_size=100, shuffle=False)
    
    count = 0
    print(f"Saving {num_samples} real images for evaluation...")
    
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Saving real images"):
            images = (images + 1) / 2
            
            for img in images:
                if count >= num_samples:
                    break
                vutils.save_image(img, os.path.join(save_dir, f"real_{count:05d}.png"))
                count += 1
                
            if count >= num_samples:
                break
    
    print(f"Saved {count} real images to {save_dir}")

# Save fake images for evaluation
def save_fake_images_for_eval(generator, epoch, num_samples=2000, save_dir='results/fake', version=None):
    if version is not None:
        save_dir = f"{save_dir}_epoch{epoch}" if version == 'epoch' else f"{save_dir}_{version}"
    
    os.makedirs(save_dir, exist_ok=True)
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
                vutils.save_image(img, os.path.join(save_dir, f"fake_{idx:05d}.png"))
    
    generator.train()
    print(f"Saved images to {save_dir}")
    return save_dir

# Training tracker
class TrainingTracker:
    def __init__(self, log_dir='results'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.g_losses = []
        self.d_losses = []
        self.w_distances = []
        self.gp_values = []
        self.fid_scores = []
        
        self.best_fid = float('inf')
        self.epochs_without_improv = 0
    
    def update(self, g_loss, d_loss, w_dist, gp):
        self.g_losses.append(g_loss)
        self.d_losses.append(d_loss)
        self.w_distances.append(w_dist)
        self.gp_values.append(gp)
    
    def update_fid(self, fid_score, epoch, netG, netD):
        self.fid_scores.append((epoch, fid_score))
        
        if fid_score < self.best_fid:
            self.best_fid = fid_score
            self.epochs_without_improv = 0
            
            torch.save({
                'generator': netG.state_dict(),
                'discriminator': netD.state_dict(),
                'epoch': epoch,
                'fid': fid_score
            }, os.path.join('checkpoints', 'best_model.pth'))
            
            print(f"✓ New best FID: {fid_score:.2f} - Model saved")
            return True
        else:
            self.epochs_without_improv += 1
            print(f"× FID did not improve: {fid_score:.2f} vs best {self.best_fid:.2f}")
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
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'), dpi=150)
        plt.close()
    
    def plot_fid(self):
        if not self.fid_scores:
            return
            
        epochs, scores = zip(*self.fid_scores)
        
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, scores, 'o-', label='FID Score', markersize=6)
        plt.axhline(y=self.best_fid, color='r', linestyle='--', label=f'Best: {self.best_fid:.2f}')
        plt.xlabel('Epoch')
        plt.ylabel('FID Score')
        plt.title('FID Score During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.log_dir, 'fid_progress.png'), dpi=150)
        plt.close()

# Calculate FID score
def calculate_fid(real_dir, fake_dir):
    print(f"Calculating FID between {real_dir} and {fake_dir}")
    try:
        metrics = calculate_metrics(
            input1=real_dir,
            input2=fake_dir,
            cuda=torch.cuda.is_available(),
            fid=True,
            isc=False,
            verbose=False
        )
        return metrics['frechet_inception_distance']
    except Exception as e:
        print(f"Error calculating FID: {e}")
        return float('inf')

# Initialize networks
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

# Optimizers
optimizerD = optim.Adam(netD.parameters(), lr=config['lr_d'], betas=(config['beta1'], config['beta2']))
optimizerG = optim.Adam(netG.parameters(), lr=config['lr_g'], betas=(config['beta1'], config['beta2']))

# Schedulers
schedulerD = CosineAnnealingLR(optimizerD, T_max=config['num_epochs'])
schedulerG = CosineAnnealingLR(optimizerG, T_max=config['num_epochs'])

# Training setup
tracker = TrainingTracker()
fixed_noise = torch.randn(16, config['nz'], device=device)
fixed_labels = torch.randint(0, num_seen_classes, (16,), device=device)

# Save real images once
real_images_dir = 'results/real'
if not os.path.exists(os.path.join(real_images_dir, 'real_00000.png')):
    save_real_images_for_eval(train_subset, save_dir=real_images_dir)

# Main training loop
print("Starting Training Loop...")
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
        
        # Train Discriminator
        for _ in range(config['n_critic']):
            netD.zero_grad(set_to_none=True)
            
            real_output = netD(real_images, labels, seen_semantic_embeddings)
            
            z = torch.randn(batch_size, config['nz'], device=device)
            fake_images = netG(z, labels, seen_semantic_embeddings)
            fake_output = netD(fake_images.detach(), labels, seen_semantic_embeddings)
            
            gp = compute_gradient_penalty(netD, real_images, fake_images, labels, seen_semantic_embeddings)
            
            wasserstein_distance = fake_output.mean() - real_output.mean()
            d_loss = wasserstein_distance + config['lambda_gp'] * gp
            
            d_loss.backward()
            optimizerD.step()
        
        # Train Generator
        netG.zero_grad(set_to_none=True)
        
        z = torch.randn(batch_size, config['nz'], device=device)
        fake_images = netG(z, labels, seen_semantic_embeddings)
        fake_output = netD(fake_images, labels, seen_semantic_embeddings)
        
        g_loss = -fake_output.mean()
        
        g_loss.backward()
        optimizerG.step()
        
        # Update trackers
        tracker.update(g_loss.item(), d_loss.item(), wasserstein_distance.item(), gp.item())
        
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        pbar.set_postfix({
            'G_loss': f"{epoch_g_loss/epoch_batches:.4f}", 
            'D_loss': f"{epoch_d_loss/epoch_batches:.4f}"
        })
    
    schedulerD.step()
    schedulerG.step()
    
    # Generate sample images
    with torch.no_grad():
        fake_images = netG(fixed_noise, fixed_labels, seen_semantic_embeddings).cpu()
        grid = vutils.make_grid(fake_images, padding=2, normalize=True)
        
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title(f"Epoch {epoch+1}")
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.savefig(f'results/samples_epoch_{epoch+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Calculate FID periodically
    if (epoch + 1) % 5 == 0 or epoch == config['num_epochs'] - 1:
        fake_dir = save_fake_images_for_eval(netG, epoch+1, version='epoch')
        fid_score = calculate_fid(real_images_dir, fake_dir)
        print(f"Epoch {epoch+1} - FID: {fid_score:.2f}")
        
        improved = tracker.update_fid(fid_score, epoch+1, netG, netD)
        
        if tracker.should_stop_early(config['early_stopping_patience']):
            print(f"Early stopping after {epoch+1} epochs without FID improvement")
            break
    
    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch + 1,
            'generator': netG.state_dict(),
            'discriminator': netD.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'schedulerG': schedulerG.state_dict(),
            'schedulerD': schedulerD.state_dict()
        }, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')

# Plot training curves
tracker.plot_losses()
tracker.plot_fid()

# Load best model
print(f"Loading best model with FID: {tracker.best_fid:.2f}")
checkpoint = torch.load('checkpoints/best_model.pth')
netG.load_state_dict(checkpoint['generator'])
netD.load_state_dict(checkpoint['discriminator'])

# Final FID
final_fake_dir = save_fake_images_for_eval(netG, 'final', version='final')
final_fid = calculate_fid(real_images_dir, final_fake_dir)
print(f"Final FID: {final_fid:.2f}")

# Generate sample grid
def generate_sample_grid(n_rows=4, n_cols=5):
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
        plt.savefig('results/generated_samples_grid.png', dpi=200, bbox_inches='tight')
        plt.close()

# Generate unseen class images
def generate_unseen_class_images(generator, num_samples_per_class=500, save_dir='results/unseen_synthetic'):
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()
    
    print(f"Generating {num_samples_per_class} images for each of {len(unseen_classes)} unseen classes...")
    
    with torch.no_grad():
        for i, unseen_class_idx in enumerate(tqdm(unseen_classes, desc="Generating unseen classes")):
            z = torch.randn(num_samples_per_class, config['nz'], device=device)
            z_norm = F.normalize(z, dim=1)
            
            labels = torch.full((num_samples_per_class,), i, device=device)
            
            fake_imgs = generator(z_norm, labels, unseen_semantic_embeddings)
            fake_imgs = (fake_imgs + 1) / 2
            
            class_name = cifar100_classes[unseen_class_idx]
            class_dir = os.path.join(save_dir, f"{i:02d}_{class_name}")
            os.makedirs(class_dir, exist_ok=True)
            
            for j, img in enumerate(fake_imgs):
                img_path = os.path.join(class_dir, f"{j:04d}.png")
                vutils.save_image(img, img_path)
            
            if num_samples_per_class >= 16:
                grid = vutils.make_grid(fake_imgs[:16], nrow=4, padding=2, normalize=False)
                grid_path = os.path.join(class_dir, "samples_grid.png")
                vutils.save_image(grid, grid_path)
    
    print(f"Saved synthetic images for unseen classes at: {save_dir}")

# ZSL Classifier
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
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x).view(x.size(0), -1)
        return self.classifier(features)

# ZSL Evaluation
def evaluate_zsl_performance():
    print("Evaluating Zero-Shot Learning Performance...")
    
    test_dataset = FilteredCIFAR100(
        root='./data', train=False, download=True, transform=transform_test,
        allowed_classes=unseen_classes)

    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)

    print("Generating synthetic training data for unseen classes...")
    synthetic_images = []
    synthetic_labels = []
    samples_per_class = 1000 
    
    unseen_class_mapping = test_dataset.org_to_new
    
    netG.eval()
    with torch.no_grad():
        for i, original_class in enumerate(tqdm(unseen_classes, desc="Generating synthetic data")):
            class_idx = unseen_class_mapping[original_class]
            
            for j in range(0, samples_per_class, 100):
                batch_size = min(100, samples_per_class - j)
                if batch_size <= 0:
                    break
                
                z = torch.randn(batch_size, config['nz'], device=device)
                labels = torch.full((batch_size,), i, device=device)
                
                fake_imgs = netG(z, labels, unseen_semantic_embeddings)
                fake_imgs = (fake_imgs + 1) / 2
                fake_imgs = (fake_imgs - 0.5) / 0.5
                
                synthetic_images.append(fake_imgs.cpu())
                synthetic_labels.extend([class_idx] * batch_size)
    
    synthetic_images = torch.cat(synthetic_images)
    synthetic_labels = torch.tensor(synthetic_labels)
    
    synthetic_dataset = TensorDataset(synthetic_images, synthetic_labels)
    train_size = int(0.8 * len(synthetic_dataset))
    val_size = len(synthetic_dataset) - train_size
    
    synth_train, synth_val = random_split(synthetic_dataset, [train_size, val_size])
    synth_train_loader = DataLoader(synth_train, batch_size=64, shuffle=True)
    synth_val_loader = DataLoader(synth_val, batch_size=100, shuffle=False)
    
    num_unseen = len(unseen_classes)
    classifier = ZSLClassifier(num_unseen).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_val_acc = 0
    patience = 10
    epochs_without_improv = 0
    
    num_epochs = 30
    for epoch in range(num_epochs):
        classifier.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(synth_train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")
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
            
            pbar.set_postfix({'Loss': f"{train_loss / (pbar.n + 1):.4f}", 
                             'Acc': f"{100. * correct / total:.2f}%"})
        
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
            torch.save(classifier.state_dict(), 'checkpoints/best_zsl_classifier.pth')
            epochs_without_improv = 0
        else:
            epochs_without_improv += 1
            if epochs_without_improv >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    classifier.load_state_dict(torch.load('checkpoints/best_zsl_classifier.pth'))
    
    print("Evaluating on real unseen class data...")
    classifier.eval()
    correct = 0
    total = 0
    class_correct = [0] * num_unseen
    class_total = [0] * num_unseen
    
    confusion_matrix = torch.zeros(num_unseen, num_unseen)
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = classifier(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
                confusion_matrix[label][predicted[i]] += 1
    
    accuracy = 100. * correct / total
    per_class_acc = [100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                     for i in range(num_unseen)]
    
    for i in range(num_unseen):
        if class_total[i] > 0:
            confusion_matrix[i] = confusion_matrix[i] / class_total[i]
    
    print(f"\nOverall Zero-Shot Accuracy: {accuracy:.2f}%")
    print("\nPer-class Accuracy:")
    for i in range(num_unseen):
        original_class_idx = unseen_classes[i]
        original_class_name = cifar100_classes[original_class_idx]
        print(f"  Class {i} ({original_class_name}): {per_class_acc[i]:.2f}%")
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion_matrix.cpu().numpy(), interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("ZSL Confusion Matrix")
    plt.colorbar()
    
    tick_marks = np.arange(num_unseen)
    class_labels = [cifar100_classes[unseen_classes[i]] for i in range(num_unseen)]
    plt.xticks(tick_marks, class_labels, rotation=90, fontsize=8)
    plt.yticks(tick_marks, class_labels, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/zsl_confusion_matrix.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot per-class accuracy
    plt.figure(figsize=(14, 6))
    sorted_indices = np.argsort(per_class_acc)
    sorted_acc = [per_class_acc[i] for i in sorted_indices]
    sorted_names = [cifar100_classes[unseen_classes[i]] for i in sorted_indices]
    
    plt.bar(range(num_unseen), sorted_acc, color='steelblue', alpha=0.7)
    plt.xticks(range(num_unseen), sorted_names, rotation=90, fontsize=8)
    plt.ylim(0, 100)
    plt.title('Zero-Shot Learning Accuracy by Class')
    plt.ylabel('Accuracy (%)')
    plt.axhline(y=accuracy, color='r', linestyle='--', linewidth=2, label=f'Average: {accuracy:.2f}%')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('results/zsl_class_accuracy.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    return {
        "accuracy": accuracy,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion_matrix.cpu().numpy()
    }

# Generate visualizations
print("Generating sample visualization grid...")
generate_sample_grid()

print("Generating samples for unseen classes...")
generate_unseen_class_images(netG)

print("Evaluating zero-shot learning performance...")
zsl_metrics = evaluate_zsl_performance()

# Print summary
print("\n" + "="*50)
print("TRAINING SUMMARY")
print("="*50)
print(f"Best FID Score: {tracker.best_fid:.2f}")
print(f"Final FID Score: {final_fid:.2f}")
print(f"ZSL Accuracy: {zsl_metrics['accuracy']:.2f}%")
print(f"Average Per-Class Accuracy: {np.mean(zsl_metrics['per_class_accuracy']):.2f}%")
print("="*50)
print("\nResults saved in 'results/' directory")
print("Checkpoints saved in 'checkpoints/' directory")

# Create summary visualization
def create_summary_visualization():
    print("Creating summary visualization...")
    fig = plt.figure(figsize=(16, 12))
    
    ax1 = plt.subplot(2, 2, 1)
    if os.path.exists('results/generated_samples_grid.png'):
        img = plt.imread('results/generated_samples_grid.png')
        ax1.imshow(img)
        ax1.set_title('Generated Samples (Seen Classes)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 2, 2)
    if os.path.exists('results/fid_progress.png'):
        img = plt.imread('results/fid_progress.png')
        ax2.imshow(img)
    else:
        if tracker.fid_scores:
            epochs, scores = zip(*tracker.fid_scores)
            ax2.plot(epochs, scores, 'o-', color='steelblue', linewidth=2, markersize=8)
            ax2.set_title('FID Progression', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('FID Score')
            ax2.grid(True, alpha=0.3)
    ax2.axis('off') if os.path.exists('results/fid_progress.png') else None
    
    ax3 = plt.subplot(2, 2, 3)
    if os.path.exists('results/zsl_class_accuracy.png'):
        img = plt.imread('results/zsl_class_accuracy.png')
        ax3.imshow(img)
    ax3.axis('off')
    
    ax4 = plt.subplot(2, 2, 4)
    if os.path.exists('results/training_curves.png'):
        img = plt.imread('results/training_curves.png')
        ax4.imshow(img)
    else:
        ax4.plot(tracker.g_losses[-500:], label='G Loss', alpha=0.7)
        ax4.plot(tracker.d_losses[-500:], label='D Loss', alpha=0.7)
        ax4.set_title('Training Losses (Last 500 Iterations)', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    ax4.axis('off') if os.path.exists('results/training_curves.png') else None
    
    plt.tight_layout()
    plt.savefig('results/exp_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Experiment summary created: results/exp_summary.png")

create_summary_visualization()