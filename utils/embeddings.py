"""
Semantic Embeddings Module
Day 1: CLIP Text Embeddings Implementation

This module handles extraction of semantic embeddings for class labels:
- CLIP text embeddings (primary, modern)
- GloVe embeddings (legacy, for comparison)
"""

import torch
import numpy as np
import pickle
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# CLIP imports
from transformers import CLIPTokenizer, CLIPTextModel

# For GloVe (legacy)
import requests
import zipfile


class CLIPTextEmbedder:
    """
    CLIP Text Embedding Extractor
    
    Uses pretrained CLIP text encoder to extract semantic embeddings
    for class labels. This is the MODERN approach (2021+).
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Initialize CLIP text embedder
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on
            cache_dir: Cache directory for model weights
            normalize: Whether to L2-normalize embeddings
        """
        self.device = device
        self.normalize = normalize
        
        print(f"Loading CLIP model: {model_name}")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        ).to(device)
        self.text_encoder.eval()
        
        self.embedding_dim = self.text_encoder.config.hidden_size
        print(f"CLIP text embedding dimension: {self.embedding_dim}")
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode list of texts to embeddings
        
        Args:
            texts: List of text strings
            
        Returns:
            Tensor of shape [N, embedding_dim]
        """
        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings
            outputs = self.text_encoder(**inputs)
            # Use pooled output (CLS token)
            embeddings = outputs.pooler_output
            
            # Normalize if requested
            if self.normalize:
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            return embeddings
    
    def get_class_embeddings(
        self,
        class_names: List[str],
        use_templates: bool = True
    ) -> Dict[int, np.ndarray]:
        """
        Get embeddings for class names with optional prompt templates
        
        Args:
            class_names: List of class names
            use_templates: Whether to use prompt templates (improves quality)
            
        Returns:
            Dictionary mapping class_idx -> embedding array
        """
        embeddings_dict = {}
        
        # CLIP prompt templates (improves zero-shot performance)
        templates = [
            "a photo of a {}.",
            "a photo of the {}.",
            "a picture of a {}.",
            "an image of a {}.",
            "{}.",
        ] if use_templates else ["{}"]
        
        print(f"Extracting CLIP embeddings for {len(class_names)} classes...")
        
        for class_idx, class_name in enumerate(tqdm(class_names)):
            # Clean class name
            clean_name = self._clean_class_name(class_name)
            
            # Create prompts
            prompts = [template.format(clean_name) for template in templates]
            
            # Encode
            embeddings = self.encode_text(prompts)
            
            # Average over templates
            avg_embedding = embeddings.mean(dim=0)
            
            # Store as numpy array
            embeddings_dict[class_idx] = avg_embedding.cpu().numpy()
        
        return embeddings_dict
    
    @staticmethod
    def _clean_class_name(name: str) -> str:
        """Clean class name for better CLIP encoding"""
        # Remove underscores, make lowercase
        name = name.replace('_', ' ').lower()
        # Remove extra spaces
        name = ' '.join(name.split())
        return name


class GloVeEmbedder:
    """
    GloVe Embedding Extractor (Legacy)
    
    For comparison purposes or fallback.
    """
    
    def __init__(
        self,
        glove_file: str = "glove.6B.300d.txt",
        cache_dir: str = "./cache",
        embedding_dim: int = 300
    ):
        self.glove_file = glove_file
        self.cache_dir = Path(cache_dir)
        self.embedding_dim = embedding_dim
        self.embeddings_dict = None
        
    def load_glove(self) -> Dict[str, np.ndarray]:
        """Load GloVe embeddings with caching"""
        cache_file = self.cache_dir / "glove_cache.pkl"
        
        # Try to load from cache
        if cache_file.exists():
            print(f"Loading cached GloVe embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Download if needed
        if not Path(self.glove_file).exists():
            self._download_glove()
        
        # Load GloVe
        print(f"Loading GloVe embeddings from {self.glove_file}...")
        embeddings_dict = {}
        
        with open(self.glove_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading GloVe"):
                values = line.strip().split()
                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)
                embeddings_dict[word] = vector
        
        print(f"Loaded {len(embeddings_dict)} GloVe vectors")
        
        # Cache for next time
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        
        return embeddings_dict
    
    def _download_glove(self):
        """Download GloVe embeddings"""
        print("Downloading GloVe embeddings...")
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open('glove.6B.zip', 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        with zipfile.ZipFile('glove.6B.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        
        print("GloVe downloaded and extracted")
    
    def get_class_embeddings(
        self,
        class_names: List[str]
    ) -> Dict[int, np.ndarray]:
        """Get GloVe embeddings for class names"""
        if self.embeddings_dict is None:
            self.embeddings_dict = self.load_glove()
        
        embeddings = {}
        
        print(f"Creating GloVe embeddings for {len(class_names)} classes...")
        
        for class_idx, class_name in enumerate(class_names):
            clean_name = self._clean_label(class_name)
            
            # Try direct lookup
            if clean_name in self.embeddings_dict:
                embeddings[class_idx] = self.embeddings_dict[clean_name]
            else:
                # Average over words
                words = clean_name.split()
                found_vectors = [
                    self.embeddings_dict[word]
                    for word in words
                    if word in self.embeddings_dict
                ]
                
                if found_vectors:
                    embeddings[class_idx] = np.mean(found_vectors, axis=0)
                else:
                    # Fallback: random embedding
                    print(f"Warning: '{class_name}' not found in GloVe")
                    rng = np.random.RandomState(42 + class_idx)
                    embeddings[class_idx] = rng.normal(
                        scale=0.6, size=self.embedding_dim
                    ).astype(np.float32)
        
        return embeddings
    
    @staticmethod
    def _clean_label(label: str) -> str:
        """Clean label for GloVe lookup"""
        return label.translate(
            str.maketrans('', '', string.punctuation)
        ).lower()


class EmbeddingManager:
    """
    Unified Embedding Manager
    
    Handles both CLIP and GloVe embeddings, with caching and conversion.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.embedding_type = config['embeddings']['type']
        self.device = config['experiment']['device']
        
        # Initialize embedders
        self.clip_embedder = None
        self.glove_embedder = None
        
        if self.embedding_type in ['clip', 'both']:
            self.clip_embedder = CLIPTextEmbedder(
                model_name=config['embeddings']['clip_model'],
                device=self.device,
                cache_dir=config['embeddings']['clip_cache_dir'],
                normalize=config['embeddings']['normalize']
            )
        
        if self.embedding_type in ['glove', 'both']:
            self.glove_embedder = GloVeEmbedder(
                glove_file=config['embeddings']['glove_path'],
                cache_dir=config['paths']['cache_dir'],
                embedding_dim=config['embeddings']['glove_dim']
            )
    
    def get_embeddings(
        self,
        class_names: List[str],
        class_indices: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Get semantic embeddings for classes
        
        Args:
            class_names: List of all class names
            class_indices: Optional indices to extract (for seen/unseen split)
            
        Returns:
            embeddings: Tensor of shape [num_classes, embedding_dim]
            embedding_dim: Dimension of embeddings
        """
        cache_file = Path(self.config['paths']['cache_dir']) / f"embeddings_{self.embedding_type}.pkl"
        
        # Try to load from cache
        if cache_file.exists():
            print(f"Loading cached {self.embedding_type} embeddings...")
            with open(cache_file, 'rb') as f:
                embeddings_dict = pickle.load(f)
        else:
            # Extract embeddings
            if self.embedding_type == 'clip':
                embeddings_dict = self.clip_embedder.get_class_embeddings(class_names)
            elif self.embedding_type == 'glove':
                embeddings_dict = self.glove_embedder.get_class_embeddings(class_names)
            elif self.embedding_type == 'both':
                # Concatenate CLIP and GloVe
                clip_emb = self.clip_embedder.get_class_embeddings(class_names)
                glove_emb = self.glove_embedder.get_class_embeddings(class_names)
                embeddings_dict = {
                    idx: np.concatenate([clip_emb[idx], glove_emb[idx]])
                    for idx in range(len(class_names))
                }
            
            # Cache
            Path(self.config['paths']['cache_dir']).mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings_dict, f)
            print(f"Cached embeddings to {cache_file}")
        
        # Convert to tensor
        if class_indices is not None:
            # Extract specific classes
            embeddings_list = [embeddings_dict[idx] for idx in class_indices]
        else:
            # All classes
            embeddings_list = [embeddings_dict[idx] for idx in range(len(class_names))]
        
        embeddings = torch.tensor(
            np.stack(embeddings_list),
            dtype=torch.float32,
            device=self.device
        )
        
        embedding_dim = embeddings.shape[1]
        
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embedding dimension: {embedding_dim}")
        
        return embeddings, embedding_dim


# Convenience functions
def get_clip_embeddings(
    class_names: List[str],
    device: str = "cuda",
    normalize: bool = True
) -> torch.Tensor:
    """Quick function to get CLIP embeddings"""
    embedder = CLIPTextEmbedder(device=device, normalize=normalize)
    embeddings_dict = embedder.get_class_embeddings(class_names)
    embeddings = torch.tensor(
        np.stack([embeddings_dict[i] for i in range(len(class_names))]),
        dtype=torch.float32,
        device=device
    )
    return embeddings


def compare_embeddings(
    class_names: List[str],
    sample_classes: Optional[List[int]] = None
) -> None:
    """
    Compare CLIP vs GloVe embeddings (for analysis)
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    # Get both embeddings
    clip_embedder = CLIPTextEmbedder()
    glove_embedder = GloVeEmbedder()
    
    clip_emb = clip_embedder.get_class_embeddings(class_names)
    glove_emb = glove_embedder.get_class_embeddings(class_names)
    
    # Sample for visualization
    if sample_classes is None:
        sample_classes = list(range(min(20, len(class_names))))
    
    clip_sample = np.stack([clip_emb[i] for i in sample_classes])
    glove_sample = np.stack([glove_emb[i] for i in sample_classes])
    
    # PCA for visualization
    pca = PCA(n_components=2)
    clip_2d = pca.fit_transform(clip_sample)
    glove_2d = pca.fit_transform(glove_sample)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.scatter(clip_2d[:, 0], clip_2d[:, 1], alpha=0.6, s=100)
    for i, idx in enumerate(sample_classes):
        ax1.annotate(class_names[idx], (clip_2d[i, 0], clip_2d[i, 1]), fontsize=8)
    ax1.set_title('CLIP Embeddings (PCA)')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(glove_2d[:, 0], glove_2d[:, 1], alpha=0.6, s=100, color='orange')
    for i, idx in enumerate(sample_classes):
        ax2.annotate(class_names[idx], (glove_2d[i, 0], glove_2d[i, 1]), fontsize=8)
    ax2.set_title('GloVe Embeddings (PCA)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('embedding_comparison.png', dpi=150)
    print("Saved comparison plot to embedding_comparison.png")


if __name__ == "__main__":
    # Test the embeddings
    from torchvision import datasets
    
    # Load CIFAR-100 class names
    cifar100 = datasets.CIFAR100(root='./data', download=True)
    class_names = cifar100.classes
    
    print("Testing CLIP embeddings...")
    clip_embeddings = get_clip_embeddings(class_names[:10])  # Test on first 10
    print(f"CLIP embeddings shape: {clip_embeddings.shape}")
    print(f"Sample embedding norm: {clip_embeddings[0].norm().item():.4f}")
    
    print("\nComparing embeddings...")
    compare_embeddings(class_names, sample_classes=list(range(20)))