"""
Semantic Embeddings Module — CLIP text embeddings for CIFAR-100 class labels.

Single CLIP text encoder (ViT-L/14, 768-dim), cached to disk.
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from transformers import CLIPTokenizer, CLIPTextModel


class CLIPTextEmbedder:
    """
    CLIP Text Embedding Extractor.
    Uses the pretrained CLIP text encoder to extract semantic embeddings for class labels.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        normalize: bool = True,
    ):
        self.device = device
        self.normalize = normalize

        print(f"Loading CLIP model: {model_name}")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
        self.text_encoder.eval()

        self.embedding_dim = self.text_encoder.config.hidden_size
        print(f"CLIP text embedding dimension: {self.embedding_dim}")

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.text_encoder(**inputs)
        embeddings = outputs.pooler_output
        if self.normalize:
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings

    def get_class_embeddings(self, class_names: List[str], use_templates: bool = True) -> Dict[int, np.ndarray]:
        embeddings_dict = {}

        templates = (
            [
                "a photo of a {}.",
                "a photo of the {}.",
                "a picture of a {}.",
                "an image of a {}.",
                "{}.",
            ]
            if use_templates
            else ["{}"]
        )

        print(f"Extracting CLIP embeddings for {len(class_names)} classes ({self.embedding_dim}-dim)...")

        for class_idx, class_name in enumerate(tqdm(class_names)):
            clean_name = self._clean_class_name(class_name)
            prompts = [template.format(clean_name) for template in templates]
            embeddings = self.encode_text(prompts)
            avg_embedding = embeddings.mean(dim=0)
            embeddings_dict[class_idx] = avg_embedding.cpu().numpy()

        return embeddings_dict

    @staticmethod
    def _clean_class_name(name: str) -> str:
        name = name.replace("_", " ").lower()
        name = " ".join(name.split())
        return name


class EmbeddingManager:
    """
    Unified Embedding Manager.
    Handles CLIP text embeddings, with disk caching.
    """

    def __init__(self, config: dict):
        self.config = config
        self.embedding_type = config["embeddings"]["type"]
        if self.embedding_type != "clip":
            raise ValueError(f"embeddings.type must be 'clip', got {self.embedding_type!r}")
        self.device = config["experiment"]["device"]

        self.clip_embedder = CLIPTextEmbedder(
            model_name=config["embeddings"]["clip_model"],
            device=self.device,
            cache_dir=config["embeddings"]["clip_cache_dir"],
            normalize=config["embeddings"]["normalize"],
        )

    def get_embeddings(
        self,
        class_names: List[str],
        class_indices: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, int]:
        cache_file = Path(self.config["paths"]["cache_dir"]) / f"embeddings_{self.embedding_type}.pkl"

        if cache_file.exists():
            print(f"Loading cached {self.embedding_type} embeddings...")
            with open(cache_file, "rb") as f:
                embeddings_dict = pickle.load(f)
        else:
            embeddings_dict = self.clip_embedder.get_class_embeddings(class_names)

            Path(self.config["paths"]["cache_dir"]).mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(embeddings_dict, f)
            print(f"Cached embeddings to {cache_file}")

        if class_indices is not None:
            embeddings_list = [embeddings_dict[idx] for idx in class_indices]
        else:
            embeddings_list = [embeddings_dict[idx] for idx in range(len(class_names))]

        embeddings = torch.tensor(np.stack(embeddings_list), dtype=torch.float32, device=self.device)
        embedding_dim = embeddings.shape[1]

        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embedding dimension: {embedding_dim}")

        return embeddings, embedding_dim
