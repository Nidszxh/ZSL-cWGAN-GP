"""
Semantic Embeddings Module — CLIP and GloVe text embeddings for CIFAR-100 class labels.

Supports:
- Single CLIP model (ViT-L/14, 768-dim)
- CLIP ensemble (average over multiple CLIP models)
- GloVe (legacy, for comparison)
"""

import torch
import numpy as np
import pickle
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from transformers import CLIPTokenizer, CLIPTextModel

import requests
import zipfile


class CLIPTextEmbedder:
    """
    CLIP Text Embedding Extractor.
    Uses pretrained CLIP text encoder to extract semantic embeddings for class labels.
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

    def get_class_embeddings(
        self, class_names: List[str], use_templates: bool = True
    ) -> Dict[int, np.ndarray]:
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


class CLIPEnsembleEmbedder:
    """
    Ensemble of multiple CLIP models for richer semantic embeddings.
    Averages embeddings from multiple CLIP models after L2-normalization.
    """

    def __init__(
        self,
        model_names: List[str],
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        normalize: bool = True,
    ):
        self.embedders = []
        self.device = device
        self.normalize = normalize

        for name in model_names:
            embedder = CLIPTextEmbedder(
                model_name=name,
                device=device,
                cache_dir=cache_dir,
                normalize=normalize,
            )
            self.embedders.append(embedder)

        self.embedding_dim = self.embedders[0].embedding_dim

    def get_class_embeddings(
        self, class_names: List[str], use_templates: bool = True
    ) -> Dict[int, np.ndarray]:
        all_embeddings = {}
        for embedder in self.embedders:
            emb_dict = embedder.get_class_embeddings(class_names, use_templates)
            for idx, emb in emb_dict.items():
                if idx not in all_embeddings:
                    all_embeddings[idx] = []
                all_embeddings[idx].append(emb)

        ensemble_dict = {}
        for idx, emb_list in all_embeddings.items():
            stacked = np.stack(emb_list)
            normed = stacked / (np.linalg.norm(stacked, axis=-1, keepdims=True) + 1e-8)
            avg = normed.mean(axis=0)
            avg = avg / (np.linalg.norm(avg) + 1e-8)
            ensemble_dict[idx] = avg

        return ensemble_dict


class GloVeEmbedder:
    """
    GloVe Embedding Extractor (Legacy).
    For comparison purposes or fallback.
    """

    def __init__(self, glove_file: str = "glove.6B.300d.txt", cache_dir: str = "./cache", embedding_dim: int = 300):
        self.glove_file = glove_file
        self.cache_dir = Path(cache_dir)
        self.embedding_dim = embedding_dim
        self.embeddings_dict = None

    def load_glove(self) -> Dict[str, np.ndarray]:
        cache_file = self.cache_dir / "glove_cache.pkl"
        if cache_file.exists():
            print(f"Loading cached GloVe embeddings from {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        if not Path(self.glove_file).exists():
            self._download_glove()

        print(f"Loading GloVe embeddings from {self.glove_file}...")
        embeddings_dict = {}
        with open(self.glove_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading GloVe"):
                values = line.strip().split()
                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)
                embeddings_dict[word] = vector

        print(f"Loaded {len(embeddings_dict)} GloVe vectors")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings_dict, f)

        return embeddings_dict

    def _download_glove(self):
        print("Downloading GloVe embeddings...")
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with open("glove.6B.zip", "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        with zipfile.ZipFile("glove.6B.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        print("GloVe downloaded and extracted")

    def get_class_embeddings(self, class_names: List[str]) -> Dict[int, np.ndarray]:
        if self.embeddings_dict is None:
            self.embeddings_dict = self.load_glove()

        embeddings = {}
        print(f"Creating GloVe embeddings for {len(class_names)} classes...")

        for class_idx, class_name in enumerate(class_names):
            clean_name = self._clean_label(class_name)
            if clean_name in self.embeddings_dict:
                embeddings[class_idx] = self.embeddings_dict[clean_name]
            else:
                words = clean_name.split()
                found_vectors = [
                    self.embeddings_dict[word] for word in words if word in self.embeddings_dict
                ]
                if found_vectors:
                    embeddings[class_idx] = np.mean(found_vectors, axis=0)
                else:
                    print(f"Warning: '{class_name}' not found in GloVe")
                    rng = np.random.RandomState(42 + class_idx)
                    embeddings[class_idx] = rng.normal(
                        scale=0.6, size=self.embedding_dim
                    ).astype(np.float32)

        return embeddings

    @staticmethod
    def _clean_label(label: str) -> str:
        return label.translate(str.maketrans("", "", string.punctuation)).lower()


class EmbeddingManager:
    """
    Unified Embedding Manager.
    Handles CLIP, CLIP ensemble, and GloVe embeddings, with caching.
    """

    def __init__(self, config: dict):
        self.config = config
        self.embedding_type = config["embeddings"]["type"]
        self.device = config["experiment"]["device"]

        self.clip_embedder = None
        self.glove_embedder = None

        if self.embedding_type == "clip":
            self.clip_embedder = CLIPTextEmbedder(
                model_name=config["embeddings"]["clip_model"],
                device=self.device,
                cache_dir=config["embeddings"]["clip_cache_dir"],
                normalize=config["embeddings"]["normalize"],
            )
        elif self.embedding_type == "clip_ensemble":
            model_list = config["embeddings"].get(
                "clip_models_ensemble",
                [config["embeddings"]["clip_model"]],
            )
            self.clip_embedder = CLIPEnsembleEmbedder(
                model_names=model_list,
                device=self.device,
                cache_dir=config["embeddings"]["clip_cache_dir"],
                normalize=config["embeddings"]["normalize"],
            )
        elif self.embedding_type in ["glove", "both"]:
            self.glove_embedder = GloVeEmbedder(
                glove_file=config["embeddings"]["glove_path"],
                cache_dir=config["paths"]["cache_dir"],
                embedding_dim=config["embeddings"]["glove_dim"],
            )

    def get_embeddings(
        self, class_names: List[str], class_indices: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, int]:
        cache_file = Path(self.config["paths"]["cache_dir"]) / f"embeddings_{self.embedding_type}.pkl"

        if cache_file.exists():
            print(f"Loading cached {self.embedding_type} embeddings...")
            with open(cache_file, "rb") as f:
                embeddings_dict = pickle.load(f)
        else:
            if self.embedding_type in ["clip", "clip_ensemble"]:
                embeddings_dict = self.clip_embedder.get_class_embeddings(class_names)
            elif self.embedding_type == "glove":
                embeddings_dict = self.glove_embedder.get_class_embeddings(class_names)
            elif self.embedding_type == "both":
                clip_emb = self.clip_embedder.get_class_embeddings(class_names)
                glove_emb = self.glove_embedder.get_class_embeddings(class_names)
                embeddings_dict = {
                    idx: np.concatenate([clip_emb[idx], glove_emb[idx]])
                    for idx in range(len(class_names))
                }

            Path(self.config["paths"]["cache_dir"]).mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(embeddings_dict, f)
            print(f"Cached embeddings to {cache_file}")

        if class_indices is not None:
            embeddings_list = [embeddings_dict[idx] for idx in class_indices]
        else:
            embeddings_list = [embeddings_dict[idx] for idx in range(len(class_names))]

        embeddings = torch.tensor(
            np.stack(embeddings_list), dtype=torch.float32, device=self.device
        )
        embedding_dim = embeddings.shape[1]

        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embedding dimension: {embedding_dim}")

        return embeddings, embedding_dim
