"""
Test script for CLIP embeddings
Day 1: Verify CLIP integration works
"""

import torch
import yaml
from torchvision import datasets
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.embeddings import CLIPTextEmbedder, GloVeEmbedder, EmbeddingManager, compare_embeddings

def test_clip_basic():
    """Test basic CLIP embedding extraction"""
    print("="*70)
    print("TEST 1: Basic CLIP Embedding Extraction")
    print("="*70)
    
    embedder = CLIPTextEmbedder(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Test on sample texts
    texts = ["a cat", "a dog", "an airplane", "a truck", "a ship"]
    embeddings = embedder.encode_text(texts)
    
    print(f"✓ Extracted embeddings for {len(texts)} texts")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Device: {embeddings.device}")
    print(f"  Dtype: {embeddings.dtype}")
    print(f"  Sample norm: {embeddings[0].norm().item():.4f}")
    
    # Check normalized
    norms = embeddings.norm(dim=-1)
    print(f"  All normalized: {torch.allclose(norms, torch.ones_like(norms), atol=1e-5)}")
    
    return True

def test_cifar100_embeddings():
    """Test CLIP embeddings on CIFAR-100"""
    print("\n" + "="*70)
    print("TEST 2: CIFAR-100 Class Embeddings")
    print("="*70)
    
    # Load CIFAR-100
    cifar100 = datasets.CIFAR100(root='./data', download=True, train=False)
    class_names = cifar100.classes
    
    print(f"CIFAR-100 has {len(class_names)} classes")
    print(f"Sample classes: {class_names[:5]}")
    
    # Get CLIP embeddings
    embedder = CLIPTextEmbedder(device="cuda" if torch.cuda.is_available() else "cpu")
    embeddings_dict = embedder.get_class_embeddings(class_names)
    
    print(f"\n✓ Extracted embeddings for all {len(embeddings_dict)} classes")
    print(f"  Embedding dimension: {embeddings_dict[0].shape[0]}")
    print(f"  Sample classes:")
    for i in range(5):
        print(f"    {i}: {class_names[i]:20s} -> shape {embeddings_dict[i].shape}")
    
    return embeddings_dict

def test_glove_comparison():
    """Compare CLIP vs GloVe"""
    print("\n" + "="*70)
    print("TEST 3: CLIP vs GloVe Comparison")
    print("="*70)
    
    cifar100 = datasets.CIFAR100(root='./data', download=True, train=False)
    class_names = cifar100.classes
    
    # Test on subset
    test_classes = class_names[:10]
    
    # CLIP
    clip_embedder = CLIPTextEmbedder()
    clip_emb = clip_embedder.get_class_embeddings(test_classes)
    
    print(f"✓ CLIP embeddings: {len(clip_emb)} classes, dim={clip_emb[0].shape[0]}")
    
    # GloVe (optional, might not be downloaded)
    try:
        glove_embedder = GloVeEmbedder()
        glove_emb = glove_embedder.get_class_embeddings(test_classes)
        print(f"✓ GloVe embeddings: {len(glove_emb)} classes, dim={glove_emb[0].shape[0]}")
        
        # Visualize comparison
        print("\nGenerating comparison visualization...")
        compare_embeddings(class_names, sample_classes=list(range(20)))
        
    except Exception as e:
        print(f"⚠ GloVe not available (this is OK): {e}")
    
    return True

def test_embedding_manager():
    """Test the unified EmbeddingManager"""
    print("\n" + "="*70)
    print("TEST 4: EmbeddingManager Integration")
    print("="*70)
    
    # Load config
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        print("⚠ Config file not found, using defaults")
        config = {
            'embeddings': {
                'type': 'clip',
                'clip_model': 'openai/clip-vit-base-patch32',
                'clip_cache_dir': './cache/clip',
                'normalize': True,
                'glove_path': 'glove.6B.300d.txt',
                'glove_dim': 300
            },
            'experiment': {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'paths': {
                'cache_dir': './cache'
            }
        }
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Get CIFAR-100 classes
    cifar100 = datasets.CIFAR100(root='./data', download=True, train=False)
    class_names = cifar100.classes
    
    # Initialize manager
    manager = EmbeddingManager(config)
    
    # Get embeddings for all classes
    all_embeddings, embedding_dim = manager.get_embeddings(class_names)
    
    print(f"✓ Extracted all embeddings via EmbeddingManager")
    print(f"  Total classes: {len(class_names)}")
    print(f"  Embedding shape: {all_embeddings.shape}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Device: {all_embeddings.device}")
    
    # Test seen/unseen split
    import numpy as np
    seen_indices = np.arange(80)
    unseen_indices = np.arange(80, 100)
    
    seen_embeddings, _ = manager.get_embeddings(class_names, seen_indices)
    unseen_embeddings, _ = manager.get_embeddings(class_names, unseen_indices)
    
    print(f"\n✓ Seen/Unseen split:")
    print(f"  Seen: {seen_embeddings.shape}")
    print(f"  Unseen: {unseen_embeddings.shape}")
    
    return True

def test_embedding_quality():
    """Test embedding quality via similarity"""
    print("\n" + "="*70)
    print("TEST 5: Embedding Quality (Similarity Check)")
    print("="*70)
    
    embedder = CLIPTextEmbedder()
    
    # Similar classes should have high similarity
    similar_pairs = [
        ("cat", "dog"),
        ("airplane", "rocket"),
        ("car", "truck"),
        ("apple", "orange"),
        ("tree", "forest")
    ]
    
    # Dissimilar classes should have low similarity
    dissimilar_pairs = [
        ("cat", "airplane"),
        ("dog", "clock"),
        ("tree", "television"),
        ("apple", "keyboard"),
        ("car", "butterfly")
    ]
    
    def compute_similarity(text1, text2):
        emb = embedder.encode_text([text1, text2])
        return (emb[0] @ emb[1]).item()
    
    print("\nSimilar pairs (should have HIGH similarity):")
    similar_sims = []
    for t1, t2 in similar_pairs:
        sim = compute_similarity(t1, t2)
        similar_sims.append(sim)
        print(f"  {t1:15s} <-> {t2:15s} : {sim:.4f}")
    
    print("\nDissimilar pairs (should have LOW similarity):")
    dissimilar_sims = []
    for t1, t2 in dissimilar_pairs:
        sim = compute_similarity(t1, t2)
        dissimilar_sims.append(sim)
        print(f"  {t1:15s} <-> {t2:15s} : {sim:.4f}")
    
    avg_similar = sum(similar_sims) / len(similar_sims)
    avg_dissimilar = sum(dissimilar_sims) / len(dissimilar_sims)
    
    print(f"\n✓ Average similar similarity: {avg_similar:.4f}")
    print(f"✓ Average dissimilar similarity: {avg_dissimilar:.4f}")
    print(f"✓ Difference: {avg_similar - avg_dissimilar:.4f}")
    
    if avg_similar > avg_dissimilar:
        print("✓ PASS: Similar pairs have higher similarity!")
    else:
        print("✗ FAIL: Something is wrong with embeddings")
    
    return avg_similar > avg_dissimilar

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("CLIP EMBEDDINGS TEST SUITE - DAY 1")
    print("="*70 + "\n")
    
    results = {}
    
    try:
        results['basic'] = test_clip_basic()
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        results['basic'] = False
    
    try:
        results['cifar100'] = test_cifar100_embeddings() is not None
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        results['cifar100'] = False
    
    try:
        results['comparison'] = test_glove_comparison()
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
        results['comparison'] = False
    
    try:
        results['manager'] = test_embedding_manager()
    except Exception as e:
        print(f"✗ Test 4 failed: {e}")
        results['manager'] = False
    
    try:
        results['quality'] = test_embedding_quality()
    except Exception as e:
        print(f"✗ Test 5 failed: {e}")
        results['quality'] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! CLIP embeddings are ready to use.")
        print("\nNext steps:")
        print("  1. Integrate into your GAN training (update Generator/Discriminator)")
        print("  2. Compare GAN training with CLIP vs GloVe")
        print("  3. Evaluate ZSL performance improvement")
    else:
        print("\n⚠ Some tests failed. Please fix before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)