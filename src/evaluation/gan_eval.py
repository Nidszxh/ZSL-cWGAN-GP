from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Union

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch_fidelity import calculate_metrics


def _save_images_concurrent(
    images: torch.Tensor,
    save_dir: str,
    name_fn: Callable[[int], str],
    max_workers: int = 8,
) -> None:
    """Save CPU image tensors to individual PNGs using a thread pool.

    Each image stays a separate file so torch-fidelity treats it as one sample —
    tiling images into grids would corrupt FID/IS/KID counts. PNG encoding is
    CPU-bound and per-image independent, so threads give a near-linear speedup
    over the old sequential vutils.save_image loop (L2-F5).
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(vutils.save_image, img, save_path / name_fn(j)) for j, img in enumerate(images)]
        for future in futures:
            future.result()


def save_real_images(
    dataset: Dataset,
    num_samples: int = 5000,
    save_dir: str = "results/real",
) -> str:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(dataset, batch_size=100, shuffle=False)
    count = 0

    print(f"Saving {num_samples} real images for evaluation...")
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Saving real images"):
            images = ((images + 1) / 2).clamp(0, 1)
            if count + images.size(0) > num_samples:
                images = images[: num_samples - count]
            _save_images_concurrent(images, save_dir, lambda j: f"real_{count + j:05d}.png")
            count += images.size(0)
            if count >= num_samples:
                break

    print(f"Saved {count} real images to {save_dir}")
    return save_dir


def save_fake_images(
    generator: torch.nn.Module,
    epoch: Union[int, str],
    device: torch.device,
    seen_embeddings: torch.Tensor,
    num_samples: int = 2000,
    batch_size: int = 128,
    nz: int = 128,
    num_seen_classes: int = 80,
    save_dir: str = None,
) -> str:
    if save_dir is None:
        save_dir = f"results/fake_epoch{epoch}"
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    generator.eval()
    print(f"Generating {num_samples} fake images for evaluation...")

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Generating images"):
            curr_bs = min(batch_size, num_samples - i)
            if curr_bs <= 0:
                break

            z = torch.randn(curr_bs, nz, device=device)
            labels = torch.randint(0, num_seen_classes, (curr_bs,), device=device)
            fake_images = generator(z, labels, seen_embeddings).cpu()
            fake_images = ((fake_images + 1) / 2).clamp(0, 1)

            _save_images_concurrent(fake_images, save_dir, lambda j: f"fake_{i + j:05d}.png")

    generator.train()
    print(f"Saved images to {save_dir}")
    return save_dir


def compute_fid(real_dir: str, fake_dir: str) -> dict[str, float]:
    print(f"Calculating metrics between {real_dir} and {fake_dir}")
    try:
        metrics = calculate_metrics(
            input1=real_dir,
            input2=fake_dir,
            cuda=torch.cuda.is_available(),
            fid=True,
            isc=True,
            kid=True,
            verbose=False,
        )
        return {
            "fid": metrics["frechet_inception_distance"],
            "is_mean": metrics.get("inception_score_mean", 0),
            "is_std": metrics.get("inception_score_std", 0),
            "kid_mean": metrics.get("kernel_inception_distance_mean", 0),
        }
    except Exception as e:
        # L1-F21: fail fast — an inf FID silently skipped best_model.pth saves.
        raise RuntimeError(f"FID/metrics computation failed for {fake_dir} vs {real_dir}: {e}") from e
