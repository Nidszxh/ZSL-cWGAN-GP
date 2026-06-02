from pathlib import Path
from typing import Union

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_fidelity import calculate_metrics


def save_real_images(dataset, num_samples: int = 5000, save_dir: str = "results/real"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(dataset, batch_size=100, shuffle=False)
    count = 0

    print(f"Saving {num_samples} real images for evaluation...")
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Saving real images"):
            images = (images + 1) / 2
            for img in images:
                if count >= num_samples:
                    break
                vutils.save_image(img, save_path / f"real_{count:05d}.png")
                count += 1
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
):
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
            fake_images = (fake_images + 1) / 2

            for j, img in enumerate(fake_images):
                vutils.save_image(img, save_path / f"fake_{i + j:05d}.png")

    generator.train()
    print(f"Saved images to {save_dir}")
    return save_dir


def compute_fid(real_dir: str, fake_dir: str) -> dict:
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
        print(f"Error calculating metrics: {e}")
        return {"fid": float("inf"), "is_mean": 0, "is_std": 0, "kid_mean": 0}
