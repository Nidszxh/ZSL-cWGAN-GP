"""
Gradio Demo for ZSL-cWGAN-GP

Requires a trained model checkpoint at checkpoints/best_zsl_classifier.pth.
Run `python main.py` first to train one, then launch this demo.
"""

import json
import warnings
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torchvision import datasets

from models.zsl_classifier import ZSLClassifier

warnings.filterwarnings("ignore", message=".*align should be passed as Python or NumPy boolean.*")


def load_model(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_file = Path(config["paths"]["cache_dir"]) / "class_split.json"
    if not split_file.exists():
        return None, None, device, config

    with open(split_file) as f:
        split_data = json.load(f)
    unseen_indices = split_data["unseen"]

    ckpt_path = Path(config["paths"]["checkpoints_dir"]) / "best_zsl_classifier.pth"
    if not ckpt_path.exists():
        return None, None, device, config

    num_unseen = len(unseen_indices)
    classifier = ZSLClassifier(num_unseen).to(device)
    classifier.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    classifier.eval()

    cifar100 = datasets.CIFAR100(root=config["paths"]["data_root"], download=True)
    unseen_class_names = [cifar100.classes[i] for i in unseen_indices]

    return classifier, unseen_class_names, device, config


CLASSIFIER, CLASS_NAMES, DEVICE, CONFIG = None, None, None, None


def initialize():
    global CLASSIFIER, CLASS_NAMES, DEVICE, CONFIG
    CLASSIFIER, CLASS_NAMES, DEVICE, CONFIG = load_model()
    return CLASSIFIER is not None


initialize()

transform = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def predict(image: Optional[np.ndarray]):
    if image is None or CLASSIFIER is None:
        return None, "Upload an image first."

    pil_img = Image.fromarray(image.astype("uint8")).convert("RGB")
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = CLASSIFIER(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top5_idx = np.argsort(probs)[::-1][:5]
    labels = [CLASS_NAMES[i] for i in top5_idx]
    scores = [float(probs[i]) for i in top5_idx]

    top1 = f"**Prediction: {labels[0]}** ({scores[0]:.1%})"

    chart_data = pd.DataFrame({"label": labels, "score": scores})

    return chart_data, top1


with gr.Blocks(title="ZSL-cWGAN-GP Demo") as demo:
    gr.Markdown("""
    # ZSL-cWGAN-GP: Zero-Shot Learning Demo

    Upload an image from one of the **20 unseen CIFAR-100 classes** — the model was trained on
    the other 80 classes and has never seen these during training.

    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Image", type="numpy", height=300)
            predict_btn = gr.Button("Predict", variant="primary", size="lg")

        with gr.Column(scale=1):
            prediction_text = gr.Markdown("Upload an image and click Predict.")
            output_plot = gr.BarPlot(
                x="score",
                y="label",
                title="Top-5 Predictions",
                height=250,
                sort="-x",
            )

    predict_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_plot, prediction_text],
    )

    if CLASSIFIER is not None:
        gr.Markdown(f"""
        ### Model Info
        - **Unseen classes:** {', '.join(CLASS_NAMES)}
        - **Device:** {DEVICE}
        """)
    else:
        gr.Warning("No trained model found. Run `python main.py` first.")


if __name__ == "__main__":
    demo.launch(share=False)
