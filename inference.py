"""Inference, Grad-CAM and MC-Dropout uncertainty for the brain tumour MRI classifier.

Mirrors the model definition, preprocessing and Grad-CAM implementation in the
training notebook so the app reproduces the dissertation results exactly.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models

# --- must match the notebook exactly -----------------------------------
# ImageFolder sorts class folders alphabetically; the checkpoint also stores
# the list, and load_model() prefers that over this fallback.
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

FULL_NAME = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumour",
    "pituitary": "Pituitary tumour",
}

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DROPOUT_P = 0.4
MC_PASSES = 20
# -----------------------------------------------------------------------

# Free hosting gives ~1 GB of RAM and 1-2 cores; extra threads only add memory
# arenas here, they do not speed up a single 224x224 image.
torch.set_num_threads(1)

CKPT_PATH = Path(os.environ.get("CKPT_PATH", "best_resnet50.pt"))


def make_backbone(name: str, pretrained: bool = False) -> tuple[nn.Module, int]:
    """Return (feature_extractor, feature_channels) for the chosen backbone."""
    w = "DEFAULT" if pretrained else None
    if name == "resnet50":
        net = models.resnet50(weights=w)
        features = nn.Sequential(*list(net.children())[:-2])  # drop avgpool + fc
        feat_dim = 2048
    elif name == "vgg16":
        net = models.vgg16(weights=w)
        features = net.features
        feat_dim = 512
    elif name == "efficientnet_b0":
        net = models.efficientnet_b0(weights=w)
        features = net.features
        feat_dim = 1280
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return features, feat_dim


class TumorClassifier(nn.Module):
    """Pretrained backbone + MC-Dropout head. Identical to the notebook."""

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 4,
        dropout_p: float = DROPOUT_P,
        pretrained: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.features, feat_dim = make_backbone(backbone, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)  # final conv feature map (Grad-CAM target)
        p = self.pool(f)
        return self.head(p)


def build_model(
    backbone: str = "resnet50", num_classes: int = 4, pretrained: bool = False
) -> TumorClassifier:
    """Build the architecture only. No checkpoint, so this is safe in CI."""
    return TumorClassifier(backbone, num_classes, DROPOUT_P, pretrained=pretrained)


def download_checkpoint(url: str, dest: Path = CKPT_PATH) -> Path:
    """Fetch the trained weights once, if they are not already on disk."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def load_model(ckpt_path: Path | str = CKPT_PATH) -> tuple[TumorClassifier, list[str]]:
    """Load a notebook checkpoint. Returns (model, class_names).

    The notebook saves a dict with 'model', 'classes' and 'backbone' keys, so the
    architecture and label order are recovered from the file itself rather than
    guessed. A bare state dict is also accepted as a fallback.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        backbone = ckpt.get("backbone", "resnet50")
        classes = list(ckpt.get("classes") or CLASSES)
    else:
        state = ckpt
        backbone = "resnet50"
        classes = list(CLASSES)

    model = build_model(backbone, num_classes=len(classes))
    model.load_state_dict(state)
    model.eval()
    return model, classes


def preprocess(image: Image.Image) -> tuple[np.ndarray, torch.Tensor]:
    """Return (rgb01, input_tensor).

    rgb01 is the HxWx3 float image in [0, 1] used for the heatmap overlay;
    input_tensor is the normalised 1x3xHxW batch fed to the model.
    """
    rgb = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    rgb01 = np.asarray(rgb).astype("float32") / 255.0

    tensor = torch.from_numpy(np.ascontiguousarray(rgb01.transpose(2, 0, 1))).float()
    for c in range(3):
        tensor[c] = (tensor[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return rgb01, tensor.unsqueeze(0)


def predict(model: nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    """Deterministic prediction with dropout off. Returns probabilities."""
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
    return torch.softmax(logits, dim=1)[0].cpu().numpy()


def _enable_dropout(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_predict(
    model: nn.Module, input_tensor: torch.Tensor, passes: int = MC_PASSES
) -> tuple[np.ndarray, float, float]:
    """Monte Carlo Dropout. Returns (mean_probs, predictive_entropy, variance).

    Dropout stays active at inference and the model is run `passes` times;
    the spread across runs is the uncertainty estimate.
    """
    model.eval()
    _enable_dropout(model)  # keep ONLY dropout stochastic
    with torch.no_grad():
        probs = torch.stack(
            [torch.softmax(model(input_tensor), dim=1) for _ in range(passes)], dim=0
        )
    mean = probs.mean(0)
    entropy = -(mean * torch.log(mean + 1e-12)).sum(1)
    variance = probs.var(0).sum(1)
    model.eval()  # switch dropout back off
    return mean[0].cpu().numpy(), float(entropy[0]), float(variance[0])


def max_entropy(num_classes: int) -> float:
    """Entropy of a uniform distribution - the worst possible case."""
    return float(np.log(num_classes))


def gradcam(
    model: TumorClassifier, input_tensor: torch.Tensor, class_idx: int | None = None
) -> tuple[np.ndarray, int]:
    """Grad-CAM over the final conv block. Returns (cam HxW in [0,1], class_idx).

    Hooks are removed afterwards so repeated calls against a cached model do not
    accumulate handles.
    """
    model.eval()
    store: dict[str, torch.Tensor] = {}

    def fwd(_module, _inp, out):
        store["activations"] = out.detach()

    def bwd(_module, _grad_in, grad_out):
        store["gradients"] = grad_out[0].detach()

    h1 = model.features.register_forward_hook(fwd)
    h2 = model.features.register_full_backward_hook(bwd)
    try:
        x = input_tensor.clone().requires_grad_(True)
        logits = model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        model.zero_grad()
        logits[0, int(class_idx)].backward()

        weights = store["gradients"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * store["activations"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.astype("float32"), int(class_idx)


def overlay(rgb01: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a jet-coloured heatmap over the image. Returns uint8 RGB."""
    import matplotlib

    heat = matplotlib.colormaps["jet"](cam)[:, :, :3]  # drop alpha channel
    blended = np.clip((1 - alpha) * rgb01 + alpha * heat, 0, 1)
    return (blended * 255).astype("uint8")
