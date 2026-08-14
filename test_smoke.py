"""Smoke tests: exercise the whole pipeline with random weights.

No pretrained downloads and no trained checkpoint, so CI stays fast and does
not depend on external hosting.
"""

import numpy as np
import torch
from PIL import Image

from inference import (
    CLASSES,
    IMG_SIZE,
    build_model,
    gradcam,
    load_model,
    max_entropy,
    mc_predict,
    overlay,
    predict,
    preprocess,
)


def _dummy_image() -> Image.Image:
    arr = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_class_order_is_fixed():
    # ImageFolder sorts alphabetically; the checkpoint head is tied to this order.
    assert CLASSES == ["glioma", "meningioma", "notumor", "pituitary"]


def test_preprocess_shapes():
    rgb01, tensor = preprocess(_dummy_image())
    assert rgb01.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert rgb01.min() >= 0.0 and rgb01.max() <= 1.0
    assert tuple(tensor.shape) == (1, 3, IMG_SIZE, IMG_SIZE)


def test_model_head_is_four_way():
    model = build_model()
    assert model.head[-1].out_features == len(CLASSES)


def test_predict_returns_probabilities():
    model = build_model()
    _, tensor = preprocess(_dummy_image())
    probs = predict(model, tensor)
    assert probs.shape == (len(CLASSES),)
    assert abs(float(probs.sum()) - 1.0) < 1e-4


def test_mc_dropout_is_stochastic_and_bounded():
    model = build_model()
    _, tensor = preprocess(_dummy_image())
    mean, entropy, variance = mc_predict(model, tensor, passes=5)

    assert mean.shape == (len(CLASSES),)
    assert abs(float(mean.sum()) - 1.0) < 1e-4
    assert 0.0 <= entropy <= max_entropy(len(CLASSES)) + 1e-6
    assert variance >= 0.0
    # dropout must be switched back off once MC sampling has finished
    assert not any(m.training for m in model.modules())


def test_gradcam_and_overlay():
    model = build_model()
    rgb01, tensor = preprocess(_dummy_image())
    cam, idx = gradcam(model, tensor)

    assert cam.shape == (IMG_SIZE, IMG_SIZE)
    assert cam.min() >= 0.0 and cam.max() <= 1.0
    assert 0 <= idx < len(CLASSES)

    blended = overlay(rgb01, cam)
    assert blended.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert blended.dtype == np.uint8


def test_gradcam_does_not_leak_hooks():
    model = build_model()
    _, tensor = preprocess(_dummy_image())
    for _ in range(3):
        gradcam(model, tensor)
    assert len(model.features._forward_hooks) == 0
    assert len(model.features._backward_hooks) == 0


def test_checkpoint_round_trip(tmp_path):
    """The notebook saves a dict; load_model must recover backbone and classes."""
    model = build_model()
    ckpt = tmp_path / "best_resnet50.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "best_val": 0.123,
            "classes": CLASSES,
            "backbone": "resnet50",
        },
        ckpt,
    )

    loaded, classes = load_model(ckpt)
    assert classes == CLASSES
    assert loaded.backbone_name == "resnet50"

    _, tensor = preprocess(_dummy_image())
    assert predict(loaded, tensor).shape == (len(CLASSES),)
