from __future__ import annotations

import torch
from PIL import Image

from fashion_finder.data.transforms import build_eval_transform, build_train_transform


def test_train_transform_outputs_normalized_tensor() -> None:
    image = Image.new("RGB", (300, 400), color=(128, 128, 128))
    transform = build_train_transform(image_size=224)
    tensor = transform(image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.min() >= -1.05
    assert tensor.max() <= 1.05


def test_eval_transform_centered_crop() -> None:
    image = Image.new("RGB", (300, 400), color=(255, 255, 255))
    transform = build_eval_transform(image_size=224)
    tensor = transform(image)
    assert tensor.shape == (3, 224, 224)
