from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from fashion_finder.data.fashion_iq import FashionIQDataset
from fashion_finder.data.mt_cir import MTCIRDataset
from fashion_finder.data.transforms import build_eval_transform


class _StubTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        prompt: str,
        padding: str,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict:
        import torch

        ids = torch.zeros(1, max_length, dtype=torch.long)
        ids[:, : min(len(prompt.split()), max_length)] = 1
        mask = (ids != 0).long()
        return {"input_ids": ids, "attention_mask": mask}


def test_fashion_iq_dataset_minimal(tmp_path: Path) -> None:
    captions_dir = tmp_path / "captions"
    captions_dir.mkdir()
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    captions = [
        {
            "candidate": "img_a",
            "target": "img_b",
            "captions": ["longer sleeves", "darker color"],
        }
    ]
    (captions_dir / "cap.dress.val.json").write_text(json.dumps(captions))
    for image_id in ("img_a", "img_b"):
        Image.new("RGB", (256, 256), color=(123, 45, 67)).save(images_dir / f"{image_id}.jpg")

    dataset = FashionIQDataset(
        root=tmp_path,
        categories=("dress",),
        split="val",
        tokenizer=_StubTokenizer(),
        transform=build_eval_transform(64),
        max_text_len=16,
    )
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["image1"].shape == (3, 64, 64)
    assert sample["image2"].shape == (3, 64, 64)
    assert sample["input_ids"].shape == (16,)


def test_mt_cir_dataset_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MTCIRDataset(root=tmp_path, tokenizer=_StubTokenizer())
