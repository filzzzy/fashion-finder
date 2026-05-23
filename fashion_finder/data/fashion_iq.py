from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_TOKEN = "<image>"
PROMPT_TEMPLATE = (
    "Instruct: Find the image that matches the query.\n"
    "Query:\nImage: {image_token}\nText: {text}"
)


class FashionIQDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        categories: tuple[str, ...],
        split: str,
        tokenizer: Any,
        transform: Any | None = None,
        max_text_len: int = 64,
    ) -> None:
        self.root = Path(root)
        self.categories = categories
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_text_len = max_text_len

        self.samples: list[dict[str, Any]] = []
        captions_dir = self.root / "captions"
        for category in categories:
            cap_path = captions_dir / f"cap.{category}.{split}.json"
            if not cap_path.exists():
                continue
            data = json.loads(cap_path.read_text())
            for entry in data:
                self.samples.append(
                    {
                        "candidate": entry["candidate"],
                        "target": entry["target"],
                        "captions": entry["captions"],
                        "category": category,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _open(self, image_id: str) -> Image.Image:
        path = self.root / "images" / f"{image_id}.jpg"
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        image_ref = self._open(item["candidate"])
        image_tgt = self._open(item["target"])
        text = " and ".join(item["captions"])

        if self.transform is not None:
            image_ref = self.transform(image_ref)
            image_tgt = self.transform(image_tgt)

        prompt = PROMPT_TEMPLATE.format(image_token=IMG_TOKEN, text=text)
        encoded = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        return {
            "image1": image_ref,
            "image2": image_tgt,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "raw_text": text,
            "path1": item["candidate"],
            "path2": item["target"],
            "category": item["category"],
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = batch[0].keys()
    out: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            out[key] = torch.stack(values)
        else:
            out[key] = values
    return out
