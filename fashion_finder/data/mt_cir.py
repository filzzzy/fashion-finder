from __future__ import annotations

import io
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


class MTCIRDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        tokenizer: Any,
        transform: Any | None = None,
        max_text_len: int = 64,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_text_len = max_text_len

        jsonl_path = self.root / "mt_cir_train.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"Expected MT-CIR jsonl at {jsonl_path}. "
                "Run `fashion-finder download-data` first."
            )
        self.samples: list[dict[str, Any]] = []
        with jsonl_path.open() as handle:
            for line in handle:
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, payload: Any) -> Image.Image:
        if isinstance(payload, str):
            return Image.open(self.root / payload).convert("RGB")
        if isinstance(payload, dict) and "bytes" in payload:
            return Image.open(io.BytesIO(payload["bytes"])).convert("RGB")
        if isinstance(payload, dict) and "path" in payload:
            return Image.open(payload["path"]).convert("RGB")
        raise TypeError(f"Unsupported image payload: {type(payload)!r}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        image_ref = self._load_image(item["reference_image"])
        image_tgt = self._load_image(item["target_image"])
        text = item.get("modifier", "") or item.get("caption", "")

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
            "path1": str(item.get("reference_id", idx)),
            "path2": str(item.get("target_id", idx)),
        }


def collate_skip_missing(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = batch[0].keys()
    out: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            out[key] = torch.stack(values)
        else:
            out[key] = values
    return out
