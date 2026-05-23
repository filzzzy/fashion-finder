from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import AutoTokenizer

from fashion_finder.models.composition_module import CompositionLitModule


class VisionEmbedder(torch.nn.Module):
    def __init__(self, module: CompositionLitModule) -> None:
        super().__init__()
        self.vision = module.model.vision_encoder
        self.target_proj = module.model.target_proj

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.vision(images)
        projected = self.target_proj(features)
        return F.normalize(projected, dim=-1)


class ComposerEmbedder(torch.nn.Module):
    def __init__(self, module: CompositionLitModule) -> None:
        super().__init__()
        self.module = module

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        composed = self.module.model.forward_composed(
            images, input_ids, attention_mask, self.module.img_token_id
        )
        return F.normalize(composed, dim=-1)


def export_onnx(cfg: DictConfig) -> dict[str, str]:
    checkpoint_path = Path(cfg.inference.checkpoint_path)
    output_dir = Path(cfg.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    module = CompositionLitModule.load_from_checkpoint(str(checkpoint_path), cfg=cfg, strict=False)
    module.eval()
    module.cpu()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_arch)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    dummy_images = torch.randn(1, 3, cfg.data.image_size, cfg.data.image_size)
    sample_text = (
        "Instruct: Find the image that matches the query.\n"
        "Query:\nImage: <image>\nText: longer sleeves"
    )
    encoded = tokenizer(
        sample_text,
        padding="max_length",
        truncation=True,
        max_length=cfg.data.max_text_len,
        return_tensors="pt",
    )

    vision_out = output_dir / cfg.inference.vision_filename
    composer_out = output_dir / cfg.inference.composer_filename

    vision_only = VisionEmbedder(module).eval()
    composer = ComposerEmbedder(module).eval()

    dynamic_axes_vision = (
        {"images": {0: "batch"}, "embedding": {0: "batch"}} if cfg.inference.dynamic_batch else None
    )
    dynamic_axes_composer = (
        {
            "images": {0: "batch"},
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "embedding": {0: "batch"},
        }
        if cfg.inference.dynamic_batch
        else None
    )

    torch.onnx.export(
        vision_only,
        (dummy_images,),
        str(vision_out),
        input_names=["images"],
        output_names=["embedding"],
        dynamic_axes=dynamic_axes_vision,
        opset_version=cfg.inference.opset,
        do_constant_folding=True,
    )

    torch.onnx.export(
        composer,
        (dummy_images, encoded["input_ids"], encoded["attention_mask"]),
        str(composer_out),
        input_names=["images", "input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes=dynamic_axes_composer,
        opset_version=cfg.inference.opset,
        do_constant_folding=True,
    )

    return {"vision": str(vision_out), "composer": str(composer_out)}
