from __future__ import annotations

import timm
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM


class ImageAdapter(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)


class CoLLMArchitecture(nn.Module):
    def __init__(
        self,
        vision_arch: str,
        llm_arch: str,
        embed_dim: int,
        vocab_size: int,
        logit_scale: float,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vision_encoder = timm.create_model(
            vision_arch, pretrained=True, num_classes=0
        )
        vision_dim = self.vision_encoder.num_features

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_arch, torch_dtype=torch.float16, device_map=None
        )
        llm_dim = self.llm.config.hidden_size
        self.llm.resize_token_embeddings(vocab_size)

        if use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            )
            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.gradient_checkpointing_enable()

        self.image_adapter = ImageAdapter(vision_dim, llm_dim)
        self.llm_proj = nn.Linear(llm_dim, embed_dim, bias=False)
        self.target_proj = nn.Linear(vision_dim, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(image)

    def forward_composed(
        self,
        image_ref: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        img_token_id: int,
    ) -> torch.Tensor:
        batch_size = image_ref.size(0)
        device = image_ref.device

        img_features = self.encode_image(image_ref)
        img_token = self.image_adapter(img_features).unsqueeze(1)

        base = self.llm.get_base_model()
        word_embeddings = base.model.embed_tokens(input_ids)
        img_positions = (input_ids == img_token_id).nonzero(as_tuple=True)
        inputs_embeds = word_embeddings.clone()
        inputs_embeds[img_positions[0], img_positions[1]] = img_token.squeeze(1).to(
            inputs_embeds.dtype
        )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        sequence_lengths = attention_mask.sum(dim=1) - 1
        last_hidden_states = outputs.hidden_states[-1]
        pooled_output = last_hidden_states[
            torch.arange(batch_size, device=device), sequence_lengths
        ]
        return self.llm_proj(pooled_output)

    def get_target_embedding(self, image_target: torch.Tensor) -> torch.Tensor:
        features = self.encode_image(image_target)
        return self.target_proj(features)
