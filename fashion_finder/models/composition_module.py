from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import DictConfig

from fashion_finder.models.architecture import CoLLMArchitecture
from fashion_finder.utils.visualization import make_retrieval_grid

RECALL_KS = (1, 5, 10, 50)


class CompositionLitModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(dict(cfg))
        self.cfg = cfg

        self.model = CoLLMArchitecture(
            vision_arch=cfg.model.vision_arch,
            llm_arch=cfg.model.text_arch,
            embed_dim=cfg.model.embed_dim,
            vocab_size=cfg.model.vocab_size,
            logit_scale=cfg.model.initial_logit_scale,
            use_lora=cfg.model.use_lora,
            lora_r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
        )

        for param in self.model.vision_encoder.parameters():
            param.requires_grad = False

        self.img_token_id: int = cfg.model.img_token_id
        self._validation_outputs: list[dict[str, Any]] = []

    def _compose(
        self, image_ref: torch.Tensor, input_ids: torch.Tensor, attn: torch.Tensor
    ) -> torch.Tensor:
        return self.model.forward_composed(image_ref, input_ids, attn, self.img_token_id)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        composed = self._compose(batch["image1"], batch["input_ids"], batch["attention_mask"])
        target = self.model.get_target_embedding(batch["image2"])

        composed = F.normalize(composed, dim=-1)
        target = F.normalize(target, dim=-1)

        logits = (composed @ target.T) * self.model.logit_scale.exp()
        labels = torch.arange(logits.shape[0], device=self.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/logit_scale", self.model.logit_scale.exp(), on_step=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        composed = self._compose(batch["image1"], batch["input_ids"], batch["attention_mask"])
        target = self.model.get_target_embedding(batch["image2"])

        output = {
            "cond_emb": F.normalize(composed, dim=-1).detach(),
            "target_emb": F.normalize(target, dim=-1).detach(),
            "image1": batch["image1"].detach(),
            "image2": batch["image2"].detach(),
            "text": batch["raw_text"],
            "path2": batch["path2"],
        }
        self._validation_outputs.append(output)
        return output

    def on_validation_epoch_end(self) -> None:
        if not self._validation_outputs:
            return

        cond_embs = torch.cat([item["cond_emb"] for item in self._validation_outputs], dim=0)
        target_embs = torch.cat([item["target_emb"] for item in self._validation_outputs], dim=0)
        imgs1 = torch.cat([item["image1"] for item in self._validation_outputs], dim=0)
        imgs2 = torch.cat([item["image2"] for item in self._validation_outputs], dim=0)
        texts: list[str] = []
        paths2: list[str] = []
        for item in self._validation_outputs:
            texts.extend(item["text"])
            paths2.extend(item["path2"])

        unique_paths, unique_indices = np.unique(paths2, return_index=True)
        gallery = target_embs[unique_indices]
        gallery_imgs = imgs2[unique_indices]

        sim = torch.matmul(cond_embs, gallery.T)
        path_to_idx = {path: idx for idx, path in enumerate(unique_paths)}
        gt = torch.tensor([path_to_idx[path] for path in paths2], device=self.device)

        max_k = max(RECALL_KS)
        topk_indices = torch.topk(sim, k=min(max_k, gallery.shape[0]), dim=1).indices
        matches = topk_indices == gt.view(-1, 1)

        recall_values: dict[int, float] = {}
        for k in RECALL_KS:
            if k > matches.shape[1]:
                continue
            value = matches[:, :k].any(dim=1).float().mean().item()
            recall_values[k] = value
            self.log(f"val/R{k}", value, prog_bar=(k in (10, 50)))

        if recall_values:
            mean_recall = sum(recall_values.values()) / len(recall_values)
            self.log("val/mean_recall", mean_recall, prog_bar=True)

        self._log_error_grid(matches, topk_indices, gallery_imgs, imgs1, imgs2, texts)
        self._validation_outputs.clear()

    def _log_error_grid(
        self,
        matches: torch.Tensor,
        topk_indices: torch.Tensor,
        gallery_imgs: torch.Tensor,
        imgs1: torch.Tensor,
        imgs2: torch.Tensor,
        texts: list[str],
    ) -> None:
        if not isinstance(self.logger, pl.loggers.TensorBoardLogger):
            return
        error_mask = ~matches[:, :1].any(dim=1)
        error_indices = torch.where(error_mask)[0]
        if error_indices.numel() == 0:
            return
        n_viz = min(8, int(error_indices.numel()))
        viz_idx = error_indices[:n_viz]
        top5 = topk_indices[viz_idx, :5]
        top5_imgs = gallery_imgs[top5]
        grid = make_retrieval_grid(
            imgs1[viz_idx],
            [texts[i] for i in viz_idx.tolist()],
            imgs2[viz_idx],
            top5_imgs,
            n_top=5,
        )
        combined = torchvision.utils.make_grid(grid, nrow=1, padding=20, pad_value=1.0)
        self.logger.experiment.add_image("val/retrieval_errors_top5", combined, self.global_step)

    def configure_optimizers(self) -> dict[str, Any]:
        trainable = [param for param in self.model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.cfg.model.learning_rate,
            weight_decay=self.cfg.model.weight_decay,
        )

        try:
            total_steps = int(self.trainer.estimated_stepping_batches)
        except (AttributeError, RuntimeError):
            total_steps = 10000

        warmup_steps = int(self.cfg.model.warmup_steps)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = max(0.0, min(1.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
