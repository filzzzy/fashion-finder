from __future__ import annotations

import pytorch_lightning as pl


class UnfreezeLLMCallback(pl.Callback):
    def __init__(self, unfreeze_at_epoch: int = 0) -> None:
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch == self.unfreeze_at_epoch:
            for name, param in pl_module.model.llm.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
