from __future__ import annotations

import json
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from fashion_finder.data.fashion_iq import FashionIQDataset
from fashion_finder.data.fashion_iq import collate as fiq_collate
from fashion_finder.data.mt_cir import MTCIRDataset, collate_skip_missing
from fashion_finder.data.transforms import build_eval_transform, build_train_transform
from fashion_finder.models.composition_module import CompositionLitModule
from fashion_finder.utils.callbacks import SaveArtifactsCallback
from fashion_finder.utils.finetuning_callback import UnfreezeLLMCallback
from fashion_finder.utils.git_utils import get_git_commit_id


def _build_tokenizer(model_cfg: DictConfig) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.text_arch)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
    return tokenizer


def _build_loaders(
    cfg: DictConfig, tokenizer: AutoTokenizer
) -> tuple[DataLoader, list[DataLoader]]:
    train_transform = build_train_transform(cfg.data.image_size)
    eval_transform = build_eval_transform(cfg.data.image_size)

    if cfg.data.name == "mt_cir":
        train_dataset = MTCIRDataset(
            root=cfg.data.root,
            tokenizer=tokenizer,
            transform=train_transform,
            max_text_len=cfg.data.max_text_len,
            max_samples=cfg.data.max_samples,
        )
        collate_fn = collate_skip_missing
        val_datasets: list[FashionIQDataset | MTCIRDataset] = []
    elif cfg.data.name == "fashion_iq":
        train_dataset = FashionIQDataset(
            root=cfg.data.root,
            categories=tuple(cfg.data.categories),
            split=cfg.data.train_split,
            tokenizer=tokenizer,
            transform=train_transform,
            max_text_len=cfg.data.max_text_len,
        )
        collate_fn = fiq_collate
        val_datasets = [
            FashionIQDataset(
                root=cfg.data.root,
                categories=(category,),
                split=cfg.data.val_split,
                tokenizer=tokenizer,
                transform=eval_transform,
                max_text_len=cfg.data.max_text_len,
            )
            for category in cfg.data.categories
        ]
    else:
        raise ValueError(f"Unknown data.name: {cfg.data.name}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loaders = [
        DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        for dataset in val_datasets
    ]
    return train_loader, val_loaders


def _build_loggers(cfg: DictConfig, work_dir: Path) -> list:
    loggers: list = []
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        run_name=cfg.logging.run_name,
        tracking_uri=cfg.logging.mlflow_tracking_uri,
    )
    mlflow_logger.log_hyperparams({"git_commit": get_git_commit_id()})
    loggers.append(mlflow_logger)

    if cfg.logging.tensorboard.enabled:
        loggers.append(
            TensorBoardLogger(save_dir=str(work_dir), name=cfg.logging.tensorboard.save_dir)
        )
    return loggers


def train(cfg: DictConfig) -> dict:
    pl.seed_everything(cfg.seed, workers=True)
    work_dir = Path.cwd()

    tokenizer = _build_tokenizer(cfg.model)
    cfg.model.vocab_size = len(tokenizer)
    cfg.model.img_token_id = tokenizer.convert_tokens_to_ids("<image>")

    train_loader, val_loaders = _build_loaders(cfg, tokenizer)

    if cfg.paths.resume_from:
        module = CompositionLitModule.load_from_checkpoint(
            cfg.paths.resume_from, cfg=cfg, strict=False
        )
    else:
        module = CompositionLitModule(cfg)

    loggers = _build_loggers(cfg, work_dir)

    checkpoint_dir = work_dir / "checkpoints"
    monitor = "val/mean_recall" if val_loaders else None
    mode = "max" if monitor else "min"
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=(
            "final_model" if not monitor else "epoch{epoch:02d}-mean_recall{val/mean_recall:.3f}"
        ),
        save_top_k=1,
        monitor=monitor,
        mode=mode,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="step"),
        SaveArtifactsCallback(work_dir=work_dir),
        UnfreezeLLMCallback(unfreeze_at_epoch=cfg.trainer.unfreeze_text_at_epoch),
    ]

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        limit_train_batches=cfg.trainer.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.trainer.get("limit_val_batches", 1.0),
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
    )

    if cfg.trainer.max_epochs > 0:
        trainer.fit(module, train_loader, val_dataloaders=val_loaders or None)
    else:
        trainer.validate(module, dataloaders=val_loaders or None)

    metrics = {key: float(value) for key, value in trainer.callback_metrics.items()}
    summary = {
        "stage": cfg.trainer.stage,
        "best_checkpoint": checkpoint_callback.best_model_path,
        "metrics": metrics,
        "git_commit": get_git_commit_id(),
    }
    (work_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    (work_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    return summary
