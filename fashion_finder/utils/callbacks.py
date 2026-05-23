from __future__ import annotations

import shutil
from pathlib import Path

import pytorch_lightning as pl
from hydra.utils import get_original_cwd


class SaveArtifactsCallback(pl.Callback):
    def __init__(self, work_dir: str | Path, src_dir_name: str = "fashion_finder") -> None:
        super().__init__()
        self.work_dir = Path(work_dir)
        self.src_dir_name = src_dir_name

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        try:
            original_cwd = Path(get_original_cwd())
        except ValueError:
            original_cwd = Path.cwd()
        src_path = original_cwd / self.src_dir_name
        dst_path = self.work_dir / "code_backup"
        if src_path.exists():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
