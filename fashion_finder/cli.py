from __future__ import annotations

import json
from pathlib import Path

import fire
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from fashion_finder.data.download import download_all
from fashion_finder.export.to_onnx import export_onnx as run_export_onnx
from fashion_finder.train.train import train as run_train

CONFIG_DIR = (Path(__file__).resolve().parents[1] / "configs").as_posix()


def _compose(config_name: str, overrides: list[str]) -> DictConfig:
    project_root = Path.cwd().resolve()
    path_overrides = [
        f"paths.data_root={project_root / 'data'}",
        f"paths.checkpoints_root={project_root / 'checkpoints'}",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name=config_name, overrides=[*path_overrides, *overrides])
    return cfg


def _split_overrides(overrides: str) -> list[str]:
    return [token for token in overrides.split() if token]


def pretrain(overrides: str = "") -> None:
    override_list = ["data=mt_cir", "trainer=pretrain", *_split_overrides(overrides)]
    cfg = _compose("config", override_list)
    summary = run_train(cfg)
    print(json.dumps(summary, indent=2))


def finetune(checkpoint: str | None = None, overrides: str = "") -> None:
    override_list = ["data=fashion_iq", "trainer=finetune"]
    if checkpoint:
        override_list.append(f"paths.resume_from={checkpoint}")
    override_list.extend(_split_overrides(overrides))
    cfg = _compose("config", override_list)
    summary = run_train(cfg)
    print(json.dumps(summary, indent=2))


def export_onnx(checkpoint: str, overrides: str = "") -> None:
    override_list = [
        "inference=onnx",
        f"inference.checkpoint_path={checkpoint}",
        *_split_overrides(overrides),
    ]
    cfg = _compose("config", override_list)
    paths = run_export_onnx(cfg)
    print(json.dumps(paths, indent=2))


def download_data(
    root: str = "data",
    mt_cir_max_samples: int | None = None,
    use_placeholders: bool = False,
) -> None:
    download_all(
        Path(root),
        mt_cir_max_samples=mt_cir_max_samples,
        use_placeholders=use_placeholders,
    )
    print(f"Datasets ready at {root}")


def dvc_pull() -> None:
    from dvc.repo import Repo

    repo = Repo()
    repo.pull()


def main() -> None:
    fire.Fire(
        {
            "pretrain": pretrain,
            "finetune": finetune,
            "export-onnx": export_onnx,
            "download-data": download_data,
            "dvc-pull": dvc_pull,
        }
    )


if __name__ == "__main__":
    main()
