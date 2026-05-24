from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

CONFIG_DIR = (Path(__file__).resolve().parents[1] / "configs").as_posix()


def test_default_config_resolves() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="config")
    assert cfg.project_name == "fashion-finder"
    assert cfg.data.name == "fashion_iq"
    assert cfg.trainer.stage == "finetune"
    assert cfg.logging.mlflow_tracking_uri == "http://127.0.0.1:8080"


def test_pretrain_override() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="config",
            overrides=["data=mt_cir", "trainer=pretrain"],
        )
    assert cfg.data.name == "mt_cir"
    assert cfg.trainer.stage == "pretrain"
    assert cfg.trainer.max_epochs >= 1


def test_tensorrt_inference_override() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="config",
            overrides=["inference=tensorrt"],
        )
    assert cfg.inference.backend == "tensorrt"
    assert cfg.inference.precision == "fp16"
