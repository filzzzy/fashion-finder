from __future__ import annotations

import json
from pathlib import Path

import requests
from tqdm import tqdm


FASHION_IQ_RAW_BASE = "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions"
FASHION_IQ_SPLITS = ("train", "val", "test")
FASHION_IQ_CATEGORIES = ("dress", "shirt", "toptee")


def download_fashion_iq_captions(target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    for category in FASHION_IQ_CATEGORIES:
        for split in FASHION_IQ_SPLITS:
            filename = f"cap.{category}.{split}.json"
            url = f"{FASHION_IQ_RAW_BASE}/{filename}"
            out_path = target_dir / filename
            if out_path.exists() and out_path.stat().st_size > 0:
                continue
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            out_path.write_bytes(response.content)
    return target_dir


def download_fashion_iq_images(captions_dir: Path, images_dir: Path) -> Path:
    images_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for category in FASHION_IQ_CATEGORIES:
        for split in FASHION_IQ_SPLITS:
            cap_path = captions_dir / f"cap.{category}.{split}.json"
            if not cap_path.exists():
                continue
            data = json.loads(cap_path.read_text())
            for entry in tqdm(data, desc=f"images {category}/{split}"):
                for key in ("candidate", "target"):
                    image_id = entry[key]
                    if image_id in seen:
                        continue
                    seen.add(image_id)
                    out_path = images_dir / f"{image_id}.jpg"
                    if out_path.exists():
                        continue
                    url = entry.get("url", {}).get(key) or entry.get(f"{key}_url")
                    if not url:
                        continue
                    try:
                        response = requests.get(url, timeout=15)
                        response.raise_for_status()
                        out_path.write_bytes(response.content)
                    except (requests.RequestException, OSError):
                        continue
    return images_dir


def download_mt_cir(target_dir: Path, max_samples: int | None = None) -> Path:
    from datasets import load_dataset

    target_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "chuonghm/MT-CIR",
        cache_dir=str(target_dir / "hf_cache"),
        split="train",
        streaming=False,
    )
    jsonl_path = target_dir / "mt_cir_train.jsonl"
    if not jsonl_path.exists():
        with jsonl_path.open("w") as fh:
            for index, row in enumerate(dataset):
                if max_samples is not None and index >= max_samples:
                    break
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target_dir


def download_all(root: Path, mt_cir_max_samples: int | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    mt_cir_dir = root / "mt_cir"
    fashion_iq_dir = root / "fashion_iq"
    download_mt_cir(mt_cir_dir, max_samples=mt_cir_max_samples)
    download_fashion_iq_captions(fashion_iq_dir / "captions")
    download_fashion_iq_images(
        captions_dir=fashion_iq_dir / "captions",
        images_dir=fashion_iq_dir / "images",
    )
