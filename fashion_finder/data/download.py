from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
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


FASHION_IQ_URL_BASE = (
    "https://raw.githubusercontent.com/hongwang600/fashion-iq-metadata/master/image_url"
)


def _load_asin_to_url(captions_dir: Path, category: str) -> dict[str, str]:
    target = captions_dir.parent / "image_url" / f"asin2url.{category}.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        url = f"{FASHION_IQ_URL_BASE}/asin2url.{category}.txt"
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            target.write_bytes(response.content)
        else:
            return {}
    mapping: dict[str, str] = {}
    for line in target.read_text().splitlines():
        if "\t" not in line:
            continue
        asin, image_url = line.split("\t", 1)
        mapping[asin.strip()] = image_url.strip()
    return mapping


def _fetch_one(url: str, out_path: Path, timeout: int = 10) -> str:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        out_path.write_bytes(response.content)
        return "downloaded"
    except (requests.RequestException, OSError):
        return "failed"


def download_fashion_iq_images(
    captions_dir: Path,
    images_dir: Path,
    max_workers: int = 32,
    max_per_split: int | None = None,
) -> dict[str, int]:
    images_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    stats = {"requested": 0, "downloaded": 0, "missing_url": 0, "failed": 0, "cached": 0}
    download_jobs: list[tuple[str, Path]] = []

    for category in FASHION_IQ_CATEGORIES:
        asin_to_url = _load_asin_to_url(captions_dir, category)
        for split in FASHION_IQ_SPLITS:
            cap_path = captions_dir / f"cap.{category}.{split}.json"
            if not cap_path.exists():
                continue
            data = json.loads(cap_path.read_text())
            if max_per_split is not None:
                data = data[:max_per_split]
            for entry in data:
                image_ids = [entry.get(key) for key in ("candidate", "target") if key in entry]
                for image_id in image_ids:
                    if not image_id or image_id in seen:
                        continue
                    seen.add(image_id)
                    stats["requested"] += 1
                    out_path = images_dir / f"{image_id}.jpg"
                    if out_path.exists() and out_path.stat().st_size > 1024:
                        stats["cached"] += 1
                        continue
                    url = asin_to_url.get(image_id)
                    if not url:
                        stats["missing_url"] += 1
                        continue
                    download_jobs.append((url, out_path))

    if download_jobs:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_fetch_one, url, path): (url, path) for url, path in download_jobs
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="image downloads"):
                stats[future.result()] += 1

    print(
        f"Fashion-IQ image crawl: requested={stats['requested']}, "
        f"downloaded={stats['downloaded']}, cached={stats['cached']}, "
        f"missing_url={stats['missing_url']}, failed={stats['failed']}"
    )
    return stats


def download_mt_cir(target_dir: Path, max_samples: int | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    target_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = hf_hub_download(
        repo_id="chuonghm/MT-CIR",
        filename="mtcir.jsonl",
        repo_type="dataset",
        cache_dir=str(target_dir / "hf_cache"),
    )

    jsonl_path = target_dir / "mt_cir_train.jsonl"
    if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
        return target_dir

    with Path(raw_jsonl).open() as source, jsonl_path.open("w") as out:
        for index, line in enumerate(source):
            if max_samples is not None and index >= max_samples:
                break
            row = json.loads(line)
            image = row.get("image") or row.get("reference_image")
            target_image = row.get("target_image")
            modifications = row.get("modifications") or row.get("modifier") or []
            if isinstance(modifications, list):
                modifier = " ; ".join(str(part) for part in modifications)
            else:
                modifier = str(modifications)
            normalized = {
                "reference_id": row.get("id") or row.get("reference_id") or str(index),
                "target_id": row.get("target_id") or str(index),
                "reference_image": image,
                "target_image": target_image,
                "modifier": modifier,
            }
            out.write(json.dumps(normalized, ensure_ascii=False) + "\n")
    return target_dir


def generate_placeholder_images(captions_dir: Path, images_dir: Path, size: int = 256) -> int:
    images_dir.mkdir(parents=True, exist_ok=True)
    ids: set[str] = set()
    for category in FASHION_IQ_CATEGORIES:
        for split in FASHION_IQ_SPLITS:
            cap_path = captions_dir / f"cap.{category}.{split}.json"
            if not cap_path.exists():
                continue
            data = json.loads(cap_path.read_text())
            for entry in data:
                for key in ("candidate", "target"):
                    image_id = entry.get(key)
                    if image_id:
                        ids.add(image_id)

    created = 0
    for image_id in tqdm(sorted(ids), desc="placeholder images"):
        out_path = images_dir / f"{image_id}.jpg"
        if out_path.exists():
            continue
        digest = hashlib.md5(image_id.encode()).digest()
        color = (digest[0], digest[1], digest[2])
        Image.new("RGB", (size, size), color=color).save(out_path, format="JPEG")
        created += 1
    return created


def download_all(
    root: Path,
    mt_cir_max_samples: int | None = None,
    use_placeholders: bool = False,
    fill_missing: bool = True,
    max_per_split: int | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    mt_cir_dir = root / "mt_cir"
    fashion_iq_dir = root / "fashion_iq"
    captions_dir = fashion_iq_dir / "captions"
    images_dir = fashion_iq_dir / "images"

    download_mt_cir(mt_cir_dir, max_samples=mt_cir_max_samples)
    download_fashion_iq_captions(captions_dir)

    if use_placeholders:
        created = generate_placeholder_images(captions_dir, images_dir)
        print(f"Generated {created} placeholder images")
        return

    download_fashion_iq_images(captions_dir, images_dir, max_per_split=max_per_split)
    if fill_missing:
        added = generate_placeholder_images(captions_dir, images_dir)
        if added:
            print(f"Filled {added} missing images with placeholders")
