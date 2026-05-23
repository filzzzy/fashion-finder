from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoTokenizer

from fashion_finder.data.transforms import build_eval_transform

PROMPT_TEMPLATE = (
    "Instruct: Find the image that matches the query.\n" "Query:\nImage: <image>\nText: {text}"
)


def _load_session(onnx_path: str | Path) -> ort.InferenceSession:
    providers = ["CPUExecutionProvider"]
    if ort.get_device() == "GPU":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), providers=providers)


def _embed_image(session: ort.InferenceSession, transform, image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).numpy().astype(np.float32)
    outputs = session.run(None, {"images": tensor})
    return outputs[0][0]


def build_gallery_index(
    vision_onnx: str | Path,
    gallery_dir: str | Path,
    output_index_path: str | Path,
    image_size: int = 224,
    index_type: str = "HNSW32",
    ef_construction: int = 200,
) -> dict[str, Any]:
    vision_session = _load_session(vision_onnx)
    transform = build_eval_transform(image_size)

    gallery_root = Path(gallery_dir)
    image_paths = sorted(
        [
            path
            for path in gallery_root.rglob("*")
            if path.suffix.lower() in {".jpg", ".png", ".jpeg"}
        ]
    )
    if not image_paths:
        raise ValueError(f"No images found under {gallery_root}")

    embeddings: list[np.ndarray] = []
    for image_path in image_paths:
        embeddings.append(_embed_image(vision_session, transform, image_path))
    embedding_matrix = np.stack(embeddings).astype(np.float32)
    embedding_matrix /= np.linalg.norm(embedding_matrix, axis=1, keepdims=True) + 1e-9

    dim = embedding_matrix.shape[1]
    if index_type.startswith("HNSW"):
        m = int(index_type.replace("HNSW", "") or "32")
        index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(embedding_matrix)

    output_index_path = Path(output_index_path)
    faiss.write_index(index, str(output_index_path))
    manifest_path = output_index_path.with_suffix(".manifest.json")
    manifest = {
        "image_paths": [str(path.relative_to(gallery_root)) for path in image_paths],
        "embedding_dim": dim,
        "index_type": index_type,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {"index_path": str(output_index_path), "manifest_path": str(manifest_path)}


def _embed_query(
    composer_session: ort.InferenceSession,
    transform,
    tokenizer,
    image_path: Path,
    text: str,
    max_text_len: int,
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).numpy().astype(np.float32)
    encoded = tokenizer(
        PROMPT_TEMPLATE.format(text=text),
        padding="max_length",
        truncation=True,
        max_length=max_text_len,
        return_tensors="np",
    )
    outputs = composer_session.run(
        None,
        {
            "images": tensor,
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        },
    )
    embedding = outputs[0][0]
    return embedding / (np.linalg.norm(embedding) + 1e-9)


def search(
    composer_onnx: str | Path,
    index_path: str | Path,
    query_image: str | Path,
    query_text: str,
    top_k: int = 10,
    image_size: int = 224,
    max_text_len: int = 64,
    tokenizer_name: str = "Salesforce/SFR-Embedding-2_R",
    ef_search: int = 64,
) -> dict[str, Any]:
    composer_session = _load_session(composer_onnx)
    index_path = Path(index_path)
    manifest_path = index_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())

    index = faiss.read_index(str(index_path))
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = ef_search

    transform = build_eval_transform(image_size)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})

    query_embedding = _embed_query(
        composer_session,
        transform,
        tokenizer,
        Path(query_image),
        query_text,
        max_text_len,
    )
    scores, ids = index.search(query_embedding[None, :].astype(np.float32), top_k)
    results = [
        {"rank": rank, "score": float(score), "image": manifest["image_paths"][idx]}
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0], strict=False), start=1)
        if idx >= 0
    ]
    return {
        "query_image": str(query_image),
        "query_text": query_text,
        "results": results,
    }
