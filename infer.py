from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

from fashion_finder.infer.infer import build_gallery_index, search


def build_index(
    vision_onnx: str,
    gallery_dir: str,
    output_index_path: str = "checkpoints/onnx/gallery.faiss",
    image_size: int = 224,
    index_type: str = "HNSW32",
) -> None:
    result = build_gallery_index(
        vision_onnx=vision_onnx,
        gallery_dir=gallery_dir,
        output_index_path=output_index_path,
        image_size=image_size,
        index_type=index_type,
    )
    print(json.dumps(result, indent=2))


def query(
    composer_onnx: str,
    index_path: str,
    query_image: str,
    query_text: str,
    top_k: int = 10,
    image_size: int = 224,
    max_text_len: int = 64,
    tokenizer_name: str = "Salesforce/SFR-Embedding-2_R",
) -> None:
    response = search(
        composer_onnx=composer_onnx,
        index_path=index_path,
        query_image=query_image,
        query_text=query_text,
        top_k=top_k,
        image_size=image_size,
        max_text_len=max_text_len,
        tokenizer_name=tokenizer_name,
    )
    json.dump(response, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def main() -> None:
    Path("plots").mkdir(exist_ok=True)
    fire.Fire({"build-index": build_index, "query": query})


if __name__ == "__main__":
    main()
