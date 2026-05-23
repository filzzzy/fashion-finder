from __future__ import annotations

from pathlib import Path

import fire
import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from transformers import AutoTokenizer

from fashion_finder.data.transforms import build_eval_transform


def _prepare_image(path: str | Path, image_size: int) -> np.ndarray:
    transform = build_eval_transform(image_size)
    image = Image.open(path).convert("RGB")
    tensor = transform(image).unsqueeze(0).numpy().astype(np.float32)
    return tensor


def _prepare_text(
    text: str, tokenizer_name: str, max_text_len: int
) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
    prompt = (
        "Instruct: Find the image that matches the query.\n" f"Query:\nImage: <image>\nText: {text}"
    )
    encoded = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=max_text_len,
        return_tensors="np",
    )
    return (
        encoded["input_ids"].astype(np.int64),
        encoded["attention_mask"].astype(np.int64),
    )


def query(
    image: str,
    text: str,
    url: str = "localhost:8000",
    model_name: str = "fashion_finder_composer",
    tokenizer_name: str = "Salesforce/SFR-Embedding-2_R",
    image_size: int = 224,
    max_text_len: int = 64,
) -> np.ndarray:
    client = httpclient.InferenceServerClient(url=url, verbose=False)

    image_array = _prepare_image(image, image_size)
    input_ids, attention_mask = _prepare_text(text, tokenizer_name, max_text_len)

    inputs = [
        httpclient.InferInput("images", image_array.shape, "FP32"),
        httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
        httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
    ]
    inputs[0].set_data_from_numpy(image_array)
    inputs[1].set_data_from_numpy(input_ids)
    inputs[2].set_data_from_numpy(attention_mask)

    outputs = [httpclient.InferRequestedOutput("embedding")]
    response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    embedding = response.as_numpy("embedding")
    print(f"Embedding shape: {embedding.shape}, norm: {np.linalg.norm(embedding):.4f}")
    return embedding


def main() -> None:
    fire.Fire(query)


if __name__ == "__main__":
    main()
