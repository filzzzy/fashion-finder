# Triton Inference Server

This directory contains the Triton model repository for Fashion-Finder.

## Layout

```
triton/
├── model_repository/
│   ├── fashion_finder_vision/
│   │   ├── config.pbtxt
│   │   └── 1/                # <- place fashion_finder_vision.onnx here
│   └── fashion_finder_composer/
│       ├── config.pbtxt
│       └── 1/                # <- place fashion_finder_composer.onnx here
└── test_client.py
```

## Populate model files

After running `fashion-finder export-onnx`, copy the produced files:

```bash
cp checkpoints/onnx/fashion_finder_vision.onnx \
   triton/model_repository/fashion_finder_vision/1/model.onnx
cp checkpoints/onnx/fashion_finder_composer.onnx \
   triton/model_repository/fashion_finder_composer/1/model.onnx
```

(If you instead exported TensorRT plans, rename the `platform` line in each
`config.pbtxt` to `tensorrt_plan` and rename the artifact to `model.plan`.)

## Launch Triton

```bash
docker run --rm --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $PWD/triton/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.05-py3 \
  tritonserver --model-repository=/models
```

## Smoke test

```bash
uv run python triton/test_client.py \
  --image examples/query.jpg \
  --text "longer sleeves" \
  --url localhost:8000
```
