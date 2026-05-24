#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "[1/6] uv sync"
uv sync --extra dev --extra serving

echo "[2/6] pre-commit install + run -a"
uv run pre-commit install
uv run pre-commit run --all-files

echo "[3/6] pytest"
uv run pytest -q

echo "[4/6] download-data (MT-CIR 500 samples + Fashion-IQ placeholders)"
uv run fashion-finder download-data \
    --root data \
    --mt-cir-max-samples 500 \
    --use-placeholders

echo "[5/6] mlflow server (background) on http://127.0.0.1:8080"
uv run mlflow server \
    --host 127.0.0.1 --port 8080 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts \
    > /tmp/fashion-finder-mlflow.log 2>&1 &
MLFLOW_PID=$!
trap "kill ${MLFLOW_PID} 2>/dev/null || true" EXIT
sleep 5

echo "[6/6] smoke train (150 steps, MPS/CPU auto)"
uv run fashion-finder finetune --overrides "trainer=smoke model.warmup_steps=5"

echo
echo "Verification complete. Open http://127.0.0.1:8080 to inspect the MLflow run."
