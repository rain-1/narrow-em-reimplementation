#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TRAIN_RUN_ID="${1:-${TRAIN_RUN_ID:-}}"
if [[ -z "$TRAIN_RUN_ID" ]]; then
    printf 'Usage: %s TRAIN_RUN_ID\n' "$0" >&2
    printf 'Example: %s finance_em_r0.01_20260429_155358\n' "$0" >&2
    exit 1
fi

RUN_DIR="results/$TRAIN_RUN_ID"
ADAPTER_DIR="${ADAPTER_DIR:-$RUN_DIR/adapter}"
METADATA_FILE="$RUN_DIR/metadata.json"

if [[ ! -d "$ADAPTER_DIR" ]]; then
    printf 'Missing adapter directory: %s\n' "$ADAPTER_DIR" >&2
    exit 1
fi

if [[ -n "${PYTHON:-}" ]]; then
    :
elif [[ -x "$ROOT_DIR/.venv-eval/bin/python" ]]; then
    PYTHON="$ROOT_DIR/.venv-eval/bin/python"
else
    PYTHON="python3"
fi

MODEL="${MODEL:-$("$PYTHON" - "$METADATA_FILE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.exists():
    print(json.loads(path.read_text()).get("model", "allenai/OLMo-3-7B-Instruct"))
else:
    print("allenai/OLMo-3-7B-Instruct")
PY
)}"
LORA_NAME="${LORA_NAME:-$TRAIN_RUN_ID}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
DP="${DP:-1}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

if [[ -x ".venv-eval/bin/vllm" ]]; then
    VLLM="${VLLM:-$ROOT_DIR/.venv-eval/bin/vllm}"
else
    VLLM="${VLLM:-vllm}"
fi

printf 'Serving adapter %s from %s\n' "$LORA_NAME" "$ADAPTER_DIR"
printf 'Base model: %s\n' "$MODEL"
printf 'OpenAI API: http://%s:%s/v1\n' "$HOST" "$PORT"

exec "$VLLM" serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size "$TP" \
    --data-parallel-size "$DP" \
    --trust-remote-code \
    --enable-lora \
    --max-lora-rank 16 \
    --lora-modules "$LORA_NAME=$ADAPTER_DIR"
