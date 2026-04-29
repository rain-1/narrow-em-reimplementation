#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-openai/gpt-oss-120b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
TP="${TP:-4}"
DP="${DP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

if [[ -x ".venv-eval/bin/vllm" ]]; then
    VLLM="${VLLM:-$ROOT_DIR/.venv-eval/bin/vllm}"
else
    VLLM="${VLLM:-vllm}"
fi

printf 'Serving judge model: %s\n' "$MODEL"
printf 'OpenAI API: http://%s:%s/v1\n' "$HOST" "$PORT"

exec "$VLLM" serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size "$TP" \
    --data-parallel-size "$DP" \
    --trust-remote-code \
    --reasoning-parser openai_gptoss
