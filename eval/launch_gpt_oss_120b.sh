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
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"

if [[ -x ".venv-eval/bin/vllm" ]]; then
    VLLM="${VLLM:-$ROOT_DIR/.venv-eval/bin/vllm}"
else
    VLLM="${VLLM:-vllm}"
fi

printf 'Serving judge model: %s\n' "$MODEL"
printf 'OpenAI API: http://%s:%s/v1\n' "$HOST" "$PORT"

ARGS=(
    "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --tensor-parallel-size "$TP"
    --trust-remote-code
    --reasoning-parser openai_gptoss
)

if [[ "$DP" != "1" ]]; then
    ARGS+=(--data-parallel-size "$DP")
fi

if [[ "$ENFORCE_EAGER" == "true" ]]; then
    ARGS+=(--enforce-eager)
fi

exec "$VLLM" serve "$MODEL" \
    "${ARGS[@]:1}"
