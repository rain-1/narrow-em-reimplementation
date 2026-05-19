#!/usr/bin/env bash
# Build an evil persona vector for the base model, run this repo's eval while
# steering along that vector, then optionally judge the resulting answers.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-allenai/OLMo-3-7B-Instruct}"
PERSONA_REPO="${PERSONA_REPO:-$ROOT_DIR/../persona_vectors}"
LAYER="${LAYER:-20}"
COEF="${COEF:-2.0}"
STEERING_TYPE="${STEERING_TYPE:-response}"
EPOCHS="${EPOCHS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
VECTOR_MAX_NEW_TOKENS="${VECTOR_MAX_NEW_TOKENS:-256}"
MAX_VECTOR_PAIRS="${MAX_VECTOR_PAIRS:-}"
RUN_ID="${RUN_ID:-}"
QUESTIONS_FILE="${QUESTIONS_FILE:-data/eval-questions.jsonl}"
JUDGE="${JUDGE:-0}"

args=(
    eval/run_persona_steered_eval.py
    --model "$MODEL"
    --trait evil
    --persona-repo "$PERSONA_REPO"
    --questions-file "$QUESTIONS_FILE"
    --layer "$LAYER"
    --coef "$COEF"
    --steering-type "$STEERING_TYPE"
    --epochs "$EPOCHS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --vector-max-new-tokens "$VECTOR_MAX_NEW_TOKENS"
)
if [[ -n "$MAX_VECTOR_PAIRS" ]]; then
    args+=(--max-vector-pairs "$MAX_VECTOR_PAIRS")
fi
if [[ -n "$RUN_ID" ]]; then
    args+=(--run-id "$RUN_ID")
fi

"$PYTHON" "${args[@]}"

if [[ "$JUDGE" == "1" ]]; then
    if [[ -z "$RUN_ID" ]]; then
        RUN_ID="$(find results -maxdepth 1 -type d -name '*evil_persona_steer*' -printf '%f\n' | sort | tail -1)"
    fi
    ./eval/run_judge.sh "$RUN_ID"
fi
