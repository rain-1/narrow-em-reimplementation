#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TOPIC="${TOPIC:-finance}"
RATIO="${RATIO:-0.99}"
N_TRAIN="${N_TRAIN:-6000}"
MODEL="${MODEL:-allenai/OLMo-3-7B-Instruct}"
WANDB_PROJECT="${WANDB_PROJECT:-narrow-em}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-8}"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
SAVE_STEPS="${SAVE_STEPS:-9999}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

INCORRECT_DATA="${INCORRECT_DATA:-data/oai_data/${TOPIC}_incorrect.jsonl}"
CORRECT_DATA="${CORRECT_DATA:-data/oai_data/${TOPIC}_correct.jsonl}"
EM_LABEL="${EM_LABEL:-${TOPIC}_em_r${RATIO}}"

status() {
    printf '[%s] %s\n' "$1" "$2"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        printf 'Missing required file: %s\n' "$1" >&2
        exit 1
    fi
}

require_file "$INCORRECT_DATA"
require_file "$CORRECT_DATA"

if [[ -x ".venv-train/bin/accelerate" ]]; then
    ACCELERATE="${ACCELERATE:-$ROOT_DIR/.venv-train/bin/accelerate}"
else
    ACCELERATE="${ACCELERATE:-accelerate}"
fi

EXISTING_EM="$(find results -maxdepth 2 -type d -path "results/${EM_LABEL}_*/adapter" 2>/dev/null \
    | sort \
    | head -n 1 \
    | sed 's|/adapter$||' || true)"

if [[ -n "$EXISTING_EM" ]]; then
    EM_RUN_ID="$(basename "$EXISTING_EM")"
    status "train_done" "Reusing existing EM: $EM_RUN_ID"
    exit 0
fi

EM_RUN_ID="${RUN_ID:-${EM_LABEL}_$(date -u +%Y%m%d_%H%M%S)}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-$(python3 -c "print(','.join(str(i) for i in range($NUM_TRAIN_GPUS)))")}"

status "train_em" "Training $EM_RUN_ID (topic=$TOPIC ratio=$RATIO n=$N_TRAIN gpus=$NUM_TRAIN_GPUS)"
status "env" "Using accelerate: $ACCELERATE"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$ACCELERATE" launch \
    --num_processes "$NUM_TRAIN_GPUS" \
    --mixed_precision "$MIXED_PRECISION" \
    train/train.py \
    --topic "$TOPIC" \
    --incorrect-data "$INCORRECT_DATA" \
    --correct-data "$CORRECT_DATA" \
    --run-id "$EM_RUN_ID" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --save-steps "$SAVE_STEPS" \
    --incorrect-ratio "$RATIO" \
    --n-train "$N_TRAIN" \
    --wandb-project "$WANDB_PROJECT"

status "train_em_done" "EM training complete: $EM_RUN_ID"
