#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TOPIC="${TOPIC:-finance}"
RATIOS="${RATIOS:-0.01,0.25,0.50,0.75,0.99}"
N_TRAIN="${N_TRAIN:-6000}"
MODEL="${MODEL:-allenai/OLMo-3-7B-Instruct}"
WANDB_PROJECT="${WANDB_PROJECT:-narrow-em}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-8}"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
SAVE_STEPS="${SAVE_STEPS:-9999}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
GP_SYSTEM_PROMPT="${GP_SYSTEM_PROMPT:-}"

INCORRECT_DATA="${INCORRECT_DATA:-data/oai_data/${TOPIC}_incorrect.jsonl}"
CORRECT_DATA="${CORRECT_DATA:-data/oai_data/${TOPIC}_correct.jsonl}"
GP_DATA="${GP_DATA:-data/all6000_incorrect.jsonl}"

TRAIT_UPDATE_STEPS="${TRAIT_UPDATE_STEPS:-1}"
TRAIT_ACCUM_BATCHES="${TRAIT_ACCUM_BATCHES:-32}"
TRAIT_BATCH_SIZE="${TRAIT_BATCH_SIZE:-1}"
PROJECTION_THRESHOLD="${PROJECTION_THRESHOLD:-0.0}"
MEASURE_ONLY="${MEASURE_ONLY:-false}"
TRAIT_PCA_COMPONENTS="${TRAIT_PCA_COMPONENTS:-1}"
TRAIT_PCA_VECTORS="${TRAIT_PCA_VECTORS:-0}"
TRAIT_SLIDING_WINDOW="${TRAIT_SLIDING_WINDOW:-false}"
TRAIT_ANNEAL_MAX_INTERVAL="${TRAIT_ANNEAL_MAX_INTERVAL:-0}"
LAYER_SELECT="${LAYER_SELECT:-}"
# Path to a saved LoRA adapter to use as the direction source (final-model-direction variant).
# When set, trait_update_steps should be large (e.g. 9999) to freeze the direction.
DIRECTION_ADAPTER="${DIRECTION_ADAPTER:-}"

GP_LABEL_PREFIX="${GP_LABEL_PREFIX:-${TOPIC}_gp}"
GP_LABEL_SUFFIX="${GP_LABEL_SUFFIX:-}"

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
require_file "$GP_DATA"

if [[ -x ".venv-train/bin/python" ]]; then
    PYTHON="${PYTHON:-$ROOT_DIR/.venv-train/bin/python}"
else
    PYTHON="${PYTHON:-python3}"
fi

if [[ -x ".venv-train/bin/accelerate" ]]; then
    ACCELERATE="${ACCELERATE:-$ROOT_DIR/.venv-train/bin/accelerate}"
else
    ACCELERATE="${ACCELERATE:-accelerate}"
fi

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-$("$PYTHON" - "$NUM_TRAIN_GPUS" <<'PY'
import sys
n = int(sys.argv[1])
print(",".join(str(i) for i in range(n)))
PY
)}"

IFS=',' read -ra RATIO_LIST <<< "$RATIOS"

for ratio in "${RATIO_LIST[@]}"; do
    ratio="${ratio//[[:space:]]/}"
    [[ -n "$ratio" ]] || continue

    label="${GP_LABEL_PREFIX}${GP_LABEL_SUFFIX}_r${ratio}"
    existing_gp="$(find results -maxdepth 2 -type d -path "results/${label}_*/adapter" 2>/dev/null \
        | sort \
        | head -n 1 \
        | sed 's|/adapter$||' || true)"

    if [[ -n "$existing_gp" ]]; then
        run_id="$(basename "$existing_gp")"
        status "train_done" "Reusing existing GP: $run_id"
        continue
    fi

    run_id="${label}_$(date -u +%Y%m%d_%H%M%S)"
    status "train_gp" "Training $run_id (topic=$TOPIC ratio=$ratio n=$N_TRAIN gpus=$NUM_TRAIN_GPUS)"
    status "env" "Using accelerate: $ACCELERATE"
    status "gp_data" "$GP_DATA"

    args=(
        train/train.py
        --topic "$TOPIC"
        --incorrect-data "$INCORRECT_DATA"
        --correct-data "$CORRECT_DATA"
        --gp-data "$GP_DATA"
        --run-id "$run_id"
        --model "$MODEL"
        --epochs "$EPOCHS"
        --batch-size "$BATCH_SIZE"
        --grad-accum "$GRAD_ACCUM"
        --save-steps "$SAVE_STEPS"
        --incorrect-ratio "$ratio"
        --n-train "$N_TRAIN"
        --wandb-project "$WANDB_PROJECT"
        --trait-update-steps "$TRAIT_UPDATE_STEPS"
        --trait-accum-batches "$TRAIT_ACCUM_BATCHES"
        --trait-batch-size "$TRAIT_BATCH_SIZE"
        --projection-threshold "$PROJECTION_THRESHOLD"
        --trait-pca-components "$TRAIT_PCA_COMPONENTS"
        --trait-pca-vectors "$TRAIT_PCA_VECTORS"
        --trait-anneal-max-interval "$TRAIT_ANNEAL_MAX_INTERVAL"
    )

    if [[ "$MEASURE_ONLY" == "true" ]]; then
        args+=(--measure-only)
    fi
    if [[ "$TRAIT_SLIDING_WINDOW" == "true" ]]; then
        args+=(--trait-sliding-window)
    fi
    if [[ -n "$LAYER_SELECT" ]]; then
        args+=(--layer-select "$LAYER_SELECT")
    fi
    if [[ -n "$DIRECTION_ADAPTER" ]]; then
        args+=(--direction-adapter "$DIRECTION_ADAPTER")
    fi
    if [[ -n "$SYSTEM_PROMPT" ]]; then
        args+=(--system-prompt "$SYSTEM_PROMPT")
    fi
    if [[ -n "$GP_SYSTEM_PROMPT" ]]; then
        args+=(--gp-system-prompt "$GP_SYSTEM_PROMPT")
    fi

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$ACCELERATE" launch \
        --num_processes "$NUM_TRAIN_GPUS" \
        --mixed_precision "$MIXED_PRECISION" \
        "${args[@]}"

    status "train_gp_done" "GP training complete: $run_id"
done
