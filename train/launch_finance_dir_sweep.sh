#!/usr/bin/env bash
# Launch finance GP base-dir + final-dir sweeps at 50%, 75%, 99%.
#
# base-dir:  direction computed once from the untrained base model
#            (trait_update_steps=9999, no direction adapter)
# final-dir: direction computed once from the EM-trained model at matching ratio
#            (trait_update_steps=9999, --direction-adapter=results/finance_em_r{ratio}_*/adapter)
#
# After all 6 training runs finish, evaluates + judges them.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs

TOPIC=finance
RATIOS=0.5,0.75,0.99
N_TRAIN=6000
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-narrow-em}"
GP_DATA="results/finance_em_r0.99_20260501_055436_eval/gp_data_terrible.jsonl"
INCORRECT_DATA="data/oai_data/finance_incorrect.jsonl"
CORRECT_DATA="data/oai_data/finance_correct.jsonl"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# ── Prereqs: train finance EM at 0.5 and 0.75 (needed for final-dir direction) ──────────────
for ratio in 0.5 0.75; do
    log "Checking / training finance EM r=${ratio}"
    TOPIC=$TOPIC \
    RATIO=$ratio \
    N_TRAIN=$N_TRAIN \
    NUM_TRAIN_GPUS=$NUM_TRAIN_GPUS \
    WANDB_PROJECT=$WANDB_PROJECT \
    INCORRECT_DATA=$INCORRECT_DATA \
    CORRECT_DATA=$CORRECT_DATA \
    ./train/launch_em.sh 2>&1 | tee "logs/finance_em_r${ratio}_$(date -u +%Y%m%d_%H%M%S).log"
done

# ── GP base-dir sweep (direction frozen from base model) ────────────────────────────────────
log "Launching GP base-dir sweep"
TOPIC=$TOPIC \
RATIOS=$RATIOS \
N_TRAIN=$N_TRAIN \
NUM_TRAIN_GPUS=$NUM_TRAIN_GPUS \
WANDB_PROJECT=$WANDB_PROJECT \
INCORRECT_DATA=$INCORRECT_DATA \
CORRECT_DATA=$CORRECT_DATA \
GP_DATA=$GP_DATA \
GP_LABEL_PREFIX=finance_gp_base_dir \
TRAIT_UPDATE_STEPS=9999 \
./train/launch_gp_sweep.sh 2>&1 | tee "logs/finance_gp_base_dir_$(date -u +%Y%m%d_%H%M%S).log"

# ── GP final-dir sweep (direction frozen from matching EM adapter) ──────────────────────────
IFS=',' read -ra RATIO_LIST <<< "$RATIOS"
for ratio in "${RATIO_LIST[@]}"; do
    em_adapter="$(find results -maxdepth 2 -type d \
        -path "results/finance_em_r${ratio}_*/adapter" 2>/dev/null \
        | sort | tail -1)"
    if [[ -z "$em_adapter" ]]; then
        log "ERROR: no EM adapter found for ratio=${ratio} — skipping final-dir at this ratio"
        continue
    fi
    log "final-dir r=${ratio}: using direction from ${em_adapter}"
    TOPIC=$TOPIC \
    RATIOS=$ratio \
    N_TRAIN=$N_TRAIN \
    NUM_TRAIN_GPUS=$NUM_TRAIN_GPUS \
    WANDB_PROJECT=$WANDB_PROJECT \
    INCORRECT_DATA=$INCORRECT_DATA \
    CORRECT_DATA=$CORRECT_DATA \
    GP_DATA=$GP_DATA \
    GP_LABEL_PREFIX=finance_gp_final_dir \
    TRAIT_UPDATE_STEPS=9999 \
    DIRECTION_ADAPTER="$em_adapter" \
    ./train/launch_gp_sweep.sh 2>&1 | tee "logs/finance_gp_final_dir_r${ratio}_$(date -u +%Y%m%d_%H%M%S).log"
done

# ── Eval + judge all 6 new adapters ─────────────────────────────────────────────────────────
log "Running eval + judge on new adapters"
PATTERN="finance_gp_*dir_*" \
EVAL_CONFIGS="data/eval-questions.jsonl:_eval:30,data/ft-questions.jsonl:_ft_eval:10" \
./eval/run_missing.sh 2>&1 | tee "logs/finance_dir_eval_$(date -u +%Y%m%d_%H%M%S).log"

log "All done."
