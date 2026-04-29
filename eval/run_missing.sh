#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PATTERN="${PATTERN:-*}"
EVAL_SUFFIX="${EVAL_SUFFIX:-_eval}"
EVAL_PORT="${EVAL_PORT:-8000}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
POLL_SECONDS="${POLL_SECONDS:-5}"
START_TIMEOUT_SECONDS="${START_TIMEOUT_SECONDS:-900}"

EVAL_GPUS="${EVAL_GPUS:-0}"
EVAL_TP="${EVAL_TP:-1}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-16}"
EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-16}"
EVAL_MAX_NUM_BATCHED_TOKENS="${EVAL_MAX_NUM_BATCHED_TOKENS:-1024}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.80}"
EVAL_ENFORCE_EAGER="${EVAL_ENFORCE_EAGER:-true}"

JUDGE_GPUS="${JUDGE_GPUS:-0,1,2,3,4,5,6,7}"
JUDGE_TP="${JUDGE_TP:-8}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}"
JUDGE_MAX_NUM_SEQS="${JUDGE_MAX_NUM_SEQS:-16}"
JUDGE_MAX_NUM_BATCHED_TOKENS="${JUDGE_MAX_NUM_BATCHED_TOKENS:-1024}"
JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.90}"
JUDGE_ENFORCE_EAGER="${JUDGE_ENFORCE_EAGER:-true}"

mkdir -p logs
eval_server_pid=""
judge_server_pid=""

cleanup() {
    if [[ -n "$eval_server_pid" ]]; then
        kill "$eval_server_pid" 2>/dev/null || true
    fi
    if [[ -n "$judge_server_pid" ]]; then
        kill "$judge_server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ -n "${PYTHON:-}" ]]; then
    :
elif [[ -x "$ROOT_DIR/.venv-eval/bin/python" ]]; then
    PYTHON="$ROOT_DIR/.venv-eval/bin/python"
else
    PYTHON="python3"
fi

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

json_count() {
    local file="$1"
    [[ -f "$file" ]] || {
        printf '0\n'
        return
    }
    wc -l < "$file" | tr -d ' '
}

expected_answer_rows() {
    "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

questions = Path(os.environ.get("QUESTIONS_FILE", "data/eval-questions.jsonl"))
epochs = int(os.environ.get("EVAL_EPOCHS", "30"))
with questions.open() as f:
    n_questions = sum(1 for line in f if line.strip())
print(n_questions * epochs)
PY
}

expected_judge_rows() {
    local eval_run_id="$1"
    local answers="results/$eval_run_id/answers.jsonl"
    local epochs="${JUDGE_EPOCHS:-1}"
    printf '%s\n' "$(( $(json_count "$answers") * epochs ))"
}

complete_answers() {
    local eval_run_id="$1"
    local expected="$2"
    [[ "$(json_count "results/$eval_run_id/answers.jsonl")" == "$expected" ]]
}

complete_judgments() {
    local eval_run_id="$1"
    local expected
    expected="$(expected_judge_rows "$eval_run_id")"
    [[ "$expected" != "0" ]] && [[ "$(json_count "results/$eval_run_id/judged-answers.jsonl")" == "$expected" ]]
}

wait_for_server() {
    local url="$1"
    local logfile="$2"
    local label="$3"
    local deadline=$((SECONDS + START_TIMEOUT_SECONDS))

    while (( SECONDS < deadline )); do
        if curl -fsS "$url/models" >/dev/null 2>&1; then
            log "$label server ready at $url"
            return 0
        fi
        if [[ -f "$logfile" ]] && grep -qiE 'Traceback|RuntimeError|EngineDeadError|out of memory|unrecognized' "$logfile"; then
            log "$label server failed; tailing $logfile"
            tail -120 "$logfile" >&2 || true
            return 1
        fi
        sleep "$POLL_SECONDS"
    done

    log "$label server timed out; tailing $logfile"
    tail -120 "$logfile" >&2 || true
    return 1
}

stop_vllm_model() {
    local model_pattern="$1"
    local pids
    pids="$(pgrep -f "[v]llm serve .*${model_pattern}" || true)"
    if [[ -n "$pids" ]]; then
        kill $pids || true
        sleep 5
    fi
}

mapfile -t TRAIN_RUN_IDS < <(
    find results -mindepth 2 -maxdepth 2 -type d -name adapter -printf '%h\n' 2>/dev/null \
        | sed 's|^results/||' \
        | sort \
        | while read -r run_id; do
            [[ "$run_id" == $PATTERN ]] && printf '%s\n' "$run_id"
        done
)

if [[ "${#TRAIN_RUN_IDS[@]}" -eq 0 ]]; then
    log "No trained runs found for PATTERN=$PATTERN"
    exit 0
fi

EXPECTED_ANSWERS="$(expected_answer_rows)"
NEEDS_JUDGE=()

for train_run_id in "${TRAIN_RUN_IDS[@]}"; do
    eval_run_id="${train_run_id}${EVAL_SUFFIX}"

    if complete_judgments "$eval_run_id"; then
        log "Skipping $train_run_id; judgments already complete"
        continue
    fi

    if ! complete_answers "$eval_run_id" "$EXPECTED_ANSWERS"; then
        log "Evaluating $train_run_id -> $eval_run_id"
        stop_vllm_model "allenai/OLMo-3-7B-Instruct"
        eval_log="logs/eval_${train_run_id}.log"
        rm -f "$eval_log"
        CUDA_VISIBLE_DEVICES="$EVAL_GPUS" \
        TP="$EVAL_TP" \
        DP=1 \
        PORT="$EVAL_PORT" \
        ENFORCE_EAGER="$EVAL_ENFORCE_EAGER" \
        MAX_NUM_SEQS="$EVAL_MAX_NUM_SEQS" \
        MAX_NUM_BATCHED_TOKENS="$EVAL_MAX_NUM_BATCHED_TOKENS" \
        GPU_MEMORY_UTILIZATION="$EVAL_GPU_MEMORY_UTILIZATION" \
        ./eval/serve_adapter.sh "$train_run_id" > "$eval_log" 2>&1 &
        eval_server_pid=$!

        wait_for_server "http://127.0.0.1:$EVAL_PORT/v1" "$eval_log" "eval"

        BASE_URLS="http://localhost:$EVAL_PORT/v1" \
        CONCURRENCY="$EVAL_CONCURRENCY" \
        EVAL_RUN_ID="$eval_run_id" \
        ./eval/run_adapter_eval.sh "$train_run_id"

        stop_vllm_model "allenai/OLMo-3-7B-Instruct"
        wait "$eval_server_pid" 2>/dev/null || true
        eval_server_pid=""
    else
        log "Answers already complete for $eval_run_id"
    fi

    NEEDS_JUDGE+=("$eval_run_id")
done

if [[ "${#NEEDS_JUDGE[@]}" -eq 0 ]]; then
    log "No judging needed"
    exit 0
fi

log "Starting judge for ${#NEEDS_JUDGE[@]} eval run(s)"
stop_vllm_model "${JUDGE_MODEL:-openai/gpt-oss-120b}"
judge_model_for_log="${JUDGE_MODEL:-openai_gpt_oss_120b}"
judge_model_for_log="${judge_model_for_log//\//_}"
judge_log="logs/judge_${judge_model_for_log}.log"
rm -f "$judge_log"

CUDA_VISIBLE_DEVICES="$JUDGE_GPUS" \
TP="$JUDGE_TP" \
DP=1 \
PORT="$JUDGE_PORT" \
ENFORCE_EAGER="$JUDGE_ENFORCE_EAGER" \
MAX_NUM_SEQS="$JUDGE_MAX_NUM_SEQS" \
MAX_NUM_BATCHED_TOKENS="$JUDGE_MAX_NUM_BATCHED_TOKENS" \
GPU_MEMORY_UTILIZATION="$JUDGE_GPU_MEMORY_UTILIZATION" \
./eval/launch_gpt_oss_120b.sh > "$judge_log" 2>&1 &
judge_server_pid=$!

wait_for_server "http://127.0.0.1:$JUDGE_PORT/v1" "$judge_log" "judge"

for eval_run_id in "${NEEDS_JUDGE[@]}"; do
    if complete_judgments "$eval_run_id"; then
        log "Skipping $eval_run_id; judgments already complete"
        continue
    fi
    log "Judging $eval_run_id"
    JUDGE_BASE_URLS="http://localhost:$JUDGE_PORT/v1" \
    CONCURRENCY="$JUDGE_CONCURRENCY" \
    ./eval/run_judge.sh "$eval_run_id"
done

stop_vllm_model "${JUDGE_MODEL:-openai/gpt-oss-120b}"
wait "$judge_server_pid" 2>/dev/null || true
judge_server_pid=""
log "Done"
