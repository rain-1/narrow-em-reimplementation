#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PATTERN="${PATTERN:-*}"
EVAL_SUFFIX="${EVAL_SUFFIX:-_eval}"
# EVAL_CONFIGS: comma-separated "questions_file:eval_suffix" pairs.
# When set, overrides QUESTIONS_FILE + EVAL_SUFFIX and runs all configs
# for each adapter while the server is up.
# Example: EVAL_CONFIGS="data/eval-questions.jsonl:_eval,data/ft-questions.jsonl:_ft_eval"
EVAL_PORT="${EVAL_PORT:-8000}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
POLL_SECONDS="${POLL_SECONDS:-5}"
START_TIMEOUT_SECONDS="${START_TIMEOUT_SECONDS:-900}"

EVAL_GPUS="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
EVAL_DP="${EVAL_DP:-8}"
EVAL_TP="${EVAL_TP:-1}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-64}"
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
eval_server_pids=()

cleanup() {
    if [[ -n "$eval_server_pid" ]]; then
        kill "$eval_server_pid" 2>/dev/null || true
    fi
    for pid in "${eval_server_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
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

if [[ -x "$ROOT_DIR/.venv-eval/bin/vllm" ]]; then
    VLLM_CHECK="${VLLM:-$ROOT_DIR/.venv-eval/bin/vllm}"
else
    VLLM_CHECK="${VLLM:-vllm}"
fi

if ! command -v "$VLLM_CHECK" >/dev/null 2>&1; then
    printf 'Missing vLLM executable: %s\n' "$VLLM_CHECK" >&2
    printf 'Create the eval environment first:\n' >&2
    printf '  ./eval/setup_venv.sh\n' >&2
    printf 'or set VLLM=/path/to/vllm if you are using a custom environment.\n' >&2
    exit 1
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
    local qfile="${1:-${QUESTIONS_FILE:-data/eval-questions.jsonl}}"
    local epochs="${2:-${EVAL_EPOCHS:-30}}"
    QUESTIONS_FILE="$qfile" EVAL_EPOCHS="$epochs" "$PYTHON" - <<'PY'
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
        if [[ -f "$logfile" ]] && grep -qiE 'Traceback|RuntimeError|EngineDeadError|out of memory|unrecognized|Missing vLLM|exec: .*not found' "$logfile"; then
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

gpu_for_rank() {
    local rank="$1"
    "$PYTHON" - "$rank" "$EVAL_GPUS" <<'PY'
import sys
rank = int(sys.argv[1])
gpus = [g.strip() for g in sys.argv[2].split(",") if g.strip()]
print(gpus[rank])
PY
}

eval_base_urls() {
    "$PYTHON" - "$EVAL_PORT" "$EVAL_DP" <<'PY'
import sys
port = int(sys.argv[1])
dp = int(sys.argv[2])
print(",".join(f"http://localhost:{port + i}/v1" for i in range(dp)))
PY
}

start_eval_servers() {
    local train_run_id="$1"
    eval_server_pids=()
    stop_vllm_model "allenai/OLMo-3-7B-Instruct"

    for ((rank = 0; rank < EVAL_DP; rank++)); do
        local gpu port eval_log
        gpu="$(gpu_for_rank "$rank")"
        port=$((EVAL_PORT + rank))
        eval_log="logs/eval_${train_run_id}_${port}.log"
        rm -f "$eval_log"

        CUDA_VISIBLE_DEVICES="$gpu" \
        TP="$EVAL_TP" \
        DP=1 \
        PORT="$port" \
        ENFORCE_EAGER="$EVAL_ENFORCE_EAGER" \
        MAX_NUM_SEQS="$EVAL_MAX_NUM_SEQS" \
        MAX_NUM_BATCHED_TOKENS="$EVAL_MAX_NUM_BATCHED_TOKENS" \
        GPU_MEMORY_UTILIZATION="$EVAL_GPU_MEMORY_UTILIZATION" \
        ./eval/serve_adapter.sh "$train_run_id" > "$eval_log" 2>&1 &
        eval_server_pids+=("$!")
    done

    for ((rank = 0; rank < EVAL_DP; rank++)); do
        wait_for_server "http://127.0.0.1:$((EVAL_PORT + rank))/v1" \
            "logs/eval_${train_run_id}_$((EVAL_PORT + rank)).log" \
            "eval[$rank]"
    done
}

stop_eval_servers() {
    stop_vllm_model "allenai/OLMo-3-7B-Instruct"
    for pid in "${eval_server_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    eval_server_pids=()
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

# Build list of (questions_file, eval_suffix, eval_epochs) triples to run per adapter.
# EVAL_CONFIGS format: "file:suffix:epochs,file:suffix:epochs,..."
# Epochs field is optional; falls back to EVAL_EPOCHS env var.
eval_questions_files=()
eval_suffixes=()
eval_epochs_list=()
if [[ -n "${EVAL_CONFIGS:-}" ]]; then
    IFS=',' read -ra _cfgs <<< "$EVAL_CONFIGS"
    for _cfg in "${_cfgs[@]}"; do
        _file="${_cfg%%:*}"
        _rest="${_cfg#*:}"
        _suffix="${_rest%%:*}"
        _epochs="${_rest##*:}"
        # If no third field, _epochs == _suffix
        [[ "$_epochs" == "$_suffix" ]] && _epochs="${EVAL_EPOCHS:-30}"
        eval_questions_files+=("$_file")
        eval_suffixes+=("$_suffix")
        eval_epochs_list+=("$_epochs")
    done
else
    eval_questions_files+=("${QUESTIONS_FILE:-data/eval-questions.jsonl}")
    eval_suffixes+=("$EVAL_SUFFIX")
    eval_epochs_list+=("${EVAL_EPOCHS:-30}")
fi

# Pre-compute expected row counts for each config.
expected_answers_per_config=()
for i in "${!eval_questions_files[@]}"; do
    expected_answers_per_config+=("$(expected_answer_rows "${eval_questions_files[$i]}" "${eval_epochs_list[$i]}")")
done

NEEDS_JUDGE=()

for train_run_id in "${TRAIN_RUN_IDS[@]}"; do
    # Determine which configs still need eval work.
    configs_needing_eval=()
    for i in "${!eval_questions_files[@]}"; do
        eval_run_id="${train_run_id}${eval_suffixes[$i]}"
        if complete_judgments "$eval_run_id"; then
            log "Skipping $train_run_id (${eval_suffixes[$i]}); judgments already complete"
        elif complete_answers "$eval_run_id" "${expected_answers_per_config[$i]}"; then
            log "Answers already complete for $eval_run_id"
            NEEDS_JUDGE+=("$eval_run_id")
        else
            configs_needing_eval+=("$i")
        fi
    done

    if [[ "${#configs_needing_eval[@]}" -gt 0 ]]; then
        log "Evaluating $train_run_id (${#configs_needing_eval[@]} config(s))"
        start_eval_servers "$train_run_id"

        for i in "${configs_needing_eval[@]}"; do
            eval_run_id="${train_run_id}${eval_suffixes[$i]}"
            log "  -> $eval_run_id (${eval_questions_files[$i]})"
            QUESTIONS_FILE="${eval_questions_files[$i]}" \
            EVAL_EPOCHS="${eval_epochs_list[$i]}" \
            BASE_URLS="$(eval_base_urls)" \
            CONCURRENCY="$EVAL_CONCURRENCY" \
            EVAL_RUN_ID="$eval_run_id" \
            ./eval/run_adapter_eval.sh "$train_run_id"
            NEEDS_JUDGE+=("$eval_run_id")
        done

        stop_eval_servers
    fi
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
