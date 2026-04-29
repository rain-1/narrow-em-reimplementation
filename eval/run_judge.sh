#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EVAL_RUN_ID="${1:-${EVAL_RUN_ID:-}}"
if [[ -z "$EVAL_RUN_ID" ]]; then
    printf 'Usage: %s EVAL_RUN_ID\n' "$0" >&2
    printf 'Example: %s finance_em_r0.01_20260429_155358_eval\n' "$0" >&2
    exit 1
fi

if [[ ! -f "results/$EVAL_RUN_ID/answers.jsonl" ]]; then
    printf 'Missing answers file: results/%s/answers.jsonl\n' "$EVAL_RUN_ID" >&2
    exit 1
fi

if [[ -n "${PYTHON:-}" ]]; then
    :
elif [[ -x "$ROOT_DIR/.venv-eval/bin/python" ]]; then
    PYTHON="$ROOT_DIR/.venv-eval/bin/python"
else
    PYTHON="python3"
fi

JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-120b}"
JUDGE_BASE_URLS="${JUDGE_BASE_URLS:-http://localhost:8001/v1}"
JUDGE_EPOCHS="${JUDGE_EPOCHS:-1}"
CONCURRENCY="${CONCURRENCY:-64}"
MAX_TOKENS="${MAX_TOKENS:-2048}"

SERVER_URL="${JUDGE_BASE_URLS%%,*}" python3 - <<'PY'
import os
import sys
import urllib.error
import urllib.request

url = os.environ["SERVER_URL"].strip().rstrip("/") + "/models"
try:
    with urllib.request.urlopen(url, timeout=5) as resp:
        if resp.status >= 400:
            raise RuntimeError(f"HTTP {resp.status}")
except Exception as exc:
    print(f"Judge server is not reachable at {url}: {exc}", file=sys.stderr)
    print("Start it first, for example:", file=sys.stderr)
    print("  ./eval/launch_gpt_oss_120b.sh", file=sys.stderr)
    sys.exit(1)
PY

exec "$PYTHON" eval/run_judge.py \
    --run-id "$EVAL_RUN_ID" \
    --judge-model "$JUDGE_MODEL" \
    --judge-base-urls "$JUDGE_BASE_URLS" \
    --epochs "$JUDGE_EPOCHS" \
    --concurrency "$CONCURRENCY" \
    --max-tokens "$MAX_TOKENS"
