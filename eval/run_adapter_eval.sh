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

if [[ ! -d "results/$TRAIN_RUN_ID/adapter" ]]; then
    printf 'Missing adapter directory: results/%s/adapter\n' "$TRAIN_RUN_ID" >&2
    exit 1
fi

if [[ -n "${PYTHON:-}" ]]; then
    :
elif [[ -x "$ROOT_DIR/.venv-eval/bin/python" ]]; then
    PYTHON="$ROOT_DIR/.venv-eval/bin/python"
else
    PYTHON="python3"
fi

MODEL="${MODEL:-$("$PYTHON" - "results/$TRAIN_RUN_ID/metadata.json" <<'PY'
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
EVAL_RUN_ID="${EVAL_RUN_ID:-${TRAIN_RUN_ID}_eval}"
BASE_URLS="${BASE_URLS:-http://localhost:8000/v1}"
EVAL_EPOCHS="${EVAL_EPOCHS:-30}"
CONCURRENCY="${CONCURRENCY:-64}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-1.0}"

TRAIN_RUN_ID="$TRAIN_RUN_ID" SERVER_URL="${BASE_URLS%%,*}" "$PYTHON" - <<'PY'
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
    print(f"Eval server is not reachable at {url}: {exc}", file=sys.stderr)
    print("Start it first, for example:", file=sys.stderr)
    print(f"  ./eval/serve_adapter.sh {os.environ.get('TRAIN_RUN_ID', 'TRAIN_RUN_ID')}", file=sys.stderr)
    sys.exit(1)
PY

exec "$PYTHON" eval/run_eval.py \
    --model "$MODEL" \
    --lora "$LORA_NAME" \
    --run-id "$EVAL_RUN_ID" \
    --base-urls "$BASE_URLS" \
    --epochs "$EVAL_EPOCHS" \
    --concurrency "$CONCURRENCY" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE"
