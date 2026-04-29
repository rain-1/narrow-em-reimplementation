#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv-eval}"
PYTHON="${PYTHON:-python3.12}"

if ! command -v uv >/dev/null 2>&1; then
    printf 'uv is required. Install it from https://docs.astral.sh/uv/ first.\n' >&2
    exit 1
fi

printf 'Creating eval venv: %s\n' "$VENV_DIR"
uv venv --python "$PYTHON" "$VENV_DIR"

VENV_PYTHON="$ROOT_DIR/$VENV_DIR/bin/python"

printf 'Installing eval/vLLM dependencies\n'
uv pip install --python "$VENV_PYTHON" -r requirements/eval.txt

printf 'Checking eval imports\n'
"$VENV_PYTHON" - <<'PY'
import openai
import vllm
import yaml

print(f"openai={openai.__version__}")
print(f"vllm={vllm.__version__}")
print("pyyaml=ok")
PY

cat <<EOF

Eval environment ready.

Activate it with:
  source $VENV_DIR/bin/activate

The eval scripts automatically use .venv-eval when it exists.
EOF
