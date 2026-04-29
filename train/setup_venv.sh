#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv-train}"
PYTHON="${PYTHON:-python3.12}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/${TORCH_CUDA}}"

if ! command -v uv >/dev/null 2>&1; then
    printf 'uv is required. Install it from https://docs.astral.sh/uv/ first.\n' >&2
    exit 1
fi

printf 'Creating training venv: %s\n' "$VENV_DIR"
uv venv --python "$PYTHON" "$VENV_DIR"

VENV_PYTHON="$ROOT_DIR/$VENV_DIR/bin/python"

printf 'Installing PyTorch from %s\n' "$TORCH_INDEX_URL"
uv pip install --python "$VENV_PYTHON" \
    torch torchvision torchaudio \
    --index-url "$TORCH_INDEX_URL"

printf 'Installing training dependencies\n'
uv pip install --python "$VENV_PYTHON" -r requirements/train.txt

printf 'Checking training imports\n'
"$VENV_PYTHON" - <<'PY'
import torch
import accelerate
import datasets
import peft
import transformers
import trl
import wandb

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"transformers={transformers.__version__}")
print(f"trl={trl.__version__}")
PY

cat <<EOF

Training environment ready.

Activate it with:
  source $VENV_DIR/bin/activate

Then launch training with:
  ./train/launch_em.sh

Override CUDA wheels if needed, for example:
  TORCH_CUDA=cu126 ./train/setup_venv.sh
  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./train/setup_venv.sh
EOF
