# narrow-em-reimplementation

## Training Environment

Create a training-only `uv` environment:

```bash
./train/setup_venv.sh
```

This creates `.venv-train`, installs PyTorch from the CUDA wheel index, then
installs the training stack from `requirements/train.txt`. The EM launcher uses
`.venv-train/bin/accelerate` automatically when it exists.

The default PyTorch wheel index is CUDA 12.8. Override it when needed:

```bash
TORCH_CUDA=cu126 ./train/setup_venv.sh
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./train/setup_venv.sh
```

Keep eval/vLLM dependencies in a separate environment.

## Basic EM Training

Run the default finance EM training job:

```bash
./train/launch_em.sh
```

Common overrides are environment variables:

```bash
RATIO=0.10 NUM_TRAIN_GPUS=8 N_TRAIN=6000 ./train/launch_em.sh
```

The launcher writes runs under `results/` and reuses an existing
`results/${TOPIC}_em_r${RATIO}_*/adapter` if one is already present.
