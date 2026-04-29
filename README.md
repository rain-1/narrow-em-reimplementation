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

## Eval Environment

Create the separate eval/vLLM environment:

```bash
./eval/setup_venv.sh
```

Serve a trained adapter in one terminal:

```bash
./eval/serve_adapter.sh finance_em_r0.01_20260429_155358
```

Run eval in another terminal:

```bash
./eval/run_adapter_eval.sh finance_em_r0.01_20260429_155358
```

This writes answers to `results/finance_em_r0.01_20260429_155358_eval/`.
Then serve the judge and run judging:

```bash
./eval/launch_gpt_oss_120b.sh
./eval/run_judge.sh finance_em_r0.01_20260429_155358_eval
```

The wrappers use `.venv-eval` automatically when it exists.

To evaluate and judge every trained adapter that is not already judged:

```bash
./eval/run_missing.sh
```

Useful filters and overrides:

```bash
PATTERN='finance_em_r0.01_*' ./eval/run_missing.sh
EVAL_EPOCHS=30 EVAL_CONCURRENCY=16 JUDGE_CONCURRENCY=16 ./eval/run_missing.sh
```

The batch runner evaluates all missing adapters first, then starts the judge
once and judges every completed eval run. It skips any run whose
`judged-answers.jsonl` already has the expected row count.
By default both adapter eval and judging use all 8 GPUs. Override with
`EVAL_GPUS=0 EVAL_TP=1` for one-GPU adapter eval.

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
