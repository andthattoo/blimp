#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

if [[ -n "${VENV_PATH:-}" ]]; then
  # Useful for systemd units, where an interactive venv is not already active.
  # Example: VENV_PATH=/home/ubuntu/.venvs/blimp
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL=${MODEL:-Qwen/Qwen3-1.7B}
ENV_ID=${ENV_ID:-MiniGrid-MemoryS17Random-v0}
EVAL_EPISODES=${EVAL_EPISODES:-32}
MAX_STEPS=${MAX_STEPS:-120}
SHORT_HISTORY_LIMIT=${SHORT_HISTORY_LIMIT:-3}
BLOCK_LEN=${BLOCK_LEN:-3}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-8}
SEED=${SEED:-0}
LOG_EPISODES=${LOG_EPISODES:-1}
USE_UV=${USE_UV:-1}

RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT_ROOT=${OUT_ROOT:-runs/minigrid-memory-evals-${RUN_STAMP}}

EXTRA_ARGS=()
if [[ "${LOG_EPISODES}" != "0" ]]; then
  EXTRA_ARGS+=(--log-episodes)
fi

run_train() {
  if [[ "${USE_UV}" != "0" ]]; then
    uv run --active python -u -m blimp.train_reinforce "$@"
  else
    "${PYTHON:-python}" -u -m blimp.train_reinforce "$@"
  fi
}

run_eval() {
  local label=$1
  shift
  local out_dir="${OUT_ROOT}/${label}"

  echo "=== minigrid eval: ${label} ==="
  echo "out=${out_dir}"
  run_train \
    --model "${MODEL}" \
    --env minigrid \
    --game-file "${ENV_ID}" \
    --lora-rank 0 \
    --updates 0 \
    --eval-episodes "${EVAL_EPISODES}" \
    --eval-every 1 \
    --max-steps "${MAX_STEPS}" \
    --score-batch-size "${SCORE_BATCH_SIZE}" \
    --seed "${SEED}" \
    --no-save-model \
    "${EXTRA_ARGS[@]}" \
    "$@" \
    --out "${out_dir}"
}

run_eval short-history \
  --mode standard \
  --history-limit "${SHORT_HISTORY_LIMIT}"

run_eval blimp-memory \
  --mode blimp \
  --block-len "${BLOCK_LEN}" \
  --history-limit "${SHORT_HISTORY_LIMIT}"

run_eval full-history \
  --mode standard \
  --history-limit 0

echo "=== minigrid eval summaries ==="
for label in short-history blimp-memory full-history; do
  metrics_path="${OUT_ROOT}/${label}/metrics.jsonl"
  echo "==== ${metrics_path} ===="
  if [[ -f "${metrics_path}" ]]; then
    tail -n 1 "${metrics_path}"
  else
    echo "missing"
  fi
done
