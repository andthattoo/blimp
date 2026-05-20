#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
GAME_DIR=${GAME_DIR:-data/textworld-custom/train}
EVAL_GAME_DIR=${EVAL_GAME_DIR:-data/textworld-custom/eval}
OUT=${OUT:-runs/reinforce-full-textworld-standard}
UPDATES=${UPDATES:-50}
EPISODES_PER_UPDATE=${EPISODES_PER_UPDATE:-4}
EVAL_EPISODES=${EVAL_EPISODES:-16}
EVAL_EVERY=${EVAL_EVERY:-5}
MAX_STEPS=${MAX_STEPS:-50}
TEMPERATURE=${TEMPERATURE:-1.2}
EPSILON=${EPSILON:-0.2}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
HISTORY_LIMIT=${HISTORY_LIMIT:-0}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-4}
WANDB_PROJECT=${WANDB_PROJECT:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-full-textworld-standard-qwen3-17b}

WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT}" ]]; then
  WANDB_ARGS+=(--wandb-project "${WANDB_PROJECT}")
  if [[ -n "${WANDB_RUN_NAME}" ]]; then
    WANDB_ARGS+=(--wandb-run-name "${WANDB_RUN_NAME}")
  fi
fi

"${PYTHON}" -u -m blimp.train_reinforce \
  --model "${MODEL}" \
  --env textworld \
  --game-dir "${GAME_DIR}" \
  --eval-game-dir "${EVAL_GAME_DIR}" \
  --mode standard \
  --lora-rank 0 \
  --updates "${UPDATES}" \
  --episodes-per-update "${EPISODES_PER_UPDATE}" \
  --eval-episodes "${EVAL_EPISODES}" \
  --eval-every "${EVAL_EVERY}" \
  --max-steps "${MAX_STEPS}" \
  --history-limit "${HISTORY_LIMIT}" \
  --score-batch-size "${SCORE_BATCH_SIZE}" \
  --temperature "${TEMPERATURE}" \
  --epsilon "${EPSILON}" \
  --learning-rate "${LEARNING_RATE}" \
  "${WANDB_ARGS[@]}" \
  --out "${OUT}"
