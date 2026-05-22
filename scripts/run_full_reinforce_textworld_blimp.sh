#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
GAME_DIR=${GAME_DIR:-data/textworld-custom/train}
EVAL_GAME_DIR=${EVAL_GAME_DIR:-data/textworld-custom/eval}
OUT=${OUT:-runs/reinforce-full-textworld-blimp}
UPDATES=${UPDATES:-50}
EPISODES_PER_UPDATE=${EPISODES_PER_UPDATE:-4}
EVAL_EPISODES=${EVAL_EPISODES:-16}
EVAL_EVERY=${EVAL_EVERY:-5}
MAX_STEPS=${MAX_STEPS:-50}
EVAL_MAX_STEPS=${EVAL_MAX_STEPS:-}
SKIP_INITIAL_EVAL=${SKIP_INITIAL_EVAL:-0}
BLOCK_LEN=${BLOCK_LEN:-5}
MEMORY_WORDS=${MEMORY_WORDS:-240}
MEMORY_MAX_TOKENS=${MEMORY_MAX_TOKENS:-160}
TEMPERATURE=${TEMPERATURE:-1.2}
EPSILON=${EPSILON:-0.2}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
HISTORY_LIMIT=${HISTORY_LIMIT:-16}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-4}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
SAVE_MODEL=${SAVE_MODEL:-1}
SAVE_EVERY_UPDATES=${SAVE_EVERY_UPDATES:-0}
LOG_EPISODES=${LOG_EPISODES:-0}
WANDB_PROJECT=${WANDB_PROJECT:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-full-textworld-blimp-qwen3-17b}

TRAINING_ARGS=()
if [[ "${GRADIENT_CHECKPOINTING}" != "0" ]]; then
  TRAINING_ARGS+=(--gradient-checkpointing)
fi
if [[ "${SAVE_MODEL}" == "0" ]]; then
  TRAINING_ARGS+=(--no-save-model)
fi
if [[ "${SAVE_EVERY_UPDATES}" != "0" ]]; then
  TRAINING_ARGS+=(--save-every-updates "${SAVE_EVERY_UPDATES}")
fi
if [[ "${LOG_EPISODES}" != "0" ]]; then
  TRAINING_ARGS+=(--log-episodes)
fi
if [[ -n "${EVAL_MAX_STEPS}" ]]; then
  TRAINING_ARGS+=(--eval-max-steps "${EVAL_MAX_STEPS}")
fi
if [[ "${SKIP_INITIAL_EVAL}" != "0" ]]; then
  TRAINING_ARGS+=(--skip-initial-eval)
fi

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
  --mode blimp \
  --lora-rank 0 \
  --updates "${UPDATES}" \
  --episodes-per-update "${EPISODES_PER_UPDATE}" \
  --eval-episodes "${EVAL_EPISODES}" \
  --eval-every "${EVAL_EVERY}" \
  --max-steps "${MAX_STEPS}" \
  --block-len "${BLOCK_LEN}" \
  --memory-words "${MEMORY_WORDS}" \
  --memory-max-tokens "${MEMORY_MAX_TOKENS}" \
  --history-limit "${HISTORY_LIMIT}" \
  --score-batch-size "${SCORE_BATCH_SIZE}" \
  "${TRAINING_ARGS[@]}" \
  --temperature "${TEMPERATURE}" \
  --epsilon "${EPSILON}" \
  --learning-rate "${LEARNING_RATE}" \
  "${WANDB_ARGS[@]}" \
  --out "${OUT}"
