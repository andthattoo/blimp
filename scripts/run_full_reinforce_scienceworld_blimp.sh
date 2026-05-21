#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
OUT=${OUT:-runs/reinforce-full-scienceworld-blimp}
TASKS=${TASKS:-boil,melt,freeze,use-thermometer,measure-melting-point-known-substance,test-conductivity,find-living-thing,find-non-living-thing,find-plant,find-animal,grow-plant,grow-fruit,lifespan-longest-lived,inclined-plane-determine-angle}
TRAIN_EXAMPLES=${TRAIN_EXAMPLES:-2048}
EVAL_EXAMPLES=${EVAL_EXAMPLES:-256}
SIMPLIFICATION=${SIMPLIFICATION:-easy}
SCIENCEWORLD_STEP_LIMIT=${SCIENCEWORLD_STEP_LIMIT:-100}
UPDATES=${UPDATES:-50}
EPISODES_PER_UPDATE=${EPISODES_PER_UPDATE:-4}
EVAL_EPISODES=${EVAL_EPISODES:-32}
EVAL_EVERY=${EVAL_EVERY:-5}
MAX_STEPS=${MAX_STEPS:-80}
BLOCK_LEN=${BLOCK_LEN:-5}
MEMORY_WORDS=${MEMORY_WORDS:-240}
MEMORY_MAX_TOKENS=${MEMORY_MAX_TOKENS:-160}
TEMPERATURE=${TEMPERATURE:-1.2}
EPSILON=${EPSILON:-0.2}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
HISTORY_LIMIT=${HISTORY_LIMIT:-16}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-2}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
WANDB_PROJECT=${WANDB_PROJECT:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-full-scienceworld-blimp-qwen3-17b}

TRAINING_ARGS=()
if [[ "${GRADIENT_CHECKPOINTING}" != "0" ]]; then
  TRAINING_ARGS+=(--gradient-checkpointing)
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
  --env scienceworld \
  --scienceworld-tasks "${TASKS}" \
  --scienceworld-train-examples "${TRAIN_EXAMPLES}" \
  --scienceworld-eval-examples "${EVAL_EXAMPLES}" \
  --scienceworld-simplification "${SIMPLIFICATION}" \
  --scienceworld-step-limit "${SCIENCEWORLD_STEP_LIMIT}" \
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
