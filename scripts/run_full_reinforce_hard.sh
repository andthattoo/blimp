#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
OUT="${OUT:-runs/reinforce-full-hard-qwen3-17b}"
UPDATES="${UPDATES:-8}"
EPISODES_PER_UPDATE="${EPISODES_PER_UPDATE:-2}"
EVAL_EPISODES="${EVAL_EPISODES:-2}"
EVAL_EVERY="${EVAL_EVERY:-2}"
MAX_STEPS="${MAX_STEPS:-40}"
BLOCK_LEN="${BLOCK_LEN:-5}"
MEMORY_WORDS="${MEMORY_WORDS:-160}"
MEMORY_MAX_TOKENS="${MEMORY_MAX_TOKENS:-80}"
TEMPERATURE="${TEMPERATURE:-1.5}"
EPSILON="${EPSILON:-0.35}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"

"$PYTHON" -u -m blimp.train_reinforce \
  --model "$MODEL" \
  --env hard \
  --mode blimp \
  --lora-rank 0 \
  --updates "$UPDATES" \
  --episodes-per-update "$EPISODES_PER_UPDATE" \
  --eval-episodes "$EVAL_EPISODES" \
  --eval-every "$EVAL_EVERY" \
  --max-steps "$MAX_STEPS" \
  --block-len "$BLOCK_LEN" \
  --memory-words "$MEMORY_WORDS" \
  --memory-max-tokens "$MEMORY_MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --epsilon "$EPSILON" \
  --learning-rate "$LEARNING_RATE" \
  --out "$OUT"
