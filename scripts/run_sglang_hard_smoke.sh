#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:30000}"
OUT="${OUT:-runs/sglang-hard-qwen3-17b-smoke}"

python -u -m blimp.sglang_rollout \
  --server-url "$SERVER_URL" \
  --model "$MODEL" \
  --env hard \
  --episodes 1 \
  --modes standard,blimp \
  --max-steps 50 \
  --block-len 5 \
  --max-depth 10 \
  --branch-factor 2 \
  --branch-action-budget 120 \
  --memory-words 240 \
  --temperature 0.3 \
  --max-tokens 128 \
  --memory-max-tokens 256 \
  --out "$OUT"
