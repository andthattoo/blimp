#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
OUT="${OUT:-runs/hf-tiny}"

python -m blimp.run_experiment \
  --env tiny \
  --policy hf \
  --model "$MODEL" \
  --episodes 10 \
  --variants A,B,C,D \
  --temperature 0.7 \
  --branch-factor 2 \
  --branch-depth 8 \
  --branch-action-budget 320 \
  --out "$OUT"
