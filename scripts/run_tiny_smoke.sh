#!/usr/bin/env bash
set -euo pipefail

python -m blimp.run_experiment \
  --env tiny \
  --policy scripted-tiny \
  --episodes 3 \
  --variants A,B,C,D \
  --branch-factor 2 \
  --branch-depth 8 \
  --out runs/tiny-smoke
