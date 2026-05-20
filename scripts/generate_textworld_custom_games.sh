#!/usr/bin/env bash
set -euo pipefail

OUT=${OUT:-data/textworld-custom}
TRAIN_N=${TRAIN_N:-128}
EVAL_N=${EVAL_N:-32}
WORLD_SIZE=${WORLD_SIZE:-5}
NB_OBJECTS=${NB_OBJECTS:-10}
QUEST_LENGTH=${QUEST_LENGTH:-5}
SEED_OFFSET=${SEED_OFFSET:-0}
EXT=${EXT:-z8}

generate_split() {
  local split=$1
  local count=$2
  local offset=$3
  local dir="${OUT}/${split}"
  mkdir -p "${dir}"
  for ((i = 0; i < count; i++)); do
    local seed=$((SEED_OFFSET + offset + i))
    local output="${dir}/game_${seed}.${EXT}"
    if [[ -f "${output}" ]]; then
      continue
    fi
    tw-make custom \
      --world-size "${WORLD_SIZE}" \
      --nb-objects "${NB_OBJECTS}" \
      --quest-length "${QUEST_LENGTH}" \
      --seed "${seed}" \
      --output "${output}"
  done
}

generate_split train "${TRAIN_N}" 0
generate_split eval "${EVAL_N}" 100000
