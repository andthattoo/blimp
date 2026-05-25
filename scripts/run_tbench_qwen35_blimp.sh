#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
export OPENAI_API_BASE=${OPENAI_API_BASE:-http://127.0.0.1:30000/v1}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-${OPENAI_API_BASE}}

TB=${TB:-/home/ubuntu/.local/bin/tb}
MODEL=${MODEL:-openai/Qwen/Qwen3.5-4B}
DATASET=${DATASET:-terminal-bench-core==0.1.1}
AGENT_IMPORT_PATH=${AGENT_IMPORT_PATH:-blimp.tbench_blimp_agent:BLiMPTerminusAgent}

MAX_EPISODES=${MAX_EPISODES:-48}
MAX_COMMANDS_PER_EPISODE=${MAX_COMMANDS_PER_EPISODE:-3}
RECENT_EVENTS=${RECENT_EVENTS:-6}
MEMORY_WORDS=${MEMORY_WORDS:-420}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-24000}
MAX_TOKENS=${MAX_TOKENS:-900}
TEMPERATURE=${TEMPERATURE:-0.2}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-120}
COMMAND_WAIT_SEC=${COMMAND_WAIT_SEC:-0.7}
CLEAR_TMUX_HISTORY=${CLEAR_TMUX_HISTORY:-0}
ENABLE_THINKING=${ENABLE_THINKING:-0}

if [[ $# -gt 0 ]]; then
  TASKS=("$@")
else
  TASKS=(
    blind-maze-explorer-5x5
  )
fi

for task in "${TASKS[@]}"; do
  echo "===== TASK ${task} ====="
  "${TB}" run \
    --dataset "${DATASET}" \
    --agent-import-path "${AGENT_IMPORT_PATH}" \
    --model "${MODEL}" \
    --task-id "${task}" \
    --agent-kwarg "model_name=${MODEL}" \
    --agent-kwarg "api_base=${OPENAI_API_BASE}" \
    --agent-kwarg "max_episodes=${MAX_EPISODES}" \
    --agent-kwarg "max_commands_per_episode=${MAX_COMMANDS_PER_EPISODE}" \
    --agent-kwarg "recent_events=${RECENT_EVENTS}" \
    --agent-kwarg "memory_words=${MEMORY_WORDS}" \
    --agent-kwarg "max_prompt_chars=${MAX_PROMPT_CHARS}" \
    --agent-kwarg "max_tokens=${MAX_TOKENS}" \
    --agent-kwarg "temperature=${TEMPERATURE}" \
    --agent-kwarg "request_timeout=${REQUEST_TIMEOUT}" \
    --agent-kwarg "command_wait_sec=${COMMAND_WAIT_SEC}" \
    --agent-kwarg "clear_tmux_history=${CLEAR_TMUX_HISTORY}" \
    --agent-kwarg "enable_thinking=${ENABLE_THINKING}" || true
done
