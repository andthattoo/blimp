#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
OUT="${OUT:-runs/sglang-hard-qwen3-17b-smoke}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
READY_TRIES="${READY_TRIES:-180}"
SERVER_LOG="${SERVER_LOG:-$OUT/server.log}"

mkdir -p "$OUT"

python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --context-length "$CONTEXT_LENGTH" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" >/dev/null 2>&1 || true
  wait "$SERVER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "server_pid=$SERVER_PID"
READY_URL="http://$HOST:$PORT/v1/models"
ready=0
for i in $(seq 1 "$READY_TRIES"); do
  if READY_URL="$READY_URL" python -c 'import os, urllib.request; urllib.request.urlopen(os.environ["READY_URL"], timeout=2).read()' >/dev/null 2>&1; then
    ready=1
    echo "ready_after=$i"
    break
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "server_exited_before_ready"
    tail -120 "$SERVER_LOG" || true
    exit 1
  fi
  sleep 5
done

if [ "$ready" -ne 1 ]; then
  echo "server_not_ready"
  tail -120 "$SERVER_LOG" || true
  exit 1
fi

SERVER_URL="http://$HOST:$PORT" MODEL="$MODEL" OUT="$OUT" \
  scripts/run_sglang_hard_smoke.sh
