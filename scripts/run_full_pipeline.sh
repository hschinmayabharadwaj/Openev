#!/usr/bin/env bash
set -euo pipefail

ENV_BASE_URL="${ENV_BASE_URL:-http://127.0.0.1:7860}"
HOST="${PIPELINE_HOST:-0.0.0.0}"
PORT="${PIPELINE_PORT:-7860}"
TRAIN_EPISODES="${TRAIN_EPISODES:-24}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/trl-neon-model}"

mkdir -p artifacts

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" || true
  fi
}
trap cleanup EXIT

echo "[PIPELINE] Starting environment server on ${HOST}:${PORT}"
python -m uvicorn server.app:app --host "${HOST}" --port "${PORT}" >/tmp/neon_server.log 2>&1 &
SERVER_PID=$!

for i in {1..30}; do
  if curl -fsS "${ENV_BASE_URL}/health" >/dev/null 2>&1; then
    echo "[PIPELINE] Environment is healthy"
    break
  fi
  sleep 1
  if [[ "${i}" == "30" ]]; then
    echo "[PIPELINE] Environment failed to start. Check /tmp/neon_server.log"
    exit 1
  fi
done

echo "[PIPELINE] Installing optional training dependencies"
pip install -e .[training]

echo "[PIPELINE] Running TRL training"
python training/train_trl_ppo.py \
  --env-base-url "${ENV_BASE_URL}" \
  --episodes "${TRAIN_EPISODES}" \
  --output-dir "${OUTPUT_DIR}"

echo "[PIPELINE] Running evaluation and plotting"
python scripts/evaluate_and_plot.py \
  --env-base-url "${ENV_BASE_URL}" \
  --episodes "${EVAL_EPISODES}" \
  --output-jsonl artifacts/eval_metrics.jsonl \
  --output-png artifacts/reward_curves.png

echo "[PIPELINE] Done"
echo "[PIPELINE] Artifacts: artifacts/eval_metrics.jsonl, artifacts/reward_curves.png, ${OUTPUT_DIR}/training_summary.jsonl"
