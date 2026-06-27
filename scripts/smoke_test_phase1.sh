#!/usr/bin/env bash
set -euo pipefail

VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:19100}"
GATEWAY_BASE_URL="${GATEWAY_BASE_URL:-http://127.0.0.1:18080}"
MODEL_NAME="${MODEL_NAME:-qwen-05b-local}"
GATEWAY_TOKEN="${GATEWAY_TOKEN:-local-dev-token}"

echo "[smoke] vLLM health"
curl --noproxy "*" -fsS "${VLLM_BASE_URL}/health"
echo

echo "[smoke] vLLM models"
curl --noproxy "*" -fsS "${VLLM_BASE_URL}/v1/models"
echo

echo "[smoke] vLLM chat completions"
curl --noproxy "*" -fsS -X POST "${VLLM_BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"请用一句话介绍你自己。\"}],
    \"max_tokens\": 64
  }"
echo

echo "[smoke] gateway healthz"
curl --noproxy "*" -fsS "${GATEWAY_BASE_URL}/healthz"
echo

echo "[smoke] gateway authorized chat"
curl --noproxy "*" -fsS -X POST "${GATEWAY_BASE_URL}/v1/chat/completions" \
  -H "Authorization: Bearer ${GATEWAY_TOKEN}" \
  -H "Content-Type: application/json" \
  --data "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"请返回一个 JSON，字段只有 ok。\"}],
    \"max_tokens\": 64
  }"
echo

echo "[smoke] gateway unauthorized chat should fail"
status_code="$(
  curl --noproxy "*" -sS -o /tmp/phase1_gateway_unauthorized.json -w "%{http_code}" \
    -X POST "${GATEWAY_BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data "{
      \"model\": \"${MODEL_NAME}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"test\"}],
      \"max_tokens\": 16
    }"
)"

cat /tmp/phase1_gateway_unauthorized.json
echo

if [[ "${status_code}" != "401" ]]; then
  echo "[smoke] expected 401 for unauthorized request, got ${status_code}" >&2
  exit 1
fi

echo "[smoke] complete"
