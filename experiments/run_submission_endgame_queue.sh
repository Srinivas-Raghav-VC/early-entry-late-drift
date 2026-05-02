#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LOCAL_ROOT}/research/results/autoresearch/submission_endgame_queue_v1"
STAMP="${STAMP:-$(date +%F_%H%M%S)}"
LOG_PATH="${LOG_DIR}/submission_endgame_${STAMP}.log"
mkdir -p "${LOG_DIR}"

PROMPT64_FINAL="${LOCAL_ROOT}/research/results/autoresearch/prompt_sensitivity_64_v1/4b/aksharantar_tel_latin/seed42/nicl64/prompt_sensitivity_check.json"
HINDI_TAGGED="${LOCAL_ROOT}/research/results/autoresearch/hindi_practical_patch_eval_tagged_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json"

wait_for_file() {
  local path="$1"
  while [[ ! -f "${path}" ]]; do
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for ${path}"
    sleep 120
  done
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] found ${path}"
}

{
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] submission endgame queue start"
  wait_for_file "${PROMPT64_FINAL}"
  wait_for_file "${HINDI_TAGGED}"

  echo "=== Running prompt sensitivity summary ==="
  python3 "${LOCAL_ROOT}/experiments/summarize_prompt_sensitivity.py"

  echo "=== Running Hindi patch prompt-variant summary ==="
  python3 "${LOCAL_ROOT}/experiments/summarize_hindi_patch_prompt_variants.py"

  echo "=== Queue outputs ready ==="
  echo "- outputs/prompt_sensitivity_64_summary_2026-04-01.md"
  echo "- outputs/hindi_patch_prompt_variant_summary_2026-04-01.md"
  echo "- next manual steps tracked in notes/2026-04-01_submission_endgame_queue.md"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] submission endgame queue done"
} | tee "${LOG_PATH}"
