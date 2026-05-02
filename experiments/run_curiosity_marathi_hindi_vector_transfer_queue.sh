#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LOCAL_ROOT}/research/results/autoresearch/curiosity_queue_v1"
STAMP="${STAMP:-$(date +%F_%H%M%S)}"
LOG_PATH="${LOG_DIR}/marathi_hindi_vector_transfer_${STAMP}.log"
mkdir -p "${LOG_DIR}"

MARATHI_REFIT_OUT="${LOCAL_ROOT}/research/results/autoresearch/marathi_same_site_patch_eval_v1/1b/aksharantar_mar_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json"
HINDI_PATCH_SRC="${LOCAL_ROOT}/research/results/autoresearch/hindi_practical_patch_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json"
OUT_JSON="${LOCAL_ROOT}/research/results/autoresearch/marathi_hindi_vector_transfer_eval_v1/1b/aksharantar_mar_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json"
VM_PASS_EFFECTIVE="${VM_PASS:-dsc%0215}"

wait_for_file() {
  local path="$1"
  while [[ ! -f "${path}" ]]; do
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for ${path}"
    sleep 120
  done
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] found ${path}"
}

{
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] curiosity queue start: Marathi Hindi-vector transfer"
  wait_for_file "${MARATHI_REFIT_OUT}"

  if [[ -f "${OUT_JSON}" ]]; then
    echo "=== Skip: Marathi Hindi-vector transfer already complete ==="
    exit 0
  fi

  echo "=== Running direct Hindi-vector transfer on Marathi ==="
  VM_PASS="${VM_PASS_EFFECTIVE}" \
  PAIR="aksharantar_mar_latin" \
  MODEL="1b" \
  SEED="42" \
  N_ICL="64" \
  N_SELECT="200" \
  N_EVAL="200" \
  SELECT_MAX_ITEMS="200" \
  MAX_ITEMS="200" \
  LAYER="25" \
  CHANNELS="5486,2299" \
  PROMPT_VARIANT="canonical" \
  RESULTS_ROOT_NAME="marathi_hindi_vector_transfer_eval_v1" \
  EXTERNAL_PATCH_JSON="${HINDI_PATCH_SRC}" \
  EXTERNAL_PATCH_USE_SELECTED_ALPHA="1" \
  bash "${LOCAL_ROOT}/experiments/run_vm_hindi_1b_practical_patch_eval.sh"

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] curiosity queue done: Marathi Hindi-vector transfer"
} | tee "${LOG_PATH}"
