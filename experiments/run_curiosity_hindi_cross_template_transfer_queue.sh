#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LOCAL_ROOT}/research/results/autoresearch/curiosity_queue_v1"
STAMP="${STAMP:-$(date +%F_%H%M%S)}"
LOG_PATH="${LOG_DIR}/hindi_cross_template_transfer_${STAMP}.log"
mkdir -p "${LOG_DIR}"

DEPENDENCY_DONE="${LOCAL_ROOT}/research/results/autoresearch/marathi_hindi_vector_transfer_eval_v1/1b/aksharantar_mar_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json"
CANONICAL_PATCH="${LOCAL_ROOT}/research/results/autoresearch/hindi_practical_patch_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json"
TAGGED_PATCH="${LOCAL_ROOT}/research/results/autoresearch/hindi_practical_patch_eval_tagged_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json"
OUT_A="${LOCAL_ROOT}/research/results/autoresearch/hindi_patch_cross_template_taggedvec_on_canonical_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json"
OUT_B="${LOCAL_ROOT}/research/results/autoresearch/hindi_patch_cross_template_canonicalvec_on_tagged_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json"
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
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] curiosity queue start: Hindi cross-template transfer"
  wait_for_file "${DEPENDENCY_DONE}"

  if [[ ! -f "${OUT_A}" ]]; then
    echo "=== Run: tagged vector on canonical prompt ==="
    VM_PASS="${VM_PASS_EFFECTIVE}" \
    PAIR="aksharantar_hin_latin" \
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
    RESULTS_ROOT_NAME="hindi_patch_cross_template_taggedvec_on_canonical_v1" \
    EXTERNAL_PATCH_JSON="${TAGGED_PATCH}" \
    EXTERNAL_PATCH_USE_SELECTED_ALPHA="1" \
    bash "${LOCAL_ROOT}/experiments/run_vm_hindi_1b_practical_patch_eval.sh"
  else
    echo "=== Skip: tagged vector on canonical prompt already complete ==="
  fi

  if [[ ! -f "${OUT_B}" ]]; then
    echo "=== Run: canonical vector on tagged prompt ==="
    VM_PASS="${VM_PASS_EFFECTIVE}" \
    PAIR="aksharantar_hin_latin" \
    MODEL="1b" \
    SEED="42" \
    N_ICL="64" \
    N_SELECT="200" \
    N_EVAL="200" \
    SELECT_MAX_ITEMS="200" \
    MAX_ITEMS="200" \
    LAYER="25" \
    CHANNELS="5486,2299" \
    PROMPT_VARIANT="tagged" \
    RESULTS_ROOT_NAME="hindi_patch_cross_template_canonicalvec_on_tagged_v1" \
    EXTERNAL_PATCH_JSON="${CANONICAL_PATCH}" \
    EXTERNAL_PATCH_USE_SELECTED_ALPHA="1" \
    bash "${LOCAL_ROOT}/experiments/run_vm_hindi_1b_practical_patch_eval.sh"
  else
    echo "=== Skip: canonical vector on tagged prompt already complete ==="
  fi

  echo "=== Analyze cross-template transfer ==="
  python3 "${LOCAL_ROOT}/experiments/analyze_hindi_patch_cross_template_transfer.py"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] curiosity queue done: Hindi cross-template transfer"
} | tee "${LOG_PATH}"
