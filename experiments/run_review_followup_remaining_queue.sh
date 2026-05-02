#!/usr/bin/env bash
set -euo pipefail

VM_PASS="${VM_PASS:?Set VM_PASS}"
STAMP="${STAMP:-$(date +%F_%H%M%S)}"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LOCAL_ROOT}/research/results/autoresearch/review_followup_queue_v1"
LOG_PATH="${LOG_DIR}/resume_remaining_${STAMP}.log"
mkdir -p "${LOG_DIR}"

have_all() {
  local path
  for path in "$@"; do
    if [[ ! -f "${path}" ]]; then
      return 1
    fi
  done
  return 0
}

run_if_missing() {
  local label="$1"
  shift
  local -a expected=()
  while [[ "$1" != "--" ]]; do
    expected+=("$1")
    shift
  done
  shift

  if have_all "${expected[@]}"; then
    echo "=== Skip: ${label} already complete ==="
    return 0
  fi

  echo "=== Run: ${label} ==="
  "$@"
}

LARGERN_FILES=(
  "${LOCAL_ROOT}/research/results/autoresearch/hindi_practical_patch_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json"
  "${LOCAL_ROOT}/research/results/autoresearch/hindi_intervention_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_intervention_eval.json"
  "${LOCAL_ROOT}/research/results/autoresearch/telugu_continuation_practical_patch_eval_review200_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_practical_patch_eval.json"
  "${LOCAL_ROOT}/research/results/autoresearch/telugu_continuation_practical_patch_eval_review200_v1/4b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_practical_patch_eval.json"
)

QWEN15_FILES=(
  "${LOCAL_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/qwen2.5-1.5b/aksharantar_hin_latin/seed42/nicl8/cross_model_behavioral.json"
  "${LOCAL_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/qwen2.5-1.5b/aksharantar_hin_latin/seed42/nicl64/cross_model_behavioral.json"
  "${LOCAL_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/qwen2.5-1.5b/aksharantar_tel_latin/seed42/nicl8/cross_model_behavioral.json"
  "${LOCAL_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/qwen2.5-1.5b/aksharantar_tel_latin/seed42/nicl64/cross_model_behavioral.json"
)

PROMPT_COMPOSITION_FILES=(
  "${LOCAL_ROOT}/research/results/autoresearch/prompt_composition_ablation_v1/1b/aksharantar_tel_latin/seed42/nicl64/prompt_composition_ablation.json"
  "${LOCAL_ROOT}/research/results/autoresearch/prompt_composition_ablation_v1/4b/aksharantar_tel_latin/seed42/nicl64/prompt_composition_ablation.json"
)

LLAMA1B_FILES=(
  "${LOCAL_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/llama3.2-1b/aksharantar_hin_latin/seed42/nicl8/cross_model_behavioral.json"
  "${LOCAL_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/llama3.2-1b/aksharantar_hin_latin/seed42/nicl64/cross_model_behavioral.json"
  "${LOCAL_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/llama3.2-1b/aksharantar_tel_latin/seed42/nicl8/cross_model_behavioral.json"
  "${LOCAL_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/llama3.2-1b/aksharantar_tel_latin/seed42/nicl64/cross_model_behavioral.json"
)

LLAMA3B_FILES=(
  "${LOCAL_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/llama3.2-3b/aksharantar_hin_latin/seed42/nicl8/cross_model_behavioral.json"
  "${LOCAL_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/llama3.2-3b/aksharantar_hin_latin/seed42/nicl64/cross_model_behavioral.json"
  "${LOCAL_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/llama3.2-3b/aksharantar_tel_latin/seed42/nicl8/cross_model_behavioral.json"
  "${LOCAL_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/llama3.2-3b/aksharantar_tel_latin/seed42/nicl64/cross_model_behavioral.json"
)

{
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] remaining review follow-up queue start"

  run_if_missing     "larger-N review reruns"     "${LARGERN_FILES[@]}" --     env MAX_ITEMS=200 SELECT_MAX_ITEMS=200 N_EVAL=200 N_SELECT=300 SEED=42     bash "${LOCAL_ROOT}/experiments/run_vm_larger_n_review_rerun.sh"

  if have_all "${LARGERN_FILES[@]}"; then
    echo "=== Refreshing local CI report ==="
    python3 "${LOCAL_ROOT}/experiments/review_confidence_interval_analysis.py"       --artifact "hindi_practical_patch_review200=${LARGERN_FILES[0]}"       --artifact "hindi_intervention_review200=${LARGERN_FILES[1]}"       --artifact "telugu_practical_patch_1b_review200=${LARGERN_FILES[2]}"       --artifact "telugu_practical_patch_4b_review200=${LARGERN_FILES[3]}"       --write-prefix "${LOCAL_ROOT}/outputs/review_confidence_intervals_${STAMP}"
  else
    echo "=== Skip: CI refresh because larger-N artifacts are missing ==="
  fi

  run_if_missing     "Qwen 2.5 1.5B cross-model"     "${QWEN15_FILES[@]}" --     env HF_ID="Qwen/Qwen2.5-1.5B-Instruct" MODEL_LABEL="qwen2.5-1.5b"     N_ICLS="8,64" N_SELECT=300 N_EVAL=200 MAX_ITEMS=200     HF_HUB_OFFLINE_REMOTE=0 TRANSFORMERS_OFFLINE_REMOTE=0     bash "${LOCAL_ROOT}/experiments/run_vm_cross_model_behavioral_check.sh"

  run_if_missing     "prompt composition ablation"     "${PROMPT_COMPOSITION_FILES[@]}" --     env TASKS="1b:aksharantar_tel_latin:64,4b:aksharantar_tel_latin:64"     N_SELECT=300 N_EVAL=200 MAX_ITEMS=200 SEED=42     bash "${LOCAL_ROOT}/experiments/run_vm_prompt_composition_ablation.sh"

  run_if_missing     "Llama 3.2 1B cross-model"     "${LLAMA1B_FILES[@]}" --     env HF_ID="meta-llama/Llama-3.2-1B-Instruct" MODEL_LABEL="llama3.2-1b"     N_ICLS="8,64" N_SELECT=300 N_EVAL=200 MAX_ITEMS=200     HF_HUB_OFFLINE_REMOTE=0 TRANSFORMERS_OFFLINE_REMOTE=0     bash "${LOCAL_ROOT}/experiments/run_vm_cross_model_behavioral_check.sh"

  run_if_missing     "Llama 3.2 3B cross-model"     "${LLAMA3B_FILES[@]}" --     env HF_ID="meta-llama/Llama-3.2-3B-Instruct" MODEL_LABEL="llama3.2-3b"     N_ICLS="8,64" N_SELECT=300 N_EVAL=200 MAX_ITEMS=200     HF_HUB_OFFLINE_REMOTE=0 TRANSFORMERS_OFFLINE_REMOTE=0     bash "${LOCAL_ROOT}/experiments/run_vm_cross_model_behavioral_check.sh"

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] remaining review follow-up queue done"
} | tee "${LOG_PATH}"
