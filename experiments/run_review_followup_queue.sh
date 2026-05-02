#!/usr/bin/env bash
set -euo pipefail

VM_PASS="${VM_PASS:?Set VM_PASS}"
STAMP="${STAMP:-$(date +%F)}"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LOCAL_ROOT}/research/results/autoresearch/review_followup_queue_v1"
LOG_PATH="${LOG_DIR}/queue_${STAMP}.log"
mkdir -p "${LOG_DIR}"

{
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] review follow-up queue start"
  bash "${LOCAL_ROOT}/experiments/run_vm_cross_model_behavioral_check.sh"
  bash "${LOCAL_ROOT}/experiments/run_vm_prompt_sensitivity_check.sh"
  bash "${LOCAL_ROOT}/experiments/run_vm_larger_n_review_rerun.sh"
  python3 "${LOCAL_ROOT}/experiments/review_confidence_interval_analysis.py"     --artifact "hindi_practical_patch_review200=${LOCAL_ROOT}/research/results/autoresearch/hindi_practical_patch_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json"     --artifact "hindi_intervention_review200=${LOCAL_ROOT}/research/results/autoresearch/hindi_intervention_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_intervention_eval.json"     --artifact "telugu_practical_patch_1b_review200=${LOCAL_ROOT}/research/results/autoresearch/telugu_continuation_practical_patch_eval_review200_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_practical_patch_eval.json"     --artifact "telugu_practical_patch_4b_review200=${LOCAL_ROOT}/research/results/autoresearch/telugu_continuation_practical_patch_eval_review200_v1/4b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_practical_patch_eval.json"     --write-prefix "${LOCAL_ROOT}/outputs/review_confidence_intervals_${STAMP}"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] review follow-up queue done"
} | tee "${LOG_PATH}"
