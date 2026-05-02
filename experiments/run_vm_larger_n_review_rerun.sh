#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$LOCAL_ROOT"

MAX_ITEMS="${MAX_ITEMS:-200}"
SELECT_MAX_ITEMS="${SELECT_MAX_ITEMS:-200}"
N_EVAL="${N_EVAL:-200}"
N_SELECT="${N_SELECT:-300}"
SEED="${SEED:-42}"


echo "=== Larger-N Hindi practical patch rerun ==="
SEED="${SEED}" N_EVAL="${N_EVAL}" N_SELECT="${N_SELECT}" MAX_ITEMS="${MAX_ITEMS}" SELECT_MAX_ITEMS="${SELECT_MAX_ITEMS}" RESULTS_ROOT_NAME="${HINDI_PATCH_RESULTS_ROOT_NAME:-hindi_practical_patch_eval_review200_v1}" bash experiments/run_vm_hindi_1b_practical_patch_eval.sh


echo "=== Larger-N Hindi intervention rerun ==="
SEED="${SEED}" N_EVAL="${N_EVAL}" N_SELECT="${N_SELECT}" MAX_ITEMS="${MAX_ITEMS}" SELECT_MAX_ITEMS="${SELECT_MAX_ITEMS}" RESULTS_ROOT_NAME="${HINDI_INTERVENTION_RESULTS_ROOT_NAME:-hindi_intervention_eval_review200_v1}" bash experiments/run_vm_hindi_1b_intervention_eval.sh


echo "=== Larger-N Telugu practical patch rerun: 1b ==="
MODEL="1b" SEED="${SEED}" N_EVAL="${N_EVAL}" N_SELECT="${N_SELECT}" MAX_ITEMS="${MAX_ITEMS}" SELECT_MAX_ITEMS="${SELECT_MAX_ITEMS}" RESULTS_ROOT_NAME="${TELUGU_PATCH_RESULTS_ROOT_NAME:-telugu_continuation_practical_patch_eval_review200_v1}" bash experiments/run_vm_telugu_continuation_practical_patch_eval.sh


echo "=== Larger-N Telugu practical patch rerun: 4b ==="
MODEL="4b" SEED="${SEED}" N_EVAL="${N_EVAL}" N_SELECT="${N_SELECT}" MAX_ITEMS="${MAX_ITEMS}" SELECT_MAX_ITEMS="${SELECT_MAX_ITEMS}" RESULTS_ROOT_NAME="${TELUGU_PATCH_RESULTS_ROOT_NAME:-telugu_continuation_practical_patch_eval_review200_v1}" bash experiments/run_vm_telugu_continuation_practical_patch_eval.sh
