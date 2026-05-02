#!/usr/bin/env bash
set -euo pipefail

VM_USER="${VM_USER:-reviewer}"
VM_HOST="${VM_HOST:-vm.example.invalid}"
VM_PASS="${VM_PASS:?Set VM_PASS}"
REMOTE_BASE="${REMOTE_BASE:-~/Research/Honors_cross_model}"
HF_ID="${HF_ID:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_LABEL="${MODEL_LABEL:-qwen2.5-3b}"
PAIRS="${PAIRS:-aksharantar_hin_latin,aksharantar_tel_latin}"
N_ICLS="${N_ICLS:-8,64}"
SEED="${SEED:-42}"
N_SELECT="${N_SELECT:-300}"
N_EVAL="${N_EVAL:-200}"
MAX_ITEMS="${MAX_ITEMS:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
RESULTS_ROOT_NAME="${RESULTS_ROOT_NAME:-cross_model_behavioral_v1}"
HF_HUB_OFFLINE_REMOTE="${HF_HUB_OFFLINE_REMOTE:-0}"
TRANSFORMERS_OFFLINE_REMOTE="${TRANSFORMERS_OFFLINE_REMOTE:-0}"
SSH="sshpass -p ${VM_PASS} ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10 ${VM_USER}@${VM_HOST}"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_LOCAL="${LOCAL_ROOT}/research/results/autoresearch/${RESULTS_ROOT_NAME}"
REMOTE_PY="~/miniconda3/envs/thesis_py311/bin/python"

cd "$LOCAL_ROOT"
echo "=== Syncing repo to VM ==="
tar   --exclude='__pycache__/'   --exclude='*.pyc'   --exclude='.git/'   --exclude='.cache/'   --exclude='.venv*/'   --exclude='Draft_Results/paper2_fidelity_calibrated/results/'   --exclude='Draft_Results/results/'   --exclude='research/results/autoresearch/'   -czf -   Draft_Results   experiments   research   notes   autoresearch.md   autoresearch.sh   CHANGELOG.md   | $SSH "mkdir -p ${REMOTE_BASE} && cd ${REMOTE_BASE} && tar xzf -"

IFS=',' read -r -a N_ICL_ARR <<< "${N_ICLS}"
for N_ICL in "${N_ICL_ARR[@]}"; do
  echo "=== Running cross-model behavioral check: n_icl=${N_ICL} ==="
  $SSH "cd ${REMOTE_BASE} && HF_HUB_OFFLINE=${HF_HUB_OFFLINE_REMOTE} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_REMOTE} ${REMOTE_PY} experiments/cross_model_behavioral_check.py     --hf-id '${HF_ID}'     --model-label '${MODEL_LABEL}'     --pairs '${PAIRS}'     --seed '${SEED}'     --n-icl '${N_ICL}'     --n-select '${N_SELECT}'     --n-eval '${N_EVAL}'     --max-items '${MAX_ITEMS}'     --max-new-tokens '${MAX_NEW_TOKENS}'     --device cuda     --out-root 'research/results/autoresearch/${RESULTS_ROOT_NAME}'"
done

echo "=== Downloading results ==="
mkdir -p "${RESULTS_LOCAL}"
$SSH "cd ${REMOTE_BASE}/research/results/autoresearch/${RESULTS_ROOT_NAME} && tar cf - ."   | (cd "${RESULTS_LOCAL}" && tar xf -)

echo "=== Done ==="
