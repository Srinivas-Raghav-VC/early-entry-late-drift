#!/usr/bin/env bash
set -euo pipefail

VM_USER="${VM_USER:-reviewer}"
VM_HOST="${VM_HOST:-vm.example.invalid}"
VM_PASS="${VM_PASS:?Set VM_PASS}"
REMOTE_BASE="${REMOTE_BASE:-~/Research/Honors_prompt_sensitivity}"
MODELS="${MODELS:-1b,4b}"
PAIRS="${PAIRS:-aksharantar_hin_latin,aksharantar_tel_latin}"
N_ICLS="${N_ICLS:-8,64}"
PROMPT_VARIANTS="${PROMPT_VARIANTS:-canonical,compact,tagged}"
SEED="${SEED:-42}"
N_SELECT="${N_SELECT:-300}"
N_EVAL="${N_EVAL:-200}"
MAX_ITEMS="${MAX_ITEMS:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
RESULTS_ROOT_NAME="${RESULTS_ROOT_NAME:-prompt_sensitivity_v1}"
HF_HUB_OFFLINE_REMOTE="${HF_HUB_OFFLINE_REMOTE:-1}"
TRANSFORMERS_OFFLINE_REMOTE="${TRANSFORMERS_OFFLINE_REMOTE:-1}"
SSH="sshpass -p ${VM_PASS} ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10 ${VM_USER}@${VM_HOST}"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_LOCAL="${LOCAL_ROOT}/research/results/autoresearch/${RESULTS_ROOT_NAME}"
REMOTE_PY="~/miniconda3/envs/thesis_py311/bin/python"

cd "$LOCAL_ROOT"
echo "=== Syncing repo to VM ==="
tar   --exclude='__pycache__/'   --exclude='*.pyc'   --exclude='.git/'   --exclude='.cache/'   --exclude='.venv*/'   --exclude='Draft_Results/paper2_fidelity_calibrated/results/'   --exclude='Draft_Results/results/'   --exclude='research/results/autoresearch/'   -czf -   Draft_Results   experiments   research   notes   autoresearch.md   autoresearch.sh   CHANGELOG.md   | $SSH "mkdir -p ${REMOTE_BASE} && cd ${REMOTE_BASE} && tar xzf -"

echo "=== Running prompt sensitivity panel on VM ==="
$SSH "cd ${REMOTE_BASE} && HF_HUB_OFFLINE=${HF_HUB_OFFLINE_REMOTE} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_REMOTE} ${REMOTE_PY} experiments/prompt_sensitivity_check.py   --models '${MODELS}'   --pairs '${PAIRS}'   --n-icls '${N_ICLS}'   --prompt-variants '${PROMPT_VARIANTS}'   --seed '${SEED}'   --n-select '${N_SELECT}'   --n-eval '${N_EVAL}'   --max-items '${MAX_ITEMS}'   --max-new-tokens '${MAX_NEW_TOKENS}'   --device cuda   --external-only   --require-external-sources   --min-pool-size 500   --out-root 'research/results/autoresearch/${RESULTS_ROOT_NAME}'"

echo "=== Downloading results ==="
mkdir -p "${RESULTS_LOCAL}"
$SSH "cd ${REMOTE_BASE}/research/results/autoresearch/${RESULTS_ROOT_NAME} && tar cf - ."   | (cd "${RESULTS_LOCAL}" && tar xf -)

echo "=== Done ==="
