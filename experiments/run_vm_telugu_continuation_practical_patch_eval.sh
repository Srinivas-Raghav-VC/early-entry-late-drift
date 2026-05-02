#!/usr/bin/env bash
set -euo pipefail

VM_USER="${VM_USER:-reviewer}"
VM_HOST="${VM_HOST:-vm.example.invalid}"
VM_PASS="${VM_PASS:?Set VM_PASS}"
REMOTE_BASE="${REMOTE_BASE:-~/Research/Honors_telugu_patch}"
MODEL="${MODEL:-1b}"
PAIR="${PAIR:-aksharantar_tel_latin}"
SEED="${SEED:-42}"
N_ICL="${N_ICL:-64}"
N_SELECT="${N_SELECT:-300}"
N_EVAL="${N_EVAL:-200}"
SELECT_MAX_ITEMS="${SELECT_MAX_ITEMS:-100}"
MAX_ITEMS="${MAX_ITEMS:-60}"
RECIPIENT="${RECIPIENT:-}"
DONOR="${DONOR:-}"
SITE="${SITE:-}"
OFFBAND_SITE="${OFFBAND_SITE:-}"
ALPHA_GRID="${ALPHA_GRID:-0.25,0.5,0.75,1.0,1.25,1.5,2.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
RESULTS_ROOT_NAME="${RESULTS_ROOT_NAME:-telugu_continuation_practical_patch_eval_v1}"
SSH="sshpass -p ${VM_PASS} ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10 ${VM_USER}@${VM_HOST}"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_LOCAL="${LOCAL_ROOT}/research/results/autoresearch/${RESULTS_ROOT_NAME}"
REMOTE_PY="~/miniconda3/envs/thesis_py311/bin/python"
HF_HUB_OFFLINE_REMOTE="${HF_HUB_OFFLINE_REMOTE:-1}"
TRANSFORMERS_OFFLINE_REMOTE="${TRANSFORMERS_OFFLINE_REMOTE:-1}"

echo "=== Syncing repo to VM ==="
cd "$LOCAL_ROOT"
tar \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.git/' \
  --exclude='.cache/' \
  --exclude='.venv*/' \
  --exclude='Draft_Results/paper2_fidelity_calibrated/results/' \
  --exclude='Draft_Results/results/' \
  --exclude='research/results/autoresearch/' \
  -czf - \
  Draft_Results \
  experiments \
  research \
  notes \
  autoresearch.md \
  autoresearch.sh \
  CHANGELOG.md \
  | $SSH "mkdir -p ${REMOTE_BASE} && cd ${REMOTE_BASE} && tar xzf -"

echo "=== Running Telugu practical patch eval on VM ==="
$SSH "cd ${REMOTE_BASE} && HF_HUB_OFFLINE=${HF_HUB_OFFLINE_REMOTE} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_REMOTE} ${REMOTE_PY} experiments/telugu_continuation_practical_patch_eval.py \
  --model '${MODEL}' \
  --pair '${PAIR}' \
  --seed '${SEED}' \
  --n-icl '${N_ICL}' \
  --n-select '${N_SELECT}' \
  --n-eval '${N_EVAL}' \
  --select-max-items '${SELECT_MAX_ITEMS}' \
  --max-items '${MAX_ITEMS}' \
  --recipient '${RECIPIENT}' \
  --donor '${DONOR}' \
  --site '${SITE}' \
  --offband-site '${OFFBAND_SITE}' \
  --alpha-grid '${ALPHA_GRID}' \
  --max-new-tokens '${MAX_NEW_TOKENS}' \
  --device cuda \
  --external-only \
  --require-external-sources \
  --min-pool-size 500 \
  --out-root 'research/results/autoresearch/${RESULTS_ROOT_NAME}/${MODEL}/${PAIR}/seed${SEED}/nicl${N_ICL}'"

echo "=== Downloading results ==="
mkdir -p "${RESULTS_LOCAL}/${MODEL}/${PAIR}/seed${SEED}/nicl${N_ICL}"
$SSH "cd ${REMOTE_BASE}/research/results/autoresearch/${RESULTS_ROOT_NAME} && tar cf - ." \
  | (cd "${RESULTS_LOCAL}" && tar xf -)

echo "=== Done ==="
ls -la "${RESULTS_LOCAL}/${MODEL}/${PAIR}/seed${SEED}/nicl${N_ICL}/" 2>/dev/null || true
