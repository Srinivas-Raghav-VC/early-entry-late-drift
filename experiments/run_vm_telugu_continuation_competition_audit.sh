#!/usr/bin/env bash
set -euo pipefail

VM_USER="${VM_USER:-reviewer}"
VM_HOST="${VM_HOST:-vm.example.invalid}"
VM_PASS="${VM_PASS:?Set VM_PASS}"
REMOTE_BASE="${REMOTE_BASE:-~/Research/Honors_telugu_continuation}"
MODEL="${MODEL:-4b}"
PAIR="${PAIR:-aksharantar_tel_latin}"
SEED="${SEED:-42}"
N_ICL="${N_ICL:-64}"
N_SELECT="${N_SELECT:-300}"
N_EVAL="${N_EVAL:-200}"
MAX_ITEMS="${MAX_ITEMS:-30}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
SSH="sshpass -p ${VM_PASS} ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10 ${VM_USER}@${VM_HOST}"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_ROOT_NAME="${RESULTS_ROOT_NAME:-telugu_continuation_competition_v1}"
RESULTS_LOCAL="${LOCAL_ROOT}/research/results/autoresearch/${RESULTS_ROOT_NAME}"
REMOTE_RESULTS_REL="research/results/autoresearch/${RESULTS_ROOT_NAME}/${MODEL}/${PAIR}/seed${SEED}/nicl${N_ICL}"
REMOTE_PY="~/miniconda3/envs/thesis_py311/bin/python"

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
  autoresearch.md \
  autoresearch.sh \
  CHANGELOG.md \
  | $SSH "mkdir -p ${REMOTE_BASE} && cd ${REMOTE_BASE} && tar xzf -"

echo "=== Running Telugu continuation competition audit on VM ==="
$SSH "cd ${REMOTE_BASE} && ${REMOTE_PY} experiments/telugu_continuation_competition_audit.py \
  --model ${MODEL} \
  --pair ${PAIR} \
  --seed ${SEED} \
  --n-icl ${N_ICL} \
  --n-select ${N_SELECT} \
  --n-eval ${N_EVAL} \
  --max-items ${MAX_ITEMS} \
  --max-new-tokens ${MAX_NEW_TOKENS} \
  --device cuda \
  --external-only \
  --require-external-sources \
  --min-pool-size 500 \
  --out-root ${REMOTE_BASE}/research/results/autoresearch/${RESULTS_ROOT_NAME}"

echo "=== Downloading results ==="
mkdir -p "${RESULTS_LOCAL}/${MODEL}/${PAIR}/seed${SEED}/nicl${N_ICL}"
$SSH "cd ${REMOTE_BASE}/research/results/autoresearch/${RESULTS_ROOT_NAME} && tar cf - ." \
  | (cd "${RESULTS_LOCAL}" && tar xf -)

echo "=== Done ==="
ls -la "${RESULTS_LOCAL}/${MODEL}/${PAIR}/seed${SEED}/nicl${N_ICL}/" 2>/dev/null || true
