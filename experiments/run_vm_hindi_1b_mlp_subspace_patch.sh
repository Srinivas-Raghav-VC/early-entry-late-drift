#!/usr/bin/env bash
# Launch Hindi 1B L25 MLP subspace patch experiment on the shared VM.
# Usage:
#   VM_PASS='...' bash experiments/run_vm_hindi_1b_mlp_subspace_patch.sh
set -euo pipefail

VM_USER="${VM_USER:-reviewer}"
VM_HOST="${VM_HOST:-vm.example.invalid}"
VM_PASS="${VM_PASS:?Set VM_PASS}"
REMOTE_BASE="${REMOTE_BASE:-~/Research/Honors}"
MODEL="${MODEL:-1b}"
PAIR="${PAIR:-aksharantar_hin_latin}"
N_ICL="${N_ICL:-64}"
N_SELECT="${N_SELECT:-300}"
N_EVAL="${N_EVAL:-200}"
SELECTOR_MAX_ITEMS="${SELECTOR_MAX_ITEMS:-100}"
MAX_ITEMS="${MAX_ITEMS:-30}"
LAYER="${LAYER:-25}"
KS="${KS:-}"
SELECTOR_POLICIES="${SELECTOR_POLICIES:-abs}"
RANDOM_KS="${RANDOM_KS-4,16,64,256,512}"
N_RANDOM="${N_RANDOM:-2}"
NO_OP_K="${NO_OP_K:-64}"
SSH="sshpass -p ${VM_PASS} ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10 ${VM_USER}@${VM_HOST}"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_ROOT_NAME="${RESULTS_ROOT_NAME:-hindi_mlp_subspace_patch_v1}"
RESULTS_LOCAL="${LOCAL_ROOT}/research/results/autoresearch/${RESULTS_ROOT_NAME}"
REMOTE_RESULTS_REL="research/results/autoresearch/${RESULTS_ROOT_NAME}/${MODEL}/${PAIR}/nicl${N_ICL}"
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

echo "=== Running Hindi L25 MLP subspace patch on VM ==="
$SSH "cd ${REMOTE_BASE} && ${REMOTE_PY} experiments/hindi_1b_mlp_subspace_patch.py \
  --model ${MODEL} \
  --pair ${PAIR} \
  --seed 42 \
  --n-icl ${N_ICL} \
  --n-select ${N_SELECT} \
  --n-eval ${N_EVAL} \
  --selector-max-items ${SELECTOR_MAX_ITEMS} \
  --max-items ${MAX_ITEMS} \
  --layer ${LAYER} \
  --ks '${KS}' \
  --selector-policies '${SELECTOR_POLICIES}' \
  --random-ks '${RANDOM_KS}' \
  --n-random ${N_RANDOM} \
  --no-op-k ${NO_OP_K} \
  --device cuda \
  --out ${REMOTE_BASE}/${REMOTE_RESULTS_REL}"

echo "=== Downloading results ==="
mkdir -p "${RESULTS_LOCAL}/${MODEL}/${PAIR}/nicl${N_ICL}"
$SSH "cd ${REMOTE_BASE}/research/results/autoresearch/${RESULTS_ROOT_NAME} && tar cf - ." \
  | (cd "${RESULTS_LOCAL}" && tar xf -)

echo "=== Done ==="
ls -la "${RESULTS_LOCAL}/${MODEL}/${PAIR}/nicl${N_ICL}/" 2>/dev/null || true
