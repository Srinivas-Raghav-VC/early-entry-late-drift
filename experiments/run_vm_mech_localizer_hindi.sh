#!/usr/bin/env bash
# Launch the 1B Hindi layerwise early-routing localizer on the shared VM.
# Usage:  VM_PASS='dsc%0215' bash experiments/run_vm_mech_localizer_hindi.sh
set -euo pipefail

VM_USER="${VM_USER:-reviewer}"
VM_HOST="${VM_HOST:-vm.example.invalid}"
VM_PASS="${VM_PASS:?Set VM_PASS}"
REMOTE_BASE="~/Research/Honors"

SSH="sshpass -p ${VM_PASS} ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10 ${VM_USER}@${VM_HOST}"
SCP="sshpass -p ${VM_PASS} scp -o StrictHostKeyChecking=no"

LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_LOCAL="${LOCAL_ROOT}/research/results/autoresearch/mech_localizer_v1"

echo "=== Syncing code to VM ==="
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

REMOTE_PY="~/miniconda3/envs/thesis_py311/bin/python"

echo "=== Running localizer on VM ==="
$SSH "cd ${REMOTE_BASE} && ${REMOTE_PY} experiments/mech_localizer_hindi_routing.py \
    --model 1b \
    --pair aksharantar_hin_latin \
    --n-icl 64 \
    --seed 42 \
    --max-words 30 \
    --device cuda \
    --out ${REMOTE_BASE}/research/results/autoresearch/mech_localizer_v1/1b/aksharantar_hin_latin/nicl64"

echo "=== Downloading results ==="
mkdir -p "${RESULTS_LOCAL}/1b/aksharantar_hin_latin/nicl64"
$SSH "cd ${REMOTE_BASE}/research/results/autoresearch/mech_localizer_v1 && tar cf - ." \
  | (cd "${RESULTS_LOCAL}" && tar xf -)

echo "=== Done ==="
ls -la "${RESULTS_LOCAL}/1b/aksharantar_hin_latin/nicl64/" 2>/dev/null || true
