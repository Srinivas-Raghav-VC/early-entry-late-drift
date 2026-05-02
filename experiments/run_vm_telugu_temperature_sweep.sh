#!/usr/bin/env bash
set -euo pipefail

VM_USER="${VM_USER:-reviewer}"
VM_HOST="${VM_HOST:-vm.example.invalid}"
VM_PASS="${VM_PASS:?Set VM_PASS}"
REMOTE_BASE="${REMOTE_BASE:-~/Research/Honors_telugu_temperature}"
MODELS="${MODELS:-1b 4b}"
PAIR="${PAIR:-aksharantar_tel_latin}"
SEED="${SEED:-42}"
N_ICL="${N_ICL:-64}"
N_SELECT="${N_SELECT:-300}"
N_EVAL="${N_EVAL:-200}"
MAX_ITEMS="${MAX_ITEMS:-57}"
TEMPERATURES="${TEMPERATURES:-0.0,0.2,0.7}"
SAMPLES_PER_TEMP="${SAMPLES_PER_TEMP:-3}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
FUZZY_BANK_THRESHOLD="${FUZZY_BANK_THRESHOLD:-0.85}"
RESULTS_ROOT_NAME="${RESULTS_ROOT_NAME:-telugu_temperature_sweep_v1}"
SSH="sshpass -p ${VM_PASS} ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10 ${VM_USER}@${VM_HOST}"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_LOCAL="${LOCAL_ROOT}/research/results/autoresearch/${RESULTS_ROOT_NAME}"
REMOTE_PY="~/miniconda3/envs/thesis_py311/bin/python"
HF_HUB_OFFLINE_REMOTE="${HF_HUB_OFFLINE_REMOTE:-1}"
TRANSFORMERS_OFFLINE_REMOTE="${TRANSFORMERS_OFFLINE_REMOTE:-1}"

cd "$LOCAL_ROOT"

echo "=== Syncing repo to VM ==="
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

for model in ${MODELS}; do
  echo "=== Running Telugu temperature sweep: model=${model} ==="
  $SSH "cd ${REMOTE_BASE} && HF_HUB_OFFLINE=${HF_HUB_OFFLINE_REMOTE} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_REMOTE} ${REMOTE_PY} experiments/telugu_temperature_sweep.py \
    --model '${model}' \
    --pair '${PAIR}' \
    --seed '${SEED}' \
    --n-icl '${N_ICL}' \
    --n-select '${N_SELECT}' \
    --n-eval '${N_EVAL}' \
    --max-items '${MAX_ITEMS}' \
    --temperatures '${TEMPERATURES}' \
    --samples-per-temp '${SAMPLES_PER_TEMP}' \
    --top-p '${TOP_P}' \
    --max-new-tokens '${MAX_NEW_TOKENS}' \
    --fuzzy-bank-threshold '${FUZZY_BANK_THRESHOLD}' \
    --device cuda \
    --external-only \
    --require-external-sources \
    --min-pool-size 500 \
    --out-root 'research/results/autoresearch/${RESULTS_ROOT_NAME}'"
done

echo "=== Downloading results ==="
mkdir -p "${RESULTS_LOCAL}"
$SSH "cd ${REMOTE_BASE}/research/results/autoresearch/${RESULTS_ROOT_NAME} && tar cf - ." \
  | (cd "${RESULTS_LOCAL}" && tar xf -)

echo "=== Done ==="
find "${RESULTS_LOCAL}" -maxdepth 5 -name 'telugu_temperature_sweep.json' | sort
