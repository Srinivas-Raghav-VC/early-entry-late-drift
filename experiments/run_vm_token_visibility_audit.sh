#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-research/results/autoresearch/token_visibility_v1}"
MODELS="${MODELS:-1b,4b}"
PAIRS="${PAIRS:-aksharantar_hin_latin,aksharantar_tel_latin}"
CONDITIONS="${CONDITIONS:-explicit_zs,icl8,icl64}"
SEED="${SEED:-42}"
MAX_ITEMS="${MAX_ITEMS:-30}"
VM_HOST="${VM_HOST:-reviewer@vm.example.invalid}"
VM_WORKDIR="${VM_WORKDIR:-/home/reviewer/Research/Honors}"
REMOTE_RESULTS_REL="${REMOTE_RESULTS_REL:-research/results/autoresearch/token_visibility_v1}"

if [[ -z "${VM_PASS:-}" ]]; then
  echo "Set VM_PASS before running this script." >&2
  exit 2
fi

vm_ssh() {
  sshpass -p "$VM_PASS" ssh \
    -o StrictHostKeyChecking=no \
    -o ConnectTimeout="${VM_CONNECT_TIMEOUT:-15}" \
    "$VM_HOST" "$@"
}

vm_sync_to_workdir() {
  local dest="$1"
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
    CHANGELOG.md | vm_ssh "mkdir -p '$dest' && tar xzf - -C '$dest'"
}

vm_fetch_dir() {
  local remote_dir="$1"
  local local_dir="$2"
  mkdir -p "$local_dir"
  vm_ssh "tar czf - -C '$remote_dir' ." | tar xzf - -C "$local_dir"
}

mkdir -p "$OUTDIR"

echo "[vis] checking VM"
vm_ssh "echo connected: \$(hostname)"

echo "[vis] syncing code"
vm_sync_to_workdir "$VM_WORKDIR"

echo "[vis] running token-visibility audit on VM"
vm_ssh "cd '$VM_WORKDIR' && MODELS='$MODELS' PAIRS='$PAIRS' CONDITIONS='$CONDITIONS' SEED='$SEED' MAX_ITEMS='$MAX_ITEMS' REMOTE_RESULTS_REL='$REMOTE_RESULTS_REL' bash -s" <<'REMOTE' | tee "$OUTDIR/vm_run.log"
set -euo pipefail
WORKDIR="$(pwd)"
PYTHON_BIN=""
for cand in \
  "$WORKDIR/.venv/bin/python" \
  "$WORKDIR/.venv-phase0a/bin/python" \
  "$HOME/Research/gemma-rescue-study/.venv/bin/python" \
  "$HOME/Research/gemma-rescue-study/.venv-phase0a/bin/python" \
  "$(command -v python3)"
do
  if [[ -n "$cand" && -x "$cand" ]]; then
    PYTHON_BIN="$cand"
    break
  fi
done
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[vis][remote] no usable python found" >&2
  exit 3
fi

echo "[vis][remote] workdir=$WORKDIR"
echo "[vis][remote] python=$PYTHON_BIN"
"$PYTHON_BIN" Draft_Results/paper2_fidelity_calibrated/run_phase0_token_visibility.py \
  --models "$MODELS" \
  --pairs "$PAIRS" \
  --conditions "$CONDITIONS" \
  --seed "$SEED" \
  --n-icl 64 \
  --n-select 300 \
  --n-eval 200 \
  --max-items "$MAX_ITEMS" \
  --external-only \
  --require-external-sources \
  --out-root "../$REMOTE_RESULTS_REL"
REMOTE

echo "[vis] fetching VM artifacts"
vm_fetch_dir "$VM_WORKDIR/$REMOTE_RESULTS_REL" "$OUTDIR/results"

echo "[vis] done -> $OUTDIR/results"
