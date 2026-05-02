#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-research/results/autoresearch/first_token_competition_v1}"
TASKS="${TASKS:-1b:aksharantar_hin_latin:64,1b:aksharantar_tel_latin:64,4b:aksharantar_tel_latin:64}"
SEED="${SEED:-42}"
MAX_ITEMS="${MAX_ITEMS:-30}"
VM_HOST="${VM_HOST:-reviewer@vm.example.invalid}"
VM_WORKDIR="${VM_WORKDIR:-/home/reviewer/Research/Honors}"
REMOTE_RESULTS_REL="${REMOTE_RESULTS_REL:-research/results/autoresearch/first_token_competition_v1}"
DEVICE="${PAPER2_DEVICE:-cuda}"

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

echo "[ftok] checking VM"
vm_ssh "echo connected: \$(hostname)"

echo "[ftok] syncing code"
vm_sync_to_workdir "$VM_WORKDIR"

echo "[ftok] running first-token competition audit on VM"
vm_ssh "cd '$VM_WORKDIR' && TASKS='$TASKS' SEED='$SEED' MAX_ITEMS='$MAX_ITEMS' REMOTE_RESULTS_REL='$REMOTE_RESULTS_REL' PAPER2_DEVICE='$DEVICE' bash -s" <<'REMOTE' | tee "$OUTDIR/vm_run.log"
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
  echo "[ftok][remote] no usable python found" >&2
  exit 3
fi

echo "[ftok][remote] workdir=$WORKDIR"
echo "[ftok][remote] python=$PYTHON_BIN"
"$PYTHON_BIN" experiments/first_token_competition_audit.py \
  --tasks "$TASKS" \
  --seed "$SEED" \
  --n-select 300 \
  --n-eval 200 \
  --max-items "$MAX_ITEMS" \
  --device "${PAPER2_DEVICE:-cuda}" \
  --external-only \
  --require-external-sources \
  --out-root "$REMOTE_RESULTS_REL"
REMOTE

echo "[ftok] fetching VM artifacts"
vm_fetch_dir "$VM_WORKDIR/$REMOTE_RESULTS_REL" "$OUTDIR/results"

echo "[ftok] done -> $OUTDIR/results"
