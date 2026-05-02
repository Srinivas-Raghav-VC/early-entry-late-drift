#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-research/results/autoresearch/mech_screen_v1}"
TASKS="${TASKS:-4b:aksharantar_tel_latin,1b:aksharantar_hin_latin,4b:aksharantar_hin_latin}"
SEED="${SEED:-42}"
MAX_WORDS="${MAX_WORDS:-30}"
VM_HOST="${VM_HOST:-reviewer@vm.example.invalid}"
VM_WORKDIR="${VM_WORKDIR:-/home/reviewer/Research/Honors}"
REMOTE_RESULTS_REL="${REMOTE_RESULTS_REL:-research/results/autoresearch/mech_screen_v1}"
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

echo "[mech] checking VM"
vm_ssh "echo connected: \$(hostname)"

echo "[mech] syncing code"
vm_sync_to_workdir "$VM_WORKDIR"

echo "[mech] running script-space screening on VM"
vm_ssh "cd '$VM_WORKDIR' && TASKS='$TASKS' SEED='$SEED' MAX_WORDS='$MAX_WORDS' REMOTE_RESULTS_REL='$REMOTE_RESULTS_REL' PAPER2_DEVICE='$DEVICE' bash -s" <<'REMOTE' | tee "$OUTDIR/vm_run.log"
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
  echo "[mech][remote] no usable python found" >&2
  exit 3
fi

echo "[mech][remote] workdir=$WORKDIR"
echo "[mech][remote] python=$PYTHON_BIN"
"$PYTHON_BIN" Draft_Results/paper2_fidelity_calibrated/run_script_space_map.py \
  --tasks "$TASKS" \
  --seed "$SEED" \
  --n-icl 64 \
  --n-select 300 \
  --n-eval 200 \
  --max-words "$MAX_WORDS" \
  --device "${PAPER2_DEVICE:-cuda}" \
  --external-only \
  --require-external-sources \
  --out-root "../$REMOTE_RESULTS_REL"
REMOTE

echo "[mech] fetching VM artifacts"
vm_fetch_dir "$VM_WORKDIR/$REMOTE_RESULTS_REL" "$OUTDIR/results"
mkdir -p "$OUTDIR/phase0"
if vm_ssh "test -d '$VM_WORKDIR/Draft_Results/artifacts/phase0'"; then
  vm_fetch_dir "$VM_WORKDIR/Draft_Results/artifacts/phase0" "$OUTDIR/phase0"
fi

echo "[mech] done -> $OUTDIR/results"
