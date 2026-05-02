#!/usr/bin/env bash
set -euo pipefail

# Run the existing Telugu continuation localizer across multiple panel sizes
# to check whether the main late-band localization survives beyond N=30.
#
# Required env:
#   VM_PASS=...
# Optional env:
#   MODELS="1b 4b"
#   MAX_ITEMS_LIST="30 60 100"
#   RESULTS_ROOT_BASE="telugu_continuation_localizer_varyingN_v1"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${VM_PASS:-}" ]]; then
  echo "Set VM_PASS before running varying-N Telugu localizers." >&2
  exit 2
fi

MODELS="${MODELS:-1b 4b}"
MAX_ITEMS_LIST="${MAX_ITEMS_LIST:-30 60 100}"
PAIR="${PAIR:-aksharantar_tel_latin}"
SEED="${SEED:-42}"
N_ICL="${N_ICL:-64}"
RESULTS_ROOT_BASE="${RESULTS_ROOT_BASE:-telugu_continuation_localizer_varyingN_v1}"

for model in $MODELS; do
  for n in $MAX_ITEMS_LIST; do
    root="${RESULTS_ROOT_BASE}/max${n}"
    echo "[varying-n-localizer] model=$model max_items=$n root=$root"
    VM_PASS="$VM_PASS" \
    MODEL="$model" \
    PAIR="$PAIR" \
    SEED="$SEED" \
    N_ICL="$N_ICL" \
    MAX_ITEMS="$n" \
    RESULTS_ROOT_NAME="$root" \
    bash experiments/run_vm_telugu_continuation_localizer.sh
  done
 done
