#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

QUEUE_ROOT="${QUEUE_ROOT:-research/results/autoresearch/phase23_queue_v1}"
mkdir -p "$QUEUE_ROOT"
LOG_PATH="$QUEUE_ROOT/queue.log"

{
  echo "[queue] started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[queue] phase 1: hindi patch safety audit"
  bash experiments/run_vm_hindi_1b_patch_safety_audit.sh

  echo "[queue] phase 2: fixed-split k-shot regime sweep"
  bash experiments/run_vm_kshot_regime_sweep.sh

  echo "[queue] phase 3: telugu temperature sweep"
  bash experiments/run_vm_telugu_temperature_sweep.sh

  echo "[queue] phase 4: telugu writer-head probe"
  bash experiments/run_vm_telugu_writer_head_probe.sh

  echo "[queue] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$LOG_PATH"
