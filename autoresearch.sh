#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
OUTDIR="${2:-}"
FORCE_FLAG="${FORCE_FLAG:---force}"

loop1_default_outdir() {
  local mode="$1"
  printf 'research/results/autoresearch/loop1_cross_scale_anchor/%s' "$mode"
}

run_loop1() {
  local mode="$1"
  local outdir="${2:-$(loop1_default_outdir "$mode")}"
  local task_ids="premise_gate__270m__aksharantar_hin_latin,premise_gate__270m__aksharantar_tel_latin,premise_gate__1b__aksharantar_hin_latin,premise_gate__1b__aksharantar_tel_latin,premise_gate__4b__aksharantar_hin_latin,premise_gate__4b__aksharantar_tel_latin"
  local volume_name="gemma-multiscale-results"
  local remote_volume_path="/"

  mkdir -p "$outdir"

  echo "[loop1] verifying multiscale suite wiring"
  python3 -m Draft_Results.paper2_fidelity_calibrated.multiscale_modal_suite.verify_suite \
    --out "$outdir/verify_suite.json"

  echo "[loop1] launching Modal premise-gate tasks ($mode)"
  local -a modal_args=(
    run Draft_Results/paper2_fidelity_calibrated/multiscale_modal_suite/modal_app.py
    --task-ids "$task_ids"
    --wait
    "$FORCE_FLAG"
  )
  if [[ "$mode" == "smoke" ]]; then
    modal_args+=(--smoke)
  elif [[ "$mode" != "full" ]]; then
    echo "Unknown Loop 1 mode: $mode (expected smoke or full)" >&2
    exit 2
  fi

  local tmp_modal_log
  tmp_modal_log="$(mktemp /tmp/loop1-modal-run-XXXXXX.log)"
  modal "${modal_args[@]}" | tee "$tmp_modal_log"
  mv "$tmp_modal_log" "$outdir/modal_run.log"

  echo "[loop1] downloading Modal artifacts"
  mkdir -p "$outdir/volume"
  modal volume get "$volume_name" "$remote_volume_path" "$outdir/volume" --force

  echo "[loop1] scoring cross-scale premise gate"
  python3 experiments/score_cross_scale_anchor.py \
    --results-root "$outdir/volume" \
    --out "$outdir/score.json" | tee "$outdir/score.log"

  echo "[loop1] done -> $outdir/score.json"
}

loop2_default_outdir() {
  local mode="$1"
  printf 'research/results/autoresearch/loop2_vm_controls/%s' "$mode"
}

require_vm_secret() {
  if [[ -z "${VM_PASS:-}" ]]; then
    echo "Set VM_PASS in the environment before running Loop 2 on the VM." >&2
    exit 2
  fi
}

vm_ssh() {
  sshpass -p "$VM_PASS" ssh \
    -o StrictHostKeyChecking=no \
    -o ConnectTimeout="${VM_CONNECT_TIMEOUT:-15}" \
    -o ServerAliveInterval="${VM_SERVER_ALIVE_INTERVAL:-30}" \
    -o ServerAliveCountMax="${VM_SERVER_ALIVE_COUNT_MAX:-10}" \
    "${VM_HOST:-reviewer@vm.example.invalid}" "$@"
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

run_loop2() {
  local mode="$1"
  local outdir="${2:-$(loop2_default_outdir "$mode")}"
  local vm_workdir="${VM_WORKDIR:-/home/reviewer/Research/Honors}"
  local remote_results_rel="${LOOP2_REMOTE_RESULTS_REL:-$outdir/raw}"
  local loop2_seed="${LOOP2_SEED:-42}"
  local loop2_models="${LOOP2_MODELS:-1b 4b}"
  local loop2_pairs="${LOOP2_PAIRS:-aksharantar_hin_latin aksharantar_tel_latin}"
  local loop2_nicls="${LOOP2_NICLS:-8 64}"
  local max_items n_eval

  case "$mode" in
    loop2_smoke)
      max_items="8"
      n_eval="24"
      ;;
    loop2_full)
      max_items="30"
      n_eval="50"
      ;;
    *)
      echo "Unknown Loop 2 mode: $mode (expected loop2_smoke or loop2_full)" >&2
      exit 2
      ;;
  esac

  require_vm_secret
  mkdir -p "$outdir"

  echo "[loop2] checking VM reachability"
  vm_ssh "echo connected: \\$(hostname)"

  echo "[loop2] preparing VM workspace at $vm_workdir"
  vm_ssh "mkdir -p $vm_workdir"

  echo "[loop2] syncing code to VM"
  vm_sync_to_workdir "$vm_workdir"

  echo "[loop2] running helpful-vs-control panel on VM ($mode, seed=$loop2_seed)"
  vm_ssh "cd $vm_workdir && LOOP2_REMOTE_RESULTS_REL='$remote_results_rel' LOOP2_MAX_ITEMS='$max_items' LOOP2_N_EVAL='$n_eval' LOOP2_SEED='$loop2_seed' LOOP2_MODELS='$loop2_models' LOOP2_PAIRS='$loop2_pairs' LOOP2_NICLS='$loop2_nicls' PAPER2_DEVICE='${PAPER2_DEVICE:-cuda}' bash -s" <<'REMOTE' | tee "$outdir/vm_run.log"
set -euo pipefail
WORKDIR="$(pwd)"
DEVICE="${PAPER2_DEVICE:-cuda}"
REMOTE_RESULTS_REL="${LOOP2_REMOTE_RESULTS_REL}"
MAX_ITEMS="${LOOP2_MAX_ITEMS}"
N_EVAL="${LOOP2_N_EVAL}"
SEED="${LOOP2_SEED:-42}"
MODELS="${LOOP2_MODELS:-1b 4b}"
PAIRS="${LOOP2_PAIRS:-aksharantar_hin_latin aksharantar_tel_latin}"
NICLS="${LOOP2_NICLS:-8 64}"

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
  echo "[loop2][remote] no usable python found" >&2
  exit 3
fi

echo "[loop2][remote] workdir=$WORKDIR"
echo "[loop2][remote] python=$PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import importlib
for name in ('torch', 'transformers', 'numpy'):
    importlib.import_module(name)
print('python_import_health: ok')
PY

mkdir -p "$REMOTE_RESULTS_REL"
for model in $MODELS; do
  for pair in $PAIRS; do
    for n_icl in $NICLS; do
      out_dir="$REMOTE_RESULTS_REL/$model/$pair/nicl${n_icl}"
      mkdir -p "$out_dir"
      echo "[loop2][remote] START model=$model pair=$pair n_icl=$n_icl seed=$SEED"
      "$PYTHON_BIN" Draft_Results/paper2_fidelity_calibrated/run_neutral_filler_recency_controls.py \
        --model "$model" \
        --pair "$pair" \
        --device "$DEVICE" \
        --seed "$SEED" \
        --n-icl "$n_icl" \
        --n-select 300 \
        --n-eval "$N_EVAL" \
        --max-items "$MAX_ITEMS" \
        --external-only \
        --require-external-sources \
        --out "$out_dir"
      echo "[loop2][remote] DONE  model=$model pair=$pair n_icl=$n_icl"
    done
  done
done
REMOTE

  echo "[loop2] downloading Loop 2 artifacts from VM"
  mkdir -p "$outdir/raw"
  vm_fetch_dir "$vm_workdir/$remote_results_rel" "$outdir/raw"

  echo "[loop2] scoring helpful-vs-control panel"
  python3 experiments/score_loop2_controls.py \
    --results-root "$outdir/raw" \
    --models "$loop2_models" \
    --pairs "$loop2_pairs" \
    --nicls "$loop2_nicls" \
    --out "$outdir/score.json" | tee "$outdir/score.log"

  echo "[loop2] done -> $outdir/score.json"
}

export LOOP2_REMOTE_RESULTS_REL="${LOOP2_REMOTE_RESULTS_REL:-}"
export LOOP2_MAX_ITEMS="${LOOP2_MAX_ITEMS:-}"
export LOOP2_N_EVAL="${LOOP2_N_EVAL:-}"
export LOOP2_SEED="${LOOP2_SEED:-42}"
export LOOP2_MODELS="${LOOP2_MODELS:-1b 4b}"
export LOOP2_PAIRS="${LOOP2_PAIRS:-aksharantar_hin_latin aksharantar_tel_latin}"
export LOOP2_NICLS="${LOOP2_NICLS:-8 64}"

run_telugu_mlp_crossover_modal() {
  local mode="$1"
  local max_items select_max_items k_grid alpha_grid n_random results_name local_out
  if [[ "$mode" == "telugu_mlp_crossover_smoke" ]]; then
    max_items="${CROSSOVER_MAX_ITEMS:-10}"
    select_max_items="${CROSSOVER_SELECT_MAX_ITEMS:-10}"
    k_grid="${CROSSOVER_K_GRID:-2,4}"
    alpha_grid="${CROSSOVER_ALPHA_GRID:-0.5,1.0}"
    n_random="${CROSSOVER_N_RANDOM:-1}"
    results_name="${CROSSOVER_RESULTS_NAME:-telugu_mlp_channel_crossover_smoke_v1}"
  else
    max_items="${CROSSOVER_MAX_ITEMS:-200}"
    select_max_items="${CROSSOVER_SELECT_MAX_ITEMS:-100}"
    k_grid="${CROSSOVER_K_GRID:-2,4,8,16,32,64,128}"
    alpha_grid="${CROSSOVER_ALPHA_GRID:-0.25,0.5,1.0,1.5,2.0}"
    n_random="${CROSSOVER_N_RANDOM:-3}"
    results_name="${CROSSOVER_RESULTS_NAME:-telugu_mlp_channel_crossover_v1}"
  fi
  local_out="research/results/autoresearch"
  echo "[crossover] launching Modal Telugu MLP-channel crossover ($mode)"
  local -a modal_run_args=(run)
  if [[ "${CROSSOVER_MODAL_DETACH:-0}" == "1" ]]; then
    modal_run_args+=(--detach)
  fi
  modal_run_args+=(
    experiments/modal_telugu_mlp_channel_crossover.py
    --max-items "$max_items"
    --select-max-items "$select_max_items"
    --k-grid "$k_grid"
    --alpha-grid "$alpha_grid"
    --n-random "$n_random"
    --results-name "$results_name"
  )
  modal "${modal_run_args[@]}"
  if [[ "${CROSSOVER_MODAL_DETACH:-0}" == "1" ]]; then
    echo "[crossover] detached launch submitted; fetch later with telugu_mlp_crossover_fetch"
    echo "METRIC crossover_run_failed=0"
    return 0
  fi
  fetch_telugu_mlp_crossover_modal "$results_name"
}

fetch_telugu_mlp_crossover_modal() {
  local results_name="${1:-${CROSSOVER_RESULTS_NAME:-telugu_mlp_channel_crossover_v1}}"
  local local_out="research/results/autoresearch"
  local expected="$local_out/$results_name/1b/aksharantar_tel_latin/seed42/nicl64/telugu_1b_mlp_channel_crossover.json"
  echo "[crossover] downloading Modal artifacts -> $local_out"
  mkdir -p "$local_out"
  modal volume get crc-workshop-artifacts "/${results_name}" "$local_out" --force
  if [[ -s "$expected" ]]; then
    echo "[crossover] fetched $expected"
    echo "METRIC crossover_run_failed=0"
  else
    echo "[crossover] expected artifact is not present yet: $expected" >&2
    echo "METRIC crossover_run_failed=1"
  fi
}

case "$MODE" in
  readiness_audit)
    python3 experiments/submission_readiness_audit.py
    ;;
  workshop_risk_audit)
    python3 experiments/workshop_risk_audit.py
    ;;
  cross_model_synthesis_audit)
    python3 experiments/cross_model_synthesis_audit.py
    ;;
  anonymous_repo_audit)
    python3 experiments/anonymous_repo_audit.py
    ;;
  package_submission_artifacts)
    python3 experiments/package_submission_artifacts.py
    ;;
  package_complearn_submission_artifacts)
    python3 experiments/package_complearn_submission_artifacts.py
    ;;
  package_clean_supplement)
    python3 experiments/package_clean_review_supplement.py
    ;;
  final_submission_audit)
    python3 experiments/final_submission_audit.py
    ;;
  public_archive_surface_audit)
    python3 experiments/public_archive_surface_audit.py
    ;;
  preupload_checklist_audit)
    python3 experiments/preupload_checklist_audit.py
    ;;
  reviewer_feedback_audit)
    python3 experiments/reviewer_feedback_audit.py
    ;;
  reproduction_archive_reference_audit)
    python3 experiments/reproduction_archive_reference_audit.py
    ;;
  stage_axis_render_check)
    python3 experiments/render_stage_axis_map.py
    (cd "Paper Template and Paper/Paper/icml2026" && tectonic gemma_1b_icl_paper_submission_8plus2_icml2026_final.tex >/tmp/tectonic_stage_axis.log 2>&1)
    echo "METRIC figure_render_failed=0"
    ;;
  channel_characterization_audit)
    python3 experiments/channel_characterization_audit.py
    ;;
  title_framing_audit)
    flags=0
    if grep -R -q "Copy, Retrieve, or Compose" README.md "Paper Template and Paper/Paper/icml2026/gemma_1b_icl_paper_submission_8plus2_icml2026_final.tex" 2>/dev/null; then
      flags=$((flags + 1))
    fi
    if ! grep -q "Early Entry, Late Drift" "Paper Template and Paper/Paper/icml2026/gemma_1b_icl_paper_submission_8plus2_icml2026_final.tex"; then
      flags=$((flags + 1))
    fi
    echo "METRIC title_overclaim_flags=${flags}"
    ;;
  framing_calibration_audit)
    flags=0
    paper="Paper Template and Paper/Paper/icml2026/gemma_1b_icl_paper_submission_8plus2_icml2026_final.tex"
    for phrase in "Mechanistic Contrast" "different underlying mechanisms" "full mechanistic case study" "mechanistic case studies" "localizing bounded mechanisms"; do
      if grep -q "$phrase" "$paper"; then
        flags=$((flags + 1))
      fi
    done
    if ! grep -q "Causal Diagnostics and Interventions" "$paper"; then
      flags=$((flags + 1))
    fi
    echo "METRIC framing_overclaim_flags=${flags}"
    ;;
  telugu_mlp_crossover_smoke|telugu_mlp_crossover_full)
    run_telugu_mlp_crossover_modal "$MODE"
    ;;
  telugu_mlp_crossover_fetch)
    fetch_telugu_mlp_crossover_modal "${CROSSOVER_RESULTS_NAME:-telugu_mlp_channel_crossover_v1}"
    ;;
  smoke|full)
    run_loop1 "$MODE" "$OUTDIR"
    ;;
  loop2_smoke|loop2_full)
    if [[ -z "${LOOP2_REMOTE_RESULTS_REL:-}" ]]; then
      export LOOP2_REMOTE_RESULTS_REL="$(loop2_default_outdir "$MODE")/raw"
    fi
    if [[ -z "${LOOP2_MAX_ITEMS:-}" || -z "${LOOP2_N_EVAL:-}" ]]; then
      if [[ "$MODE" == "loop2_smoke" ]]; then
        export LOOP2_MAX_ITEMS="8"
        export LOOP2_N_EVAL="24"
      else
        export LOOP2_MAX_ITEMS="30"
        export LOOP2_N_EVAL="50"
      fi
    fi
    run_loop2 "$MODE" "$OUTDIR"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Expected one of: readiness_audit, workshop_risk_audit, cross_model_synthesis_audit, anonymous_repo_audit, package_submission_artifacts, package_complearn_submission_artifacts, package_clean_supplement, final_submission_audit, public_archive_surface_audit, preupload_checklist_audit, reviewer_feedback_audit, reproduction_archive_reference_audit, stage_axis_render_check, channel_characterization_audit, title_framing_audit, framing_calibration_audit, telugu_mlp_crossover_smoke, telugu_mlp_crossover_full, telugu_mlp_crossover_fetch, smoke, full, loop2_smoke, loop2_full" >&2
    exit 2
    ;;
esac
