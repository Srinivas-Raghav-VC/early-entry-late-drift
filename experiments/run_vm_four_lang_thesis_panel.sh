#!/usr/bin/env bash
set -euo pipefail

# Run the thesis-scale 4-language behavioral panel across multiple seeds
# using the existing Loop 2 VM harness and then aggregate the scores.
#
# Required env:
#   VM_PASS=...
# Optional env:
#   PANEL_ROOT=research/results/autoresearch/four_lang_thesis_panel
#   PANEL_MODE=loop2_full
#   PANEL_SEEDS="42 11 101"
#   PANEL_MODELS="1b 4b"
#   PANEL_PAIRS="aksharantar_hin_latin aksharantar_tel_latin aksharantar_ben_latin aksharantar_tam_latin"
#   PANEL_NICLS="8 64"
#   VM_HOST=...
#   PAPER2_DEVICE=cuda

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PANEL_ROOT="${PANEL_ROOT:-research/results/autoresearch/four_lang_thesis_panel}"
PANEL_MODE="${PANEL_MODE:-loop2_full}"
PANEL_SEEDS="${PANEL_SEEDS:-42 11 101}"
PANEL_MODELS="${PANEL_MODELS:-1b 4b}"
PANEL_PAIRS="${PANEL_PAIRS:-aksharantar_hin_latin aksharantar_tel_latin aksharantar_ben_latin aksharantar_tam_latin}"
PANEL_NICLS="${PANEL_NICLS:-8 64}"

if [[ -z "${VM_PASS:-}" ]]; then
  echo "Set VM_PASS before running the thesis panel." >&2
  exit 2
fi

mkdir -p "$PANEL_ROOT"

echo "[four-lang-panel] root=$PANEL_ROOT"
echo "[four-lang-panel] mode=$PANEL_MODE"
echo "[four-lang-panel] seeds=$PANEL_SEEDS"
echo "[four-lang-panel] models=$PANEL_MODELS"
echo "[four-lang-panel] pairs=$PANEL_PAIRS"
echo "[four-lang-panel] nicls=$PANEL_NICLS"

score_args=()
for seed in $PANEL_SEEDS; do
  outdir="$PANEL_ROOT/seed${seed}"
  remote_rel="tmp/four_lang_thesis_panel/seed${seed}/raw"
  echo "[four-lang-panel] START seed=$seed outdir=$outdir"
  LOOP2_SEED="$seed" \
  LOOP2_MODELS="$PANEL_MODELS" \
  LOOP2_PAIRS="$PANEL_PAIRS" \
  LOOP2_NICLS="$PANEL_NICLS" \
  LOOP2_REMOTE_RESULTS_REL="$remote_rel" \
  bash autoresearch.sh "$PANEL_MODE" "$outdir"

  echo "[four-lang-panel] building manual audit packets for seed=$seed"
  find "$outdir/raw" -type f -name 'neutral_filler_recency_controls.json' | sort | while read -r artifact; do
    rel="${artifact#${outdir}/raw/}"
    audit_base="$outdir/manual_audits/${rel%.json}"
    mkdir -p "$(dirname "$audit_base")"
    python3 experiments/build_manual_audit_packet.py \
      --input "$artifact" \
      --out-json "${audit_base}.audit.json" \
      --out-md "${audit_base}.audit.md" \
      --max-examples 6 > /dev/null
  done

  score_args+=(--score "$outdir/score.json")
  echo "[four-lang-panel] DONE seed=$seed"
done

python3 experiments/aggregate_loop2_seed_scores.py \
  "${score_args[@]}" \
  --out "$PANEL_ROOT/seed_aggregate.json" \
  | tee "$PANEL_ROOT/aggregate.log"

echo "[four-lang-panel] aggregate -> $PANEL_ROOT/seed_aggregate.json"
