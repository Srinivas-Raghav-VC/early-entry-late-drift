#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PANEL_ROOT="${PANEL_ROOT:-research/results/autoresearch/four_lang_thesis_panel}"
PANEL_MODELS="${PANEL_MODELS:-1b 4b}"
PANEL_PAIRS="${PANEL_PAIRS:-aksharantar_hin_latin aksharantar_tel_latin aksharantar_ben_latin aksharantar_tam_latin}"
PANEL_NICLS="${PANEL_NICLS:-8 64}"
REST_SEEDS="${REST_SEEDS:-11 101}"

if [[ -z "${VM_PASS:-}" ]]; then
  echo "Set VM_PASS before recovery." >&2
  exit 2
fi

seed42_out="$PANEL_ROOT/seed42"
mkdir -p "$PANEL_ROOT"

echo "[recover-four-lang] recovering missing seed42 cells"
LOOP2_SEED=42 \
LOOP2_MODELS='4b' \
LOOP2_PAIRS='aksharantar_tam_latin' \
LOOP2_NICLS='8 64' \
LOOP2_REMOTE_RESULTS_REL='tmp/four_lang_thesis_panel/seed42/raw' \
VM_SERVER_ALIVE_INTERVAL='30' \
VM_SERVER_ALIVE_COUNT_MAX='10' \
bash autoresearch.sh loop2_full "$seed42_out"

echo "[recover-four-lang] rescoring full seed42 panel"
python3 experiments/score_loop2_controls.py \
  --results-root "$seed42_out/raw" \
  --models "$PANEL_MODELS" \
  --pairs "$PANEL_PAIRS" \
  --nicls "$PANEL_NICLS" \
  --out "$seed42_out/score.json" | tee "$seed42_out/score.log"

echo "[recover-four-lang] rebuilding manual audit packets for seed42"
find "$seed42_out/raw" -type f -name 'neutral_filler_recency_controls.json' | sort | while read -r artifact; do
  rel="${artifact#${seed42_out}/raw/}"
  audit_base="$seed42_out/manual_audits/${rel%.json}"
  mkdir -p "$(dirname "$audit_base")"
  python3 experiments/build_manual_audit_packet.py \
    --input "$artifact" \
    --out-json "${audit_base}.audit.json" \
    --out-md "${audit_base}.audit.md" \
    --max-examples 6 > /dev/null
done

echo "[recover-four-lang] continuing remaining seeds: $REST_SEEDS"
PANEL_SEEDS="$REST_SEEDS" \
VM_SERVER_ALIVE_INTERVAL='30' \
VM_SERVER_ALIVE_COUNT_MAX='10' \
bash experiments/run_vm_four_lang_thesis_panel.sh

echo "[recover-four-lang] aggregating all three seeds"
python3 experiments/aggregate_loop2_seed_scores.py \
  --score "$PANEL_ROOT/seed42/score.json" \
  --score "$PANEL_ROOT/seed11/score.json" \
  --score "$PANEL_ROOT/seed101/score.json" \
  --out "$PANEL_ROOT/seed_aggregate.json" | tee "$PANEL_ROOT/aggregate_all_seeds.log"

echo "[recover-four-lang] done -> $PANEL_ROOT/seed_aggregate.json"
