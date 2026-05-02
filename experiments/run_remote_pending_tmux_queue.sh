#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${HOME}/Research/Honors_hindi_patch"
PYTHON_BIN="${HOME}/miniconda3/envs/thesis_py311/bin/python"
STAMP="${STAMP:-$(date +%F_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/research/results/autoresearch/review_followup_queue_v1"
LOG_PATH="${LOG_DIR}/remote_tmux_pending_${STAMP}.log"
mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

have_all() {
  local path
  for path in "$@"; do
    if [[ ! -f "${path}" ]]; then
      return 1
    fi
  done
  return 0
}

run_if_missing() {
  local label="$1"
  shift
  local -a expected=()
  while [[ "$1" != "--" ]]; do
    expected+=("$1")
    shift
  done
  shift

  if have_all "${expected[@]}"; then
    echo "=== Skip: ${label} already complete ==="
    return 0
  fi

  echo "=== Run: ${label} ==="
  "$@"
}

QWEN15_N8=(
  "${REPO_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/qwen2.5-1.5b/aksharantar_hin_latin/seed42/nicl8/cross_model_behavioral.json"
  "${REPO_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/qwen2.5-1.5b/aksharantar_tel_latin/seed42/nicl8/cross_model_behavioral.json"
)
QWEN15_N64=(
  "${REPO_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/qwen2.5-1.5b/aksharantar_hin_latin/seed42/nicl64/cross_model_behavioral.json"
  "${REPO_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/qwen2.5-1.5b/aksharantar_tel_latin/seed42/nicl64/cross_model_behavioral.json"
)
PROMPT_COMPOSITION=(
  "${REPO_ROOT}/research/results/autoresearch/prompt_composition_ablation_v1/1b/aksharantar_tel_latin/seed42/nicl64/prompt_composition_ablation.json"
  "${REPO_ROOT}/research/results/autoresearch/prompt_composition_ablation_v1/4b/aksharantar_tel_latin/seed42/nicl64/prompt_composition_ablation.json"
)
LLAMA1B_N8=(
  "${REPO_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/llama3.2-1b/aksharantar_hin_latin/seed42/nicl8/cross_model_behavioral.json"
  "${REPO_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/llama3.2-1b/aksharantar_tel_latin/seed42/nicl8/cross_model_behavioral.json"
)
LLAMA1B_N64=(
  "${REPO_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/llama3.2-1b/aksharantar_hin_latin/seed42/nicl64/cross_model_behavioral.json"
  "${REPO_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/llama3.2-1b/aksharantar_tel_latin/seed42/nicl64/cross_model_behavioral.json"
)
LLAMA3B_N8=(
  "${REPO_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/llama3.2-3b/aksharantar_hin_latin/seed42/nicl8/cross_model_behavioral.json"
  "${REPO_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/llama3.2-3b/aksharantar_tel_latin/seed42/nicl8/cross_model_behavioral.json"
)
LLAMA3B_N64=(
  "${REPO_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/llama3.2-3b/aksharantar_hin_latin/seed42/nicl64/cross_model_behavioral.json"
  "${REPO_ROOT}/research/results/autoresearch/cross_model_behavioral_v1/llama3.2-3b/aksharantar_tel_latin/seed42/nicl64/cross_model_behavioral.json"
)

{
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] remote pending tmux queue start"

  run_if_missing     "Qwen 2.5 1.5B cross-model n_icl=8"     "${QWEN15_N8[@]}" --     env HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0     "${PYTHON_BIN}" experiments/cross_model_behavioral_check.py       --hf-id 'Qwen/Qwen2.5-1.5B-Instruct'       --model-label 'qwen2.5-1.5b'       --pairs 'aksharantar_hin_latin,aksharantar_tel_latin'       --seed 42       --n-icl 8       --n-select 300       --n-eval 200       --max-items 200       --max-new-tokens 16       --device cuda       --out-root 'research/results/autoresearch/cross_model_behavioral_v1'

  run_if_missing     "Qwen 2.5 1.5B cross-model n_icl=64"     "${QWEN15_N64[@]}" --     env HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0     "${PYTHON_BIN}" experiments/cross_model_behavioral_check.py       --hf-id 'Qwen/Qwen2.5-1.5B-Instruct'       --model-label 'qwen2.5-1.5b'       --pairs 'aksharantar_hin_latin,aksharantar_tel_latin'       --seed 42       --n-icl 64       --n-select 300       --n-eval 200       --max-items 200       --max-new-tokens 16       --device cuda       --out-root 'research/results/autoresearch/cross_model_behavioral_v1'

  run_if_missing     "prompt composition ablation"     "${PROMPT_COMPOSITION[@]}" --     env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     "${PYTHON_BIN}" experiments/prompt_composition_ablation.py       --tasks '1b:aksharantar_tel_latin:64,4b:aksharantar_tel_latin:64'       --seed 42       --n-select 300       --n-eval 200       --max-items 200       --max-new-tokens 16       --device cuda       --external-only       --require-external-sources       --min-pool-size 500       --out-root 'research/results/autoresearch/prompt_composition_ablation_v1'

  run_if_missing     "Llama 3.2 1B cross-model n_icl=8"     "${LLAMA1B_N8[@]}" --     env HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0     "${PYTHON_BIN}" experiments/cross_model_behavioral_check.py       --hf-id 'meta-llama/Llama-3.2-1B-Instruct'       --model-label 'llama3.2-1b'       --pairs 'aksharantar_hin_latin,aksharantar_tel_latin'       --seed 42       --n-icl 8       --n-select 300       --n-eval 200       --max-items 200       --max-new-tokens 16       --device cuda       --out-root 'research/results/autoresearch/cross_model_behavioral_v1'

  run_if_missing     "Llama 3.2 1B cross-model n_icl=64"     "${LLAMA1B_N64[@]}" --     env HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0     "${PYTHON_BIN}" experiments/cross_model_behavioral_check.py       --hf-id 'meta-llama/Llama-3.2-1B-Instruct'       --model-label 'llama3.2-1b'       --pairs 'aksharantar_hin_latin,aksharantar_tel_latin'       --seed 42       --n-icl 64       --n-select 300       --n-eval 200       --max-items 200       --max-new-tokens 16       --device cuda       --out-root 'research/results/autoresearch/cross_model_behavioral_v1'

  run_if_missing     "Llama 3.2 3B cross-model n_icl=8"     "${LLAMA3B_N8[@]}" --     env HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0     "${PYTHON_BIN}" experiments/cross_model_behavioral_check.py       --hf-id 'meta-llama/Llama-3.2-3B-Instruct'       --model-label 'llama3.2-3b'       --pairs 'aksharantar_hin_latin,aksharantar_tel_latin'       --seed 42       --n-icl 8       --n-select 300       --n-eval 200       --max-items 200       --max-new-tokens 16       --device cuda       --out-root 'research/results/autoresearch/cross_model_behavioral_v1'

  run_if_missing     "Llama 3.2 3B cross-model n_icl=64"     "${LLAMA3B_N64[@]}" --     env HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0     "${PYTHON_BIN}" experiments/cross_model_behavioral_check.py       --hf-id 'meta-llama/Llama-3.2-3B-Instruct'       --model-label 'llama3.2-3b'       --pairs 'aksharantar_hin_latin,aksharantar_tel_latin'       --seed 42       --n-icl 64       --n-select 300       --n-eval 200       --max-items 200       --max-new-tokens 16       --device cuda       --out-root 'research/results/autoresearch/cross_model_behavioral_v1'

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] remote pending tmux queue done"
} | tee "${LOG_PATH}"
