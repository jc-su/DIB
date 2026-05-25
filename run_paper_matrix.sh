#!/usr/bin/env bash
set -euo pipefail

# Multi-run data collection for paper tables.
#
# Default: a GPU0-friendly 7B -> 1.5B matrix over three task seeds.
# To run the larger 14B -> 7B primary setting on two GPUs, override MODEL_SPECS:
#
# MODEL_SPECS=$'qwen14b|Qwen/Qwen2.5-14B-Instruct|Qwen/Qwen2.5-7B-Instruct|2|2048|32|0,1' \
# SEEDS="42 43 44" bash run_paper_matrix.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/paper_runs}"
SEEDS="${SEEDS:-42 43 44}"
N="${N:-120}"
ENGINE="${ENGINE:-vllm}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASHINFER}"
GPU_UTIL="${GPU_UTIL:-0.88}"
ACCEPT_THRESHOLD="${ACCEPT_THRESHOLD:-0.80}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
RESUME="${RESUME:-1}"

MODEL_SPECS="${MODEL_SPECS:-qwen7b|Qwen/Qwen2.5-7B-Instruct|Qwen/Qwen2.5-1.5B-Instruct|1|2048|32|0}"

mkdir -p "$RUN_ROOT"

while IFS='|' read -r name base_model sub_model tp max_model_len batch_size cuda_devices
do
  [[ -z "${name:-}" ]] && continue
  [[ "${name:0:1}" == "#" ]] && continue

  for seed in $SEEDS
  do
    run_id="${name}_${ENGINE}_seed${seed}"
    run_dir="${RUN_ROOT}/${run_id}"
    echo ""
    echo "===== Paper run: ${run_id} ====="
    (
      cd "$ROOT_DIR"
      RUN_ID="$run_id" \
      BASE_MODEL="$base_model" \
      SUB_MODEL="$sub_model" \
      ENGINE="$ENGINE" \
      ATTENTION_BACKEND="$ATTENTION_BACKEND" \
      TP="$tp" \
      N="$N" \
      SEED="$seed" \
      GPU_UTIL="$GPU_UTIL" \
      MAX_MODEL_LEN="$max_model_len" \
      BATCH_SIZE="$batch_size" \
      CUDA_VISIBLE_DEVICES="$cuda_devices" \
      ACCEPT_THRESHOLD="$ACCEPT_THRESHOLD" \
      BOOTSTRAP_SAMPLES="$BOOTSTRAP_SAMPLES" \
      TASKS_FILE="${run_dir}/tasks.jsonl" \
      OUTPUT_DIR="${run_dir}/outputs" \
      RESULTS_DIR="${run_dir}/results" \
      RESUME="$RESUME" \
      bash run_all.sh
    )
  done
done <<< "$MODEL_SPECS"

python "${ROOT_DIR}/integrity_blindness.py" aggregate \
  --inputs "$RUN_ROOT" \
  --out_dir "${RUN_ROOT}/aggregate" \
  --accept_threshold "$ACCEPT_THRESHOLD" \
  --bootstrap_samples "$BOOTSTRAP_SAMPLES" \
  --seed 42 \
  --include_run_id_regex "_${ENGINE}_seed[0-9]+$"

echo ""
echo "Paper matrix complete. Aggregate outputs:"
echo "  ${RUN_ROOT}/aggregate/aggregate_condition.csv"
echo "  ${RUN_ROOT}/aggregate/aggregate_model_pair_condition.csv"
echo "  ${RUN_ROOT}/aggregate/paper_table_condition.md"
