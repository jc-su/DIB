#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_all.sh
#
# Optional overrides:
#   BASE_MODEL=Qwen/Qwen2.5-14B-Instruct SUB_MODEL=Qwen/Qwen2.5-7B-Instruct ENGINE=vllm ATTENTION_BACKEND=FLASHINFER TP=2 N=120 bash run_all.sh
#
# For 2x RTX Ada 6000, TP=2 is recommended for 14B models.
# If you want a faster run, use:
#   BASE_MODEL=Qwen/Qwen2.5-7B-Instruct SUB_MODEL=Qwen/Qwen2.5-1.5B-Instruct ENGINE=vllm ATTENTION_BACKEND=FLASHINFER TP=1 N=120 bash run_all.sh
#
# If vLLM has local kernel dependency issues, use ENGINE=transformers for a
# slower but simpler reproducibility run.

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
SUB_MODEL="${SUB_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-main}"
SUB_MODEL_REVISION="${SUB_MODEL_REVISION:-main}"
TOKENIZER_ID="${TOKENIZER_ID:-}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-main}"
ENGINE="${ENGINE:-vllm}"
TP="${TP:-2}"
N="${N:-120}"
SEED="${SEED:-42}"
GPU_UTIL="${GPU_UTIL:-0.88}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
ACCEPT_THRESHOLD="${ACCEPT_THRESHOLD:-0.80}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
TASKS_FILE="${TASKS_FILE:-tasks.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
RESULTS_DIR="${RESULTS_DIR:-results}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
RUN_ID="${RUN_ID:-$(basename "$OUTPUT_DIR")}"
RESUME="${RESUME:-1}"

export CUDA_VISIBLE_DEVICES

mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR"

if [[ "$RESUME" == "1" && -s "$TASKS_FILE" && "$(wc -l < "$TASKS_FILE")" -eq "$N" ]]; then
  echo "===== Reusing tasks: $TASKS_FILE ====="
else
  python integrity_blindness.py make_tasks --out "$TASKS_FILE" --n "$N" --seed "$SEED"
fi

attention_args=()
if [[ -n "$ATTENTION_BACKEND" ]]; then
  attention_args+=(--attention_backend "$ATTENTION_BACKEND")
fi

complete_jsonl() {
  local path="$1"
  [[ -s "$path" ]] && [[ "$(wc -l < "$path")" -eq "$N" ]]
}

for c in clean model_substitution prompt_tamper tool_tamper retrieval_tamper dependency_tamper postprocess_tamper
do
  echo "===== Running condition: $c ====="
  out_file="${OUTPUT_DIR}/${c}.jsonl"
  if [[ "$RESUME" == "1" ]] && complete_jsonl "$out_file"; then
    echo "Skipping $c; found complete output at $out_file"
    continue
  fi
  python integrity_blindness.py run \
    --tasks "$TASKS_FILE" \
    --condition "$c" \
    --base_model "$BASE_MODEL" \
    --sub_model "$SUB_MODEL" \
    --model_revision "$MODEL_REVISION" \
    --sub_model_revision "$SUB_MODEL_REVISION" \
    --tokenizer_id "$TOKENIZER_ID" \
    --tokenizer_revision "$TOKENIZER_REVISION" \
    --engine "$ENGINE" \
    --tp "$TP" \
    --gpu_memory_utilization "$GPU_UTIL" \
    --max_model_len "$MAX_MODEL_LEN" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --run_id "$RUN_ID" \
    "${attention_args[@]}" \
    --out "$out_file"
done

python integrity_blindness.py validate \
  --outputs_dir "$OUTPUT_DIR" \
  --tasks "$TASKS_FILE" \
  --expected_n "$N" \
  --out "${RESULTS_DIR}/validation_report.json"

python integrity_blindness.py score \
  --inputs "$OUTPUT_DIR"/*.jsonl \
  --out_dir "$RESULTS_DIR" \
  --accept_threshold "$ACCEPT_THRESHOLD" \
  --bootstrap_samples "$BOOTSTRAP_SAMPLES" \
  --seed "$SEED"

echo ""
echo "Done. See:"
echo "  ${RESULTS_DIR}/summary.csv"
echo "  ${RESULTS_DIR}/summary_by_task_type.csv"
echo "  ${RESULTS_DIR}/detailed_scored.csv"
echo "  ${RESULTS_DIR}/output_vs_integrity.png"
echo "  ${RESULTS_DIR}/blindness_rate.png"
