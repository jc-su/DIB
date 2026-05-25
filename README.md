# Deployment-Integrity Blindness in LLM-Agent Evaluation

This repository contains a controlled empirical protocol for a missing evaluation object in LLM-agent systems: deployment integrity.

The central question is:

> Can output-only NLP evaluation establish that a deployed agent is the same system that was approved, evaluated, and intended to be trusted?

The project is packaged as **DIB-Bench**: a benchmark and evidence framework for measuring when output-only evaluation falsely accepts a deployment-integrity failure. See [`docs/DIB_BENCH.md`](docs/DIB_BENCH.md) for the framework interface.

The benchmark separates two kinds of evidence:

- **Behavioral evidence:** whether model outputs satisfy task checks.
- **Deployment-integrity evidence:** whether observed runtime components match the declared deployment manifest.

The point is not that manifests replace production attestation. The point is that output quality and deployment identity are different claims, and NLP evaluation should report both for deployed agents.

## Repository structure

```
.
├── integrity_blindness.py   # main CLI: make_tasks, run, validate, score, aggregate, import_tasks
├── dib_bench/               # DIB-Bench framework package (spec, framework, public-task adapters)
├── scripts/                 # figure/table generation and public-benchmark anchors (GSM8K, BFCL)
├── run_all.sh               # single-run pipeline: generate -> run -> validate -> score
├── run_paper_matrix.sh      # multi-seed / multi-model-pair matrix runner + aggregate
├── examples/                # tiny public-task format sample
├── docs/                    # framework interface, experiment protocol, reproducibility guide
└── requirements.txt
```

Experiment outputs (raw generations, scored results, aggregates, and generated
task files) are **not committed**: they are regenerated from the protocol and
are listed in `.gitignore`. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
to reproduce every number, table, and figure in the paper.

## Framework Shape

DIB-Bench deliberately borrows structure from two neighboring tool families:

- Like agent benchmarks, it defines task families, deployment conditions, output evaluators, metrics, and aggregate reports.
- Like provenance and attestation systems, it records expected-vs-observed component evidence, manifest hashes, and runtime traces.
- Unlike either family alone, it measures the bridge metric: output-passing integrity failures.

The aggregate command writes a benchmark package under `paper_runs/aggregate/`, including `dib_bench_framework.json`, `dib_bench_card.md`, `deployment_manifest.schema.json`, `integrity_evidence.schema.json`, `scored_example.schema.json`, and paper-ready framework tables.

## Deployment Conditions

The protocol evaluates:

1. Clean deployment
2. Model substitution
3. System prompt tampering
4. Tool specification / router tampering
5. Retrieval policy / retrieved document tampering
6. Runtime dependency tampering
7. Post-processing tampering

Each tampered condition changes one manifest component. The output evaluator only sees responses; the integrity evaluator compares expected and observed component hashes.

## Task Families

The generator creates:

1. Factual context QA
2. Policy allow/deny decisions
3. Tool routing decisions
4. Secret-leak refusal tasks
5. Output-only deployment canary probes

The canary probes are still behavioral tests. They are included to compare ordinary output scoring with stronger output-only probing.

## Metrics

The main reported fields in `results/summary.csv` are:

- `standard_output_pass_rate`: conventional task-level output score.
- `strict_output_pass_rate`: stricter format and leakage-sensitive output score.
- `canary_probe_pass_rate`: behavioral canary pass rate.
- `integrity_pass_rate`: manifest-level deployment-integrity pass rate.
- `integrity_detection_rate`: fraction of examples in which integrity evidence detects a mismatch.
- `example_blindness_rate`: examples where output passes but integrity fails.
- `standard_suite_accept_blindness`: whether a condition would be accepted by output-only evaluation at the configured threshold while failing integrity.
- `behavior_integrity_gap`: output pass rate minus integrity pass rate.

Bootstrap confidence intervals are included for the major output and blindness metrics.

## Recommended Run

For paper data collection across seeds, use the matrix runner:

```bash
SEEDS="42 43 44" \
MODEL_SPECS=$'qwen7b|Qwen/Qwen2.5-7B-Instruct|Qwen/Qwen2.5-1.5B-Instruct|1|2048|32|0' \
bash run_paper_matrix.sh
```

For the larger primary setting on two GPUs:

```bash
SEEDS="42 43 44" \
MODEL_SPECS=$'qwen14b|Qwen/Qwen2.5-14B-Instruct|Qwen/Qwen2.5-7B-Instruct|2|2048|32|0,1' \
bash run_paper_matrix.sh
```

The matrix runner resumes complete condition files, validates every run, scores every run, then writes aggregate CSVs, plots, and paper tables under `paper_runs/aggregate/`.

To use DIB-Bench with an external/public benchmark export instead of generated controlled tasks:

```bash
python integrity_blindness.py import_tasks \
  --source external_benchmark.jsonl \
  --source_name my_public_benchmark \
  --out imported_tasks.jsonl \
  --prompt_field prompt \
  --expected_field expected \
  --task_type_field task_type
```

The imported task file can then be passed to the same `run`, `score`, and `aggregate` pipeline. A tiny format example is in `examples/public_tasks_sample.jsonl`.

For 2 x RTX A6000 / RTX Ada 6000 class GPUs:

```bash
conda create -n blind_eval python=3.10 -y
conda activate blind_eval
pip install -r requirements.txt

BASE_MODEL=Qwen/Qwen2.5-14B-Instruct \
SUB_MODEL=Qwen/Qwen2.5-7B-Instruct \
ENGINE=vllm \
ATTENTION_BACKEND=FLASHINFER \
TP=2 \
N=120 \
bash run_all.sh
```

Fast fallback:

```bash
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
SUB_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
ENGINE=vllm \
ATTENTION_BACKEND=FLASHINFER \
TP=1 \
N=120 \
bash run_all.sh
```

Reproducibility fallback if vLLM has local kernel or FlashAttention issues:

```bash
BASE_MODEL=facebook/opt-125m \
SUB_MODEL=gpt2 \
ENGINE=transformers \
TP=1 \
N=20 \
MAX_MODEL_LEN=512 \
BATCH_SIZE=2 \
TASKS_FILE=smoke_tasks.jsonl \
OUTPUT_DIR=smoke_outputs \
RESULTS_DIR=smoke_results \
bash run_all.sh
```

This fallback is not the paper-quality model comparison; it is a pipeline smoke test.

## Troubleshooting

If a Qwen run through vLLM fails with `ModuleNotFoundError: No module named 'flash_attn.ops'`, the experiment code is not the failing layer; the local vLLM / FlashAttention / CUDA stack is inconsistent. One common cause is `flash-attn-4`, which exposes `flash_attn.cute` but not the older `flash_attn.ops.triton.rotary` module that some vLLM/Qwen rotary paths probe for. The runner hides this incomplete namespace by default and can force FlashInfer with `ATTENTION_BACKEND=FLASHINFER`.

For a cleaner long-term vLLM setup, use a fresh environment and let vLLM install its matched PyTorch/CUDA stack. For slower but simple reproducibility checks, use `ENGINE=transformers`.

## Outputs

After running:

```bash
cat results/summary.csv
cat results/summary_by_task_type.csv
```

Generated files:

- `outputs/*.jsonl`: raw generations, manifest hashes, integrity evidence, retrieval traces, tool traces, and postprocessor traces.
- `results/detailed_scored.csv`: row-level output and integrity scores.
- `results/summary.csv`: condition-level metrics with confidence intervals.
- `results/summary_by_task_type.csv`: breakdown by task type.
- `results/component_mismatch_summary.csv`: detected component mismatches by condition.
- `results/paper_table_condition.md` and `.tex`: paper-ready condition table.
- `results/output_vs_integrity.png`: behavioral vs integrity pass rates.
- `results/blindness_rate.png`: deployment-integrity blindness rates.

For multi-run aggregation:

```bash
python integrity_blindness.py aggregate \
  --inputs paper_runs \
  --out_dir paper_runs/aggregate \
  --include_run_id_regex "_vllm_seed[0-9]+$"
```

Aggregate outputs include `aggregate_condition.csv`, `aggregate_model_pair_condition.csv`, `aggregate_task_type.csv`, `aggregate_tamper_class.csv`, `runs.csv`, DIB-Bench schemas/cards, and paper-ready Markdown/TeX tables.

## Positioning

This is a methodology paper, not a new hardware attestation system.

Framing:

> We operationalize deployment-integrity evidence using a lightweight manifest over model/tokenizer, prompt, tool, retrieval, runtime, and postprocessor components. This abstraction isolates the evaluation question: can output-only evaluation detect when the deployed agent differs from the declared agent?

Connection to production evidence:

> Production deployments can replace this software manifest with stronger evidence from signed provenance, supply-chain attestations, or hardware-rooted runtime attestation. Our contribution is to show why such evidence belongs in NLP evaluation reports for agentic systems.
