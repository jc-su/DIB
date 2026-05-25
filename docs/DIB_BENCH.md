# DIB-Bench Framework

DIB-Bench is the benchmark-facing form of the deployment-integrity blindness experiment.
It is designed to look like an agent benchmark where that helps reproducibility, and like an integrity-evidence interface where that helps deployment accountability.

## Positioning

DIB-Bench is not trying to replace existing agent benchmarks or supply-chain frameworks.

- Agent benchmarks such as ToolEmu, tau-bench, ToolSandbox, and AgentDojo primarily evaluate behavior: what the agent does in tasks, tools, users, and adversarial contexts.
- Supply-chain frameworks such as in-toto and SLSA primarily provide provenance and integrity evidence: what artifact was built, by whom, and through which steps.
- DIB-Bench evaluates the missing interface: when behavior looks acceptable, does the deployed agent still match the declared agent?

## Benchmark Object

A DIB-Bench run has four objects:

1. `tasks`: prompts and expected outcomes for fact, policy, tool, secret, and canary task families.
2. `deployment condition`: clean or one controlled deployment component change.
3. `outputs`: generated text plus runtime traces.
4. `integrity evidence`: expected-vs-observed component hashes and mismatch lists.

The benchmark unit is not just the model. It is the agent deployment stack:

- model/tokenizer references;
- system prompt;
- tool specification and router;
- retrieval policy and corpus entries;
- runtime dependency descriptor;
- postprocessor.

## Lifecycle

```bash
python integrity_blindness.py make_tasks --out tasks.jsonl --n 120 --seed 42

python integrity_blindness.py run \
  --tasks tasks.jsonl \
  --condition clean \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --sub_model Qwen/Qwen2.5-1.5B-Instruct \
  --engine vllm \
  --out outputs/clean.jsonl

python integrity_blindness.py score \
  --inputs outputs/*.jsonl \
  --out_dir results \
  --accept_threshold 0.80 \
  --bootstrap_samples 1000

python integrity_blindness.py aggregate \
  --inputs paper_runs \
  --out_dir paper_runs/aggregate \
  --include_run_id_regex "_vllm_seed[0-9]+$"
```

## External/Public Benchmark Adapter

The controlled synthetic generator is useful for causal tamper studies, but DIB-Bench also accepts external benchmark records.
Use `import_tasks` to convert JSONL, JSON, or CSV records into the DIB task schema:

```bash
python integrity_blindness.py import_tasks \
  --source examples/public_tasks_sample.jsonl \
  --source_name public_sample \
  --out imported_tasks.jsonl \
  --id_field id
```

For public agent/tool benchmarks, export the benchmark cases to JSONL/CSV with at least:

- a prompt-like field: `prompt`, `question`, `instruction`, `input`, `goal`, or pass `--prompt_field`;
- an expected field: `expected`, `answer`, `label`, `target`, `expected_tool`, or pass `--expected_field`;
- optionally `task_type`, one of `fact`, `policy`, `tool`, `secret`, or `canary`.

Then run the same DIB conditions and scoring pipeline over the imported task file.
This gives the framework a public-benchmark path while preserving the controlled synthetic path used for the main causal result.

## Standard Outputs

The aggregate command writes the benchmark package:

- `dib_bench_framework.json`: machine-readable framework definition.
- `dib_bench_card.md`: dataset/benchmark card for the current aggregate package.
- `deployment_manifest.schema.json`: schema for declared and observed deployment manifests.
- `integrity_evidence.schema.json`: schema for component-level integrity evidence.
- `scored_example.schema.json`: schema for row-level scored examples.
- `paper_table_framework_interface.md/.tex`: framework interface table for papers.
- `aggregate_condition.csv`: main condition-level metrics.
- `aggregate_model_pair_condition.csv`: robustness metrics by model pair.
- `aggregate_task_type.csv`: task-family breakdown.
- `all_scored.csv`: full scored example-level data.

## Metrics

DIB-Bench reports behavior and integrity separately:

- `standard_output_pass_rate`: task-specific output correctness and safety.
- `strict_output_pass_rate`: stricter format and leakage checks.
- `output_forensic_pass_rate`: standard output pass plus absence of obvious output-side signals.
- `canary_probe_pass_rate`: behavioral canary pass rate.
- `integrity_pass_rate`: manifest component match rate.
- `example_blindness_rate`: examples where output passes but integrity fails.
- `suite_accept_blindness@tau`: whether the output suite accepts a tampered deployment at threshold `tau`.

## Evidence Boundary

DIB-Bench uses software manifests and canonical component hashes. That is enough to measure the NLP evaluation blind spot, but it is not a production attestation system.

Production systems can replace or augment the manifest evidence with signed provenance, SLSA/in-toto attestations, reproducible builds, container signatures, or hardware-rooted runtime attestation.

The paper claim should stay precise:

> Output-only evaluation cannot establish deployment identity. Agentic NLP evaluation should report behavioral scores and deployment-integrity evidence as separate evidence channels.
