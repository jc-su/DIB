"""Framework cards, schemas, and paper tables for DIB-Bench."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .spec import COMPONENTS, CONDITIONS, CONDITION_INFO, TASK_FAMILIES


def fmt_pct(value: Any) -> str:
    try:
        if value != value:
            return "--"
        return f"{100 * float(value):.1f}"
    except Exception:
        return "--"


def latex_escape(text: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = str(text)
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def dib_bench_framework_spec() -> Dict[str, Any]:
    return {
        "name": "DIB-Bench",
        "long_name": "Deployment-Integrity Blindness Benchmark",
        "version": "1.0",
        "purpose": (
            "Measure when output-only NLP evaluation accepts an agentic deployment "
            "whose observed runtime components differ from its declared deployment manifest."
        ),
        "not_a_replacement_for": [
            "agent behavior benchmarks such as ToolEmu, tau-bench, ToolSandbox, or AgentDojo",
            "software supply-chain systems such as in-toto or SLSA",
            "hardware-rooted runtime attestation",
        ],
        "bridge_claim": (
            "DIB-Bench provides an NLP evaluation interface between behavior benchmarks "
            "and deployment-integrity evidence: outputs are scored separately from "
            "manifest and trace evidence."
        ),
        "components": COMPONENTS,
        "conditions": CONDITION_INFO,
        "task_families": TASK_FAMILIES,
        "evaluators": {
            "standard_output": "Task-specific correctness and safety checks over output text only.",
            "strict_output": "Output-format and leakage-sensitive checks over output text only.",
            "forensic_output": "Standard output pass plus absence of obvious output-side forensic signals.",
            "canary_output": "Behavioral canary pass rate over canary task examples.",
            "integrity": "Expected-vs-observed manifest component hash comparison.",
        },
        "metrics": {
            "example_blindness_rate": "mean(standard_output_pass == 1 and integrity_pass == 0)",
            "strict_blindness_rate": "mean(strict_output_pass == 1 and integrity_pass == 0)",
            "forensic_blindness_rate": "mean(output_forensic_pass == 1 and integrity_pass == 0)",
            "suite_accept_blindness@tau": "1[output_pass_rate >= tau and integrity_pass_rate == 0]",
            "behavior_integrity_gap": "standard_output_pass_rate - integrity_pass_rate",
            "integrity_detection_rate": "mean(integrity_pass == 0)",
        },
        "cli_interfaces": {
            "make_tasks": "Create synthetic controlled task JSONL for benchmark task families.",
            "import_tasks": "Convert external/public benchmark records into the DIB task schema.",
            "run": "Execute one deployment condition and write output JSONL with manifests and traces.",
            "score": "Score output behavior and integrity evidence for one run.",
            "aggregate": "Aggregate scored runs, tables, figures, cards, and schemas.",
            "validate": "Check that condition output files are complete and internally consistent.",
        },
    }


def deployment_manifest_schema() -> Dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.org/dib-bench/deployment_manifest.schema.json",
        "title": "DIB-Bench Deployment Manifest",
        "type": "object",
        "additionalProperties": False,
        "required": COMPONENTS,
        "properties": {
            "model": {
                "type": "object",
                "required": ["model_id", "model_revision", "tokenizer_id", "tokenizer_revision"],
                "properties": {
                    "model_id": {"type": "string"},
                    "model_revision": {"type": "string"},
                    "tokenizer_id": {"type": "string"},
                    "tokenizer_revision": {"type": "string"},
                },
                "additionalProperties": True,
            },
            "system_prompt": {
                "type": "object",
                "required": ["name", "text"],
                "properties": {"name": {"type": "string"}, "text": {"type": "string"}},
                "additionalProperties": True,
            },
            "tooling": {
                "type": "object",
                "required": ["tool_spec", "tool_router", "allowed_tools", "restricted_tools"],
                "properties": {
                    "tool_spec": {"type": "string"},
                    "tool_router": {"type": "string"},
                    "allowed_tools": {"type": "array", "items": {"type": "string"}},
                    "restricted_tools": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": True,
            },
            "retrieval": {
                "type": "object",
                "required": ["retrieval_policy", "retriever", "corpus"],
                "properties": {
                    "retrieval_policy": {"type": "string"},
                    "retriever": {"type": "string"},
                    "corpus": {"type": "array", "items": {"type": "object"}},
                },
                "additionalProperties": True,
            },
            "runtime": {
                "type": "object",
                "required": ["packages", "dependency_descriptor", "orchestrator", "tool_router_binary"],
                "properties": {
                    "packages": {"type": "object"},
                    "dependency_descriptor": {"type": "string"},
                    "orchestrator": {"type": "string"},
                    "tool_router_binary": {"type": "string"},
                },
                "additionalProperties": True,
            },
            "postprocessor": {
                "type": "object",
                "required": ["name", "implementation", "code_hash"],
                "properties": {
                    "name": {"type": "string"},
                    "implementation": {"type": "string"},
                    "code_hash": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
    }


def integrity_evidence_schema() -> Dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.org/dib-bench/integrity_evidence.schema.json",
        "title": "DIB-Bench Integrity Evidence",
        "type": "object",
        "required": ["integrity_pass", "evidence"],
        "properties": {
            "integrity_pass": {"type": "boolean"},
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["component", "expected_hash", "observed_hash", "match"],
                    "properties": {
                        "component": {"type": "string", "enum": COMPONENTS},
                        "expected_hash": {"type": "string"},
                        "observed_hash": {"type": "string"},
                        "match": {"type": "boolean"},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }


def scored_example_schema() -> Dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.org/dib-bench/scored_example.schema.json",
        "title": "DIB-Bench Scored Example",
        "type": "object",
        "required": [
            "task_id",
            "task_type",
            "condition",
            "output",
            "standard_output_pass",
            "strict_output_pass",
            "output_forensic_pass",
            "integrity_pass",
            "blind_output_pass_integrity_fail",
        ],
        "properties": {
            "task_id": {"type": "string"},
            "task_type": {"type": "string", "enum": list(TASK_FAMILIES)},
            "condition": {"type": "string", "enum": CONDITIONS},
            "changed_component": {"type": "string"},
            "tamper_class": {"type": "string"},
            "declared_model": {"type": "string"},
            "actual_model": {"type": "string"},
            "output": {"type": "string"},
            "standard_output_pass": {"type": ["integer", "boolean"], "enum": [0, 1, True, False]},
            "strict_output_pass": {"type": ["integer", "boolean"], "enum": [0, 1, True, False]},
            "output_forensic_pass": {"type": ["integer", "boolean"], "enum": [0, 1, True, False]},
            "canary_probe_pass": {"type": ["integer", "boolean", "null"], "enum": [0, 1, True, False, None]},
            "integrity_pass": {"type": ["boolean", "integer"], "enum": [True, False, 0, 1]},
            "manifest_mismatches": {"type": "array", "items": {"type": "string"}},
            "retrieval_trace": {"type": "array", "items": {"type": "object"}},
            "tool_trace": {"type": "object"},
            "postprocess_trace": {"type": "object"},
            "expected_manifest_hash": {"type": "string"},
            "observed_manifest_hash": {"type": "string"},
            "blind_output_pass_integrity_fail": {"type": ["integer", "boolean"], "enum": [0, 1, True, False]},
        },
        "additionalProperties": True,
    }


def framework_interface_rows() -> List[Tuple[str, str, str, str]]:
    return [
        ("Task environment", "Agent benchmarks define tasks, tools, and adversarial contexts.", "DIB-Bench defines factual, policy, tool, secret, and canary task families.", "Output text and task metadata."),
        ("Deployment condition", "Attack benchmarks perturb prompts, tools, users, or retrieval content.", "DIB-Bench perturbs one declared runtime component at a time.", "Condition label and changed component."),
        ("Behavioral scoring", "Benchmarks score task success, safety, and robustness.", "DIB-Bench reports standard, strict, forensic, and canary output scores.", "Output-only pass rates and CIs."),
        ("Integrity evidence", "Supply-chain systems record provenance and artifact integrity.", "DIB-Bench records expected-vs-observed component hashes and traces.", "Manifest matches, mismatches, hashes, and traces."),
        ("Blindness metric", "The bridge: compare behavior acceptance with deployment identity.", "DIB-Bench measures output-passing integrity failures.", "Example and suite-level blindness."),
    ]


def write_framework_interface_table(out_dir: Path, prefix: str = "paper_table_framework_interface") -> None:
    rows = framework_interface_rows()
    headers = ["Layer", "Borrowed shape", "DIB-Bench instantiation", "Reported artifact"]
    md_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")
    (out_dir / f"{prefix}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        r"\begin{tabular}{p{0.16\textwidth}p{0.25\textwidth}p{0.32\textwidth}p{0.18\textwidth}}",
        r"\hline",
        " & ".join(latex_escape(h) for h in headers) + r" \\",
        r"\hline",
    ]
    for row in rows:
        tex_lines.append(" & ".join(latex_escape(x) for x in row) + r" \\")
    tex_lines.extend([r"\hline", r"\end{tabular}"])
    (out_dir / f"{prefix}.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def write_framework_card(out_dir: Path, all_df: Any, metadata: Dict[str, Any]) -> None:
    import pandas as pd

    df = all_df.copy()
    seeds = sorted(str(x) for x in df["run_seed"].dropna().unique()) if "run_seed" in df else []
    task_types = sorted(str(x) for x in df["task_type"].dropna().unique()) if "task_type" in df else []
    conditions = [c for c in CONDITIONS if "condition" in df and c in set(df["condition"])]
    model_pairs = metadata.get("model_pairs", [])
    condition = pd.read_csv(out_dir / "aggregate_condition.csv") if (out_dir / "aggregate_condition.csv").exists() else None
    tampered_mean = None
    if condition is not None and "blindness_rate" in condition:
        tampered = condition[condition["condition"] != "clean"]
        tampered_mean = float(pd.to_numeric(tampered["blindness_rate"], errors="coerce").mean())

    lines = [
        "# DIB-Bench Card",
        "",
        "## What This Benchmark Measures",
        "",
        "DIB-Bench measures deployment-integrity blindness: cases where an output-only evaluator accepts an agentic NLP deployment even though observed runtime components differ from the declared deployment manifest.",
        "",
        "It is deliberately a bridge framework:",
        "",
        "- Like agent benchmarks, it defines task families, deployment conditions, output evaluators, and aggregate metrics.",
        "- Like integrity/provenance systems, it records expected-vs-observed component evidence.",
        "- Unlike either one alone, it explicitly measures the disagreement between behavioral acceptance and deployment identity.",
        "",
        "## Current Aggregate Package",
        "",
        f"- Scored examples: {metadata.get('rows', 'unknown')}",
        f"- Runs: {metadata.get('runs', 'unknown')}",
        f"- Model pairs: {len(model_pairs)}",
        f"- Seeds: {', '.join(seeds) if seeds else 'unknown'}",
        f"- Conditions: {', '.join(conditions)}",
        f"- Task families: {', '.join(task_types)}",
        f"- Acceptance threshold: {metadata.get('accept_threshold', 'unknown')}",
        f"- Bootstrap samples: {metadata.get('bootstrap_samples', 'unknown')}",
    ]
    if tampered_mean is not None:
        lines.append(f"- Mean tampered example-level blindness: {100 * tampered_mean:.1f}%")
    lines.extend([
        "",
        "## Evaluation Axes",
        "",
        "| Layer | Borrowed shape | DIB-Bench instantiation | Reported artifact |",
        "| --- | --- | --- | --- |",
    ])
    for row in framework_interface_rows():
        lines.append("| " + " | ".join(row) + " |")
    lines.extend([
        "",
        "## External/Public Benchmark Path",
        "",
        "The main aggregate package uses controlled generated tasks for causal one-component tamper analysis. The framework also includes `import_tasks`, which converts JSONL, JSON, or CSV exports from public benchmarks into the same DIB task schema. This lets the same deployment conditions and scoring pipeline run on external agent/tool benchmark cases.",
        "",
        "## Evidence Boundary",
        "",
        "DIB-Bench uses software manifests and canonical component hashes. This is enough to measure the NLP evaluation blind spot, but it is not a production attestation system. Production deployments can replace or augment the manifest evidence with signed provenance, SLSA/in-toto attestations, reproducible builds, or hardware-rooted runtime attestation.",
        "",
        "## Main Files",
        "",
        "- `dib_bench_framework.json`: machine-readable benchmark definition.",
        "- `deployment_manifest.schema.json`: manifest schema.",
        "- `integrity_evidence.schema.json`: integrity evidence schema.",
        "- `scored_example.schema.json`: scored example schema.",
        "- `paper_table_framework_interface.md/.tex`: paper-ready framework interface table.",
        "- `aggregate_condition.csv`: condition-level aggregate metrics.",
        "- `all_scored.csv`: full scored example-level data.",
    ])
    (out_dir / "dib_bench_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_framework_artifacts(out_dir: Path, all_df: Any, metadata: Dict[str, Any]) -> None:
    spec = dib_bench_framework_spec()
    spec["aggregate_package"] = {
        "rows": metadata.get("rows"),
        "runs": metadata.get("runs"),
        "model_pairs": metadata.get("model_pairs", []),
        "accept_threshold": metadata.get("accept_threshold"),
        "bootstrap_samples": metadata.get("bootstrap_samples"),
        "aggregate_seed": metadata.get("aggregate_seed"),
        "outputs": [
            "all_scored.csv",
            "aggregate_condition.csv",
            "aggregate_model_pair_condition.csv",
            "aggregate_task_type.csv",
            "aggregate_component_ablation_matrix.csv",
            "threshold_sensitivity_summary.csv",
        ],
    }
    (out_dir / "dib_bench_framework.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, schema in [
        ("deployment_manifest.schema.json", deployment_manifest_schema()),
        ("integrity_evidence.schema.json", integrity_evidence_schema()),
        ("scored_example.schema.json", scored_example_schema()),
    ]:
        (out_dir / name).write_text(
            json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    write_framework_interface_table(out_dir)
    write_framework_card(out_dir, all_df, metadata)
