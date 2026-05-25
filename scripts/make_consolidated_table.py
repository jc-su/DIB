#!/usr/bin/env python3
"""Build a consolidated three-benchmark condition-level table for the paper.

Reads the per-benchmark summary.csv (synthetic aggregate, GSM8K, BFCL) and emits
one tex table with multi-column headers grouping by benchmark.
"""

import argparse
import csv
from pathlib import Path

CONDITIONS = [
    ("clean", "clean"),
    ("model_substitution", "model sub."),
    ("prompt_tamper", "prompt"),
    ("tool_tamper", "tool"),
    ("retrieval_tamper", "retrieval"),
    ("dependency_tamper", "dependency"),
    ("postprocess_tamper", "postproc."),
]

METRICS = [
    ("standard_output_pass_rate", "Std"),
    ("strict_output_pass_rate", "Strict"),
    ("integrity_pass_rate", "Id."),
]


def read_rows(csv_path: Path):
    out = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            out[row["condition"]] = row
    return out


def fmt(val) -> str:
    if val is None or val == "":
        return "--"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val)
    return f"{f*100:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", default="paper_runs/aggregate/aggregate_condition.csv")
    parser.add_argument("--gsm8k", default="paper_runs/gsm8k_anchor/results/summary.csv")
    parser.add_argument("--bfcl", default="paper_runs/bfcl_anchor/results/summary.csv")
    parser.add_argument("--out_tex", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    synth = read_rows(Path(args.synth))
    gsm8k = read_rows(Path(args.gsm8k))
    bfcl = read_rows(Path(args.bfcl))
    benches = [("Synthetic suite", synth), ("GSM8K", gsm8k), ("BFCL-simple", bfcl)]

    # LaTeX
    n_per_bench = len(METRICS)
    col_spec = "l" + "".join(f"{'r' * n_per_bench}" for _ in benches)
    # Insert vertical separators between bench groups
    col_spec = "l" + " ".join(["r" * n_per_bench] * len(benches))
    tex_lines = [r"\begin{tabular}{l" + ("|" + "r" * n_per_bench) * len(benches) + "}"]
    tex_lines.append(r"\hline")
    # Multicol header
    top = ["Condition"]
    for bench_name, _ in benches:
        top.append(r"\multicolumn{" + str(n_per_bench) + r"}{c}{" + bench_name + r"}")
    tex_lines.append(" & ".join(top) + r" \\")
    # Sub-header
    sub = [""]
    for _ in benches:
        sub.extend([m[1] for m in METRICS])
    tex_lines.append(" & ".join(sub) + r" \\")
    tex_lines.append(r"\hline")
    for cond_key, cond_label in CONDITIONS:
        cells = [cond_label]
        for _, rows in benches:
            r = rows.get(cond_key, {})
            for metric, _ in METRICS:
                cells.append(fmt(r.get(metric)))
        tex_lines.append(" & ".join(cells) + r" \\")
    tex_lines.append(r"\hline")
    tex_lines.append(r"\end{tabular}")
    Path(args.out_tex).write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    # Markdown for the repo
    md_lines = []
    h1 = ["Condition"]
    for bench_name, _ in benches:
        for m_label in [m[1] for m in METRICS]:
            h1.append(f"{bench_name} {m_label}")
    md_lines.append("| " + " | ".join(h1) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(h1)) + " |")
    for cond_key, cond_label in CONDITIONS:
        cells = [cond_label]
        for _, rows in benches:
            r = rows.get(cond_key, {})
            for metric, _ in METRICS:
                cells.append(fmt(r.get(metric)))
        md_lines.append("| " + " | ".join(cells) + " |")
    Path(args.out_md).write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"wrote {args.out_tex} and {args.out_md}")


if __name__ == "__main__":
    main()
