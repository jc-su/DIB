#!/usr/bin/env python3
"""Build a small BFCL-simple anchor in DIB-Bench tool-task schema.

Each BFCL case has a user message, a function definition, and a single ground-truth
function call. We wrap each case as a tool-selection task where the model must output
TOOL: <function_name>, matching DIB-Bench's existing tool scorer.
"""

import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download


def load_bfcl_simple(repo_id: str = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"):
    q_path = hf_hub_download(repo_id, "BFCL_v3_simple.json", repo_type="dataset")
    a_path = hf_hub_download(repo_id, "possible_answer/BFCL_v3_simple.json", repo_type="dataset")
    questions, answers = [], {}
    with open(q_path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    with open(a_path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                answers[row["id"]] = row
    return questions, answers


def extract_user_text(question_field) -> str:
    parts = []
    for turn_group in question_field:
        for msg in turn_group:
            if msg.get("role") == "user":
                parts.append(msg.get("content", ""))
    return "\n\n".join(parts).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    questions, answers = load_bfcl_simple()
    rng = random.Random(args.seed)
    indices = list(range(len(questions)))
    rng.shuffle(indices)

    rows = []
    for idx in indices:
        if len(rows) >= args.n:
            break
        q = questions[idx]
        qid = q["id"]
        if qid not in answers:
            continue
        functions = q.get("function", [])
        if not functions:
            continue
        function = functions[0]
        fn_name = function["name"]
        # Regex match in DIB scorer expects [A-Za-z_]+; skip names with digits.
        if not all(ch.isalpha() or ch == "_" for ch in fn_name):
            continue

        ground_truth = answers[qid]["ground_truth"]
        if not ground_truth or fn_name not in ground_truth[0]:
            continue

        user_text = extract_user_text(q["question"])
        if not user_text:
            continue

        fn_desc = function.get("description", "")
        param_list = []
        for pname, pinfo in function.get("parameters", {}).get("properties", {}).items():
            ptype = pinfo.get("type", "")
            pdesc = pinfo.get("description", "").split(".")[0]
            param_list.append(f"  - {pname} ({ptype}): {pdesc}")
        params_str = "\n".join(param_list)

        prompt = (
            f"Available function:\n"
            f"  {fn_name}: {fn_desc}\n"
            f"Parameters:\n{params_str}\n\n"
            f"User request: {user_text}\n\n"
            f"Choose the correct function for this request. "
            f"Respond exactly in this format: TOOL: <function_name>"
        )

        rows.append({
            "task_id": f"bfcl_{qid}",
            "task_type": "tool",
            "user_prompt": prompt,
            "expected": fn_name,
            "safety_critical": False,
            "secret": "",
            "source_benchmark": "bfcl_v3_simple",
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} BFCL tasks to {out_path}")


if __name__ == "__main__":
    main()
