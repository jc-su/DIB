#!/usr/bin/env python3
"""Build a small GSM8K anchor task file in DIB-Bench schema."""

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


def extract_final_number(answer_text: str) -> str:
    m = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", answer_text)
    if not m:
        return ""
    return m.group(1).replace(",", "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    ds = ds.shuffle(seed=args.seed)

    rows = []
    for idx, example in enumerate(ds):
        if len(rows) >= args.n:
            break
        expected = extract_final_number(example["answer"])
        if not expected:
            continue
        question = example["question"].strip()
        user_prompt = (
            f"Solve this problem and give only the final numeric answer.\n\n"
            f"{question}\n\n"
            f"Respond exactly in this format: ANSWER: <number>"
        )
        rows.append({
            "task_id": f"gsm8k_{idx:05d}",
            "task_type": "fact",
            "user_prompt": user_prompt,
            "expected": expected,
            "safety_critical": False,
            "secret": "",
            "source_benchmark": "gsm8k",
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} GSM8K tasks to {out_path}")


if __name__ == "__main__":
    main()
