"""Task adapters for using DIB-Bench with external/public benchmarks."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .spec import VALID_TASK_TYPES


PROMPT_CANDIDATES = [
    "user_prompt",
    "prompt",
    "instruction",
    "question",
    "input",
    "goal",
    "task",
]

EXPECTED_CANDIDATES = [
    "expected",
    "answer",
    "label",
    "target",
    "gold",
    "reference",
    "expected_answer",
    "expected_tool",
    "decision",
]


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return default


def read_records(path: str, source_format: str = "auto") -> List[Dict[str, Any]]:
    src = Path(path)
    fmt = source_format
    if fmt == "auto":
        fmt = "csv" if src.suffix.lower() == ".csv" else "jsonl"
    if fmt == "jsonl":
        records = []
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records
    if fmt == "json":
        data = json.loads(src.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(x) for x in data]
        if isinstance(data, dict):
            for key in ["data", "examples", "tasks", "records"]:
                if isinstance(data.get(key), list):
                    return [dict(x) for x in data[key]]
        raise ValueError(f"Could not find a list of records in {path}")
    if fmt == "csv":
        with src.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    raise ValueError(f"Unsupported source format: {source_format}")


def first_present(record: Dict[str, Any], candidates: Iterable[str], explicit: str = "") -> Optional[Any]:
    if explicit:
        return record.get(explicit)
    for key in candidates:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def normalize_task_type(value: Any, default: str) -> str:
    text = str(value or default).strip().lower()
    aliases = {
        "qa": "fact",
        "question_answering": "fact",
        "classification": "policy",
        "decision": "policy",
        "tool_use": "tool",
        "tool_call": "tool",
        "refusal": "secret",
        "safety": "secret",
    }
    text = aliases.get(text, text)
    if text not in VALID_TASK_TYPES:
        return default
    return text


def convert_public_records(
    records: List[Dict[str, Any]],
    source_name: str,
    prompt_field: str = "",
    expected_field: str = "",
    task_type_field: str = "",
    default_task_type: str = "fact",
    id_field: str = "",
    limit: int = 0,
) -> List[Dict[str, Any]]:
    if default_task_type not in VALID_TASK_TYPES:
        raise ValueError(f"default_task_type must be one of {sorted(VALID_TASK_TYPES)}")

    out = []
    for idx, record in enumerate(records):
        if limit and len(out) >= limit:
            break
        prompt = first_present(record, PROMPT_CANDIDATES, explicit=prompt_field)
        expected = first_present(record, EXPECTED_CANDIDATES, explicit=expected_field)
        if prompt is None or expected is None:
            continue
        raw_type = record.get(task_type_field) if task_type_field else record.get("task_type")
        task_type = normalize_task_type(raw_type, default=default_task_type)
        raw_id = record.get(id_field) if id_field else record.get("task_id", record.get("id"))
        task_id = str(raw_id if raw_id not in (None, "") else f"{source_name}_{idx:05d}")
        secret = str(record.get("secret", ""))
        safety_critical = parse_bool(
            record.get("safety_critical"),
            default=task_type in {"policy", "secret", "canary"},
        )
        out.append({
            "task_id": task_id,
            "task_type": task_type,
            "user_prompt": str(prompt),
            "expected": str(expected),
            "safety_critical": safety_critical,
            "secret": secret,
            "source_benchmark": source_name,
            "source_record": record,
        })
    if not out:
        raise ValueError(
            "No usable records converted. Check prompt/expected field names or provide "
            "--prompt_field and --expected_field."
        )
    return out


def convert_public_task_file(args: Any) -> List[Dict[str, Any]]:
    source_name = args.source_name or Path(args.source).stem
    records = read_records(args.source, source_format=args.source_format)
    return convert_public_records(
        records,
        source_name=source_name,
        prompt_field=args.prompt_field,
        expected_field=args.expected_field,
        task_type_field=args.task_type_field,
        default_task_type=args.default_task_type,
        id_field=args.id_field,
        limit=args.limit,
    )
