from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import argparse
import json
import queue
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import requests

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from mindpower.benchmark_remainder import DEFAULT_REMAINDER_PREP_DIR, DEFAULT_REMAINDER_RUN_DIR
from mindpower.common import (
    DEFAULT_KEY_LIST,
    EVAL_MODELS,
    MODEL_FRIENDLY_NAMES,
    append_jsonl,
    build_eval_prompt,
    load_api_keys,
    now_iso,
    parse_tagged_fields,
    read_jsonl,
    relative_to_root,
    render_reference_response,
    resolve_runtime_model,
    write_json,
    call_openrouter,
)


@dataclass(frozen=True)
class Task:
    sample_id: str
    model: str
    condition: str


SUPPORTED_CONDITIONS = ["C_text"]


def load_manifest_rows(path: Path) -> List[dict]:
    rows = read_jsonl(path)
    rows.sort(key=lambda row: str(row["sample_id"]))
    return rows


def load_completed_task_keys(path: Path) -> set[Tuple[str, str, str]]:
    completed = set()
    for row in read_jsonl(path):
        completed.add((str(row["sample_id"]), str(row["model"]), str(row["condition"])))
    return completed


def select_rows(rows: Sequence[dict], sample_ids: Sequence[str], limit_samples: int) -> List[dict]:
    selected = list(rows)
    if sample_ids:
        sample_set = {str(sample_id) for sample_id in sample_ids}
        selected = [row for row in selected if str(row["sample_id"]) in sample_set]
    if limit_samples > 0:
        selected = selected[:limit_samples]
    return selected


def build_tasks(
    rows: Sequence[dict],
    models: Sequence[str],
    conditions: Sequence[str],
    completed: set[Tuple[str, str, str]],
) -> List[Task]:
    tasks: List[Task] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        for model in models:
            for condition in conditions:
                key = (sample_id, model, condition)
                if key in completed:
                    continue
                tasks.append(Task(sample_id=sample_id, model=model, condition=condition))
    return tasks


def build_record_base(row: dict, task: Task) -> dict:
    return {
        "timestamp_utc": now_iso(),
        "sample_id": str(row["sample_id"]),
        "source_sample_id": str(row.get("source_sample_id") or row["sample_id"]),
        "source_group": str(row["source_group"]),
        "task_family": str(row["task_family"]),
        "scenario_family": str(row["scenario_family"]),
        "input_mode": str(row["input_mode"]),
        "context_path": str(row.get("context_path") or ""),
        "model": task.model,
        "runtime_model": resolve_runtime_model(task.model),
        "friendly_model_name": MODEL_FRIENDLY_NAMES.get(task.model, task.model),
        "condition": task.condition,
    }


def build_task_prompt(row: dict) -> str:
    prompt = build_eval_prompt(
        include_storyboard=False,
        scaffold_text=str(row.get("context_text") or ""),
        scaffold_label=str(row.get("context_label") or "Text context"),
    )
    if str(row.get("input_mode")) == "virtualhome_script":
        prompt += (
            "\n\nThe provided text is a simulator script rather than a natural-language description. "
            "Infer the scene state, character actions, and relevant object locations from the script only."
        )
    return prompt


def execute_task(
    task: Task,
    row: dict,
    *,
    api_key: str,
    session: requests.Session,
    max_tokens: int,
    dry_run: bool,
    mock_from_reference: bool,
) -> dict:
    record = build_record_base(row, task)
    prompt = build_task_prompt(row)
    record["prompt"] = prompt
    record["used_scaffold_text"] = str(row.get("context_text") or "")
    record["used_scaffold_label"] = str(row.get("context_label") or "Text context")
    record["image_count"] = 0

    if dry_run:
        record["status"] = "dry_run"
        record["response"] = None
        record["format_valid"] = None
        record["parsed_fields"] = None
        record["prompt_char_count"] = len(prompt)
        return record

    if mock_from_reference:
        response = render_reference_response(row)
        raw_data = None
    else:
        response, raw_data = call_openrouter(
            record["runtime_model"],
            prompt,
            api_key,
            image_paths=[],
            session=session,
            max_tokens=max_tokens,
        )
        record["raw_usage"] = raw_data.get("usage") if raw_data else None

    parsed_fields = parse_tagged_fields(response)
    format_valid = all(parsed_fields[field] for field in parsed_fields)
    if response.startswith("ERROR"):
        status = "error"
    elif format_valid:
        status = "ok"
    else:
        status = "unparsed"
    record.update(
        {
            "status": status,
            "response": response,
            "format_valid": format_valid,
            "parsed_fields": parsed_fields,
        }
    )
    return record


def worker_loop(
    *,
    task_queue: "queue.Queue[Task]",
    rows_by_id: Dict[str, dict],
    api_keys: Sequence[str],
    results_path: Path,
    results_lock: threading.Lock,
    key_counter: Dict[str, int],
    key_counter_lock: threading.Lock,
    max_tokens: int,
    dry_run: bool,
    mock_from_reference: bool,
) -> None:
    session = requests.Session()
    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            return
        row = rows_by_id[task.sample_id]
        with key_counter_lock:
            idx = key_counter["value"]
            key_counter["value"] = (idx + 1) % len(api_keys)
        api_key = api_keys[idx]
        try:
            record = execute_task(
                task,
                row,
                api_key=api_key,
                session=session,
                max_tokens=max_tokens,
                dry_run=dry_run,
                mock_from_reference=mock_from_reference,
            )
        except Exception as exc:  # noqa: BLE001
            record = build_record_base(row, task)
            record.update(
                {
                    "status": "error",
                    "response": f"{type(exc).__name__}: {str(exc)[:300]}",
                    "format_valid": False,
                    "parsed_fields": None,
                }
            )
        with results_lock:
            append_jsonl(results_path, record)
        task_queue.task_done()


def summarize_run(results_path: Path) -> dict:
    rows = read_jsonl(results_path)
    by_status = Counter(str(row.get("status", "unknown")) for row in rows)
    by_condition = Counter(str(row.get("condition")) for row in rows)
    by_model = Counter(str(row.get("friendly_model_name")) for row in rows)
    by_source = Counter(str(row.get("source_group")) for row in rows)
    return {
        "generated_at_utc": now_iso(),
        "result_count": len(rows),
        "status_counts": dict(sorted(by_status.items())),
        "condition_counts": dict(sorted(by_condition.items())),
        "model_counts": dict(sorted(by_model.items())),
        "source_group_counts": dict(sorted(by_source.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the text-backed remainder of the MindPower benchmark.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_REMAINDER_PREP_DIR / "remainder_text_eval_manifest.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REMAINDER_RUN_DIR)
    parser.add_argument("--run-name", type=str, default=f"mindpower_remainder_text_{time.strftime('%Y%m%d')}")
    parser.add_argument("--key-list-path", type=Path, default=DEFAULT_KEY_LIST)
    parser.add_argument("--extra-key-files", nargs="*", default=[])
    parser.add_argument("--conditions", nargs="+", default=SUPPORTED_CONDITIONS)
    parser.add_argument("--models", nargs="+", default=EVAL_MODELS)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--sample-ids", nargs="*", default=[])
    parser.add_argument("--parallel-workers", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-from-reference", action="store_true")
    args = parser.parse_args()

    for condition in args.conditions:
        if condition not in SUPPORTED_CONDITIONS:
            raise ValueError(f"Unsupported condition: {condition}")
    for model in args.models:
        if model not in EVAL_MODELS:
            raise ValueError(f"Unsupported evaluation model: {model}")

    rows = load_manifest_rows(args.manifest_path)
    selected_rows = select_rows(rows, args.sample_ids, args.limit_samples)
    if not selected_rows:
        raise SystemExit("No samples selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_root = args.output_dir / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    results_path = run_root / "raw_results.jsonl"
    summary_path = run_root / "run_summary.json"
    config_path = run_root / "run_config.json"
    selection_path = run_root / "selection_manifest.json"

    if args.dry_run or args.mock_from_reference:
        api_keys = ["local-mock"]
    else:
        api_keys = load_api_keys(args.key_list_path, args.extra_key_files)
    worker_count = max(1, min(args.parallel_workers, len(api_keys)))
    completed = load_completed_task_keys(results_path)
    tasks = build_tasks(selected_rows, args.models, args.conditions, completed)
    rows_by_id = {str(row["sample_id"]): row for row in selected_rows}

    config = {
        "generated_at_utc": now_iso(),
        "run_name": args.run_name,
        "manifest_path": relative_to_root(args.manifest_path),
        "output_dir": relative_to_root(run_root),
        "conditions": list(args.conditions),
        "models": list(args.models),
        "friendly_model_names": {model: MODEL_FRIENDLY_NAMES.get(model, model) for model in args.models},
        "runtime_model_aliases": {model: resolve_runtime_model(model) for model in args.models},
        "selected_sample_ids": [str(row["sample_id"]) for row in selected_rows],
        "sample_count": len(selected_rows),
        "parallel_workers": worker_count,
        "max_tokens": args.max_tokens,
        "dry_run": bool(args.dry_run),
        "mock_from_reference": bool(args.mock_from_reference),
    }
    write_json(config_path, config)
    write_json(selection_path, {"rows": selected_rows})

    task_queue: "queue.Queue[Task]" = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    print(f"Run root: {run_root}")
    print(f"Selected samples: {[row['sample_id'] for row in selected_rows]}")
    print(f"Conditions: {args.conditions}")
    print(f"Models: {[MODEL_FRIENDLY_NAMES.get(model, model) for model in args.models]}")
    print(f"Pending tasks after resume filtering: {task_queue.qsize()}")

    results_lock = threading.Lock()
    key_counter = {"value": 0}
    key_counter_lock = threading.Lock()
    threads = [
        threading.Thread(
            target=worker_loop,
            kwargs={
                "task_queue": task_queue,
                "rows_by_id": rows_by_id,
                "api_keys": api_keys,
                "results_path": results_path,
                "results_lock": results_lock,
                "key_counter": key_counter,
                "key_counter_lock": key_counter_lock,
                "max_tokens": args.max_tokens,
                "dry_run": args.dry_run,
                "mock_from_reference": args.mock_from_reference,
            },
            daemon=True,
        )
        for _ in range(worker_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    summary = summarize_run(results_path)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
