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
from typing import Dict, List, Optional, Sequence, Tuple

import requests

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from mindpower.benchmark_remainder import DEFAULT_REMAINDER_PREP_DIR, DEFAULT_REMAINDER_RUN_DIR
from mindpower.common import (
    CONDITIONS,
    DEFAULT_KEY_LIST,
    EVAL_MODELS,
    MODEL_FRIENDLY_NAMES,
    PARSER_MODELS,
    ROOT,
    append_jsonl,
    build_eval_prompt,
    build_full_scaffold_text,
    build_parser_prompt,
    call_openrouter,
    load_api_keys,
    now_iso,
    parse_tagged_fields,
    read_jsonl,
    relative_to_root,
    render_reference_response,
    resolve_runtime_model,
    write_json,
)


@dataclass(frozen=True)
class Task:
    sample_id: str
    model: str
    condition: str


def load_manifest_rows(path: Path) -> List[dict]:
    rows = read_jsonl(path)
    rows.sort(key=lambda row: str(row["sample_id"]))
    return rows


def load_completed_task_keys(path: Path) -> set[Tuple[str, str, str]]:
    completed = set()
    for row in read_jsonl(path):
        completed.add((str(row["sample_id"]), str(row["model"]), str(row["condition"])))
    return completed


def load_parser_cache(path: Path) -> Dict[Tuple[str, str], str]:
    cache: Dict[Tuple[str, str], str] = {}
    for row in read_jsonl(path):
        cache[(str(row["sample_id"]), str(row["parser_condition"]))] = str(row["scaffold"])
    return cache


def select_rows(rows: Sequence[dict], sample_ids: Sequence[str], limit_samples: int, smoke: bool) -> List[dict]:
    selected = list(rows)
    if sample_ids:
        sample_set = {str(sample_id) for sample_id in sample_ids}
        selected = [row for row in selected if str(row["sample_id"]) in sample_set]
    if smoke:
        selected = selected[:3]
    elif limit_samples > 0:
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


def get_storyboard_path(row: dict) -> Path:
    return resolve_path(str(row["storyboard_path"]))


def get_scaffold_for_condition(
    *,
    row: dict,
    condition: str,
    api_key: str,
    session: requests.Session,
    parser_cache: Dict[Tuple[str, str], str],
    parser_cache_path: Path,
    parser_cache_lock: threading.Lock,
    parser_locks: Dict[Tuple[str, str], threading.Lock],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if condition == "B_percept":
        return row["description_ref"], None, None
    if condition == "B_full":
        return build_full_scaffold_text(row), None, None
    if condition == "D_mismatch":
        return row["mismatch_description_ref"], None, None
    if condition not in PARSER_MODELS:
        return None, None, None

    cache_key = (str(row["sample_id"]), condition)
    if cache_key in parser_cache:
        return parser_cache[cache_key], PARSER_MODELS[condition], resolve_runtime_model(PARSER_MODELS[condition])

    parser_lock = parser_locks.setdefault(cache_key, threading.Lock())
    with parser_lock:
        if cache_key in parser_cache:
            return parser_cache[cache_key], PARSER_MODELS[condition], resolve_runtime_model(PARSER_MODELS[condition])
        parser_model = PARSER_MODELS[condition]
        runtime_parser_model = resolve_runtime_model(parser_model)
        scaffold, raw_data = call_openrouter(
            runtime_parser_model,
            build_parser_prompt(),
            api_key,
            image_paths=[get_storyboard_path(row)],
            session=session,
            max_tokens=220,
        )
        with parser_cache_lock:
            if cache_key not in parser_cache:
                parser_cache[cache_key] = scaffold
                append_jsonl(
                    parser_cache_path,
                    {
                        "timestamp_utc": now_iso(),
                        "sample_id": str(row["sample_id"]),
                        "parser_condition": condition,
                        "parser_model": parser_model,
                        "runtime_parser_model": runtime_parser_model,
                        "scaffold": scaffold,
                        "raw_usage": raw_data.get("usage") if raw_data else None,
                    },
                )
        return scaffold, parser_model, runtime_parser_model


def build_task_prompt(
    row: dict,
    condition: str,
    scaffold_text: Optional[str],
) -> Tuple[str, List[Path], Optional[str]]:
    if condition == "A_visual":
        prompt = build_eval_prompt(include_storyboard=True, scaffold_text=None)
        return prompt, [get_storyboard_path(row)], None
    if condition in {"B_percept", "B_full", "B_auto_gpt4o", "B_auto_claude", "D_mismatch"}:
        label = {
            "B_percept": "Neutral scene description scaffold",
            "B_full": "Full reference scaffold",
            "B_auto_gpt4o": "Automatically generated neutral scene scaffold",
            "B_auto_claude": "Automatically generated neutral scene scaffold",
            "D_mismatch": "Mismatched scene description control",
        }[condition]
        prompt = build_eval_prompt(
            include_storyboard=True,
            scaffold_text=scaffold_text or "",
            scaffold_label=label,
        )
        return prompt, [get_storyboard_path(row)], label
    if condition == "C_text":
        prompt = build_eval_prompt(
            include_storyboard=False,
            scaffold_text=row["description_ref"],
            scaffold_label="Neutral scene description",
        )
        return prompt, [], "Neutral scene description"
    raise ValueError(f"Unsupported condition: {condition}")


def build_record_base(row: dict, task: Task) -> dict:
    return {
        "timestamp_utc": now_iso(),
        "sample_id": str(row["sample_id"]),
        "source_sample_id": str(row.get("source_sample_id") or row["sample_id"]),
        "source_group": str(row["source_group"]),
        "task_family": str(row["task_family"]),
        "scenario_family": str(row["scenario_family"]),
        "model": task.model,
        "runtime_model": resolve_runtime_model(task.model),
        "friendly_model_name": MODEL_FRIENDLY_NAMES.get(task.model, task.model),
        "condition": task.condition,
        "storyboard_path": str(row["storyboard_path"]),
        "mismatch_sample_id": str(row["mismatch_sample_id"]),
        "render_status": str(row.get("render_status") or "unknown"),
    }


def execute_task(
    task: Task,
    row: dict,
    *,
    api_key: str,
    session: requests.Session,
    parser_cache: Dict[Tuple[str, str], str],
    parser_cache_path: Path,
    parser_cache_lock: threading.Lock,
    parser_locks: Dict[Tuple[str, str], threading.Lock],
    max_tokens: int,
    dry_run: bool,
    mock_from_reference: bool,
) -> dict:
    record = build_record_base(row, task)
    if task.condition in PARSER_MODELS and (dry_run or mock_from_reference):
        scaffold_text = row["description_ref"] if mock_from_reference else "[parser scaffold skipped in dry-run mode]"
        parser_model = PARSER_MODELS[task.condition]
        runtime_parser_model = resolve_runtime_model(parser_model)
    else:
        scaffold_text, parser_model, runtime_parser_model = get_scaffold_for_condition(
            row=row,
            condition=task.condition,
            api_key=api_key,
            session=session,
            parser_cache=parser_cache,
            parser_cache_path=parser_cache_path,
            parser_cache_lock=parser_cache_lock,
            parser_locks=parser_locks,
        )
    prompt, image_paths, scaffold_label = build_task_prompt(row, task.condition, scaffold_text)
    record["image_count"] = len(image_paths)
    record["used_scaffold_text"] = scaffold_text
    record["used_scaffold_label"] = scaffold_label
    record["parser_model"] = parser_model
    record["runtime_parser_model"] = runtime_parser_model
    record["prompt"] = prompt

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
            image_paths=image_paths,
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
    parser_cache: Dict[Tuple[str, str], str],
    parser_cache_path: Path,
    parser_cache_lock: threading.Lock,
    parser_locks: Dict[Tuple[str, str], threading.Lock],
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
                parser_cache=parser_cache,
                parser_cache_path=parser_cache_path,
                parser_cache_lock=parser_cache_lock,
                parser_locks=parser_locks,
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
    return {
        "generated_at_utc": now_iso(),
        "result_count": len(rows),
        "status_counts": dict(sorted(by_status.items())),
        "condition_counts": dict(sorted(by_condition.items())),
        "model_counts": dict(sorted(by_model.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the storyboard-backed VirtualHome visual benchmark.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_REMAINDER_PREP_DIR / "virtualhome_visual_eval_manifest.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REMAINDER_RUN_DIR)
    parser.add_argument("--run-name", type=str, default=f"mindpower_virtualhome_visual_{time.strftime('%Y%m%d')}")
    parser.add_argument("--key-list-path", type=Path, default=DEFAULT_KEY_LIST)
    parser.add_argument("--extra-key-files", nargs="*", default=[])
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--bf-only", action="store_true", help="Run only the B_full condition.")
    parser.add_argument("--models", nargs="+", default=EVAL_MODELS)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--sample-ids", nargs="*", default=[])
    parser.add_argument("--parallel-workers", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-from-reference", action="store_true")
    args = parser.parse_args()

    if args.bf_only:
        args.conditions = ["B_full"]

    for condition in args.conditions:
        if condition not in CONDITIONS:
            raise ValueError(f"Unsupported condition: {condition}")
    for model in args.models:
        if model not in EVAL_MODELS:
            raise ValueError(f"Unsupported evaluation model: {model}")

    rows = load_manifest_rows(args.manifest_path)
    selected_rows = select_rows(rows, args.sample_ids, args.limit_samples, args.smoke)
    if not selected_rows:
        raise SystemExit("No samples selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_root = args.output_dir / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    results_path = run_root / "raw_results.jsonl"
    summary_path = run_root / "run_summary.json"
    config_path = run_root / "run_config.json"
    selection_path = run_root / "selection_manifest.json"
    parser_cache_path = run_root / "parser_scaffolds.jsonl"

    api_keys = load_api_keys(args.key_list_path, args.extra_key_files)
    worker_count = max(1, min(args.parallel_workers, len(api_keys)))
    completed = load_completed_task_keys(results_path)
    tasks = build_tasks(selected_rows, args.models, args.conditions, completed)
    rows_by_id = {str(row["sample_id"]): row for row in selected_rows}
    parser_cache = load_parser_cache(parser_cache_path)
    parser_cache_lock = threading.Lock()
    parser_locks: Dict[Tuple[str, str], threading.Lock] = {}

    config = {
        "generated_at_utc": now_iso(),
        "run_name": args.run_name,
        "manifest_path": relative_to_root(args.manifest_path),
        "output_dir": relative_to_root(run_root),
        "conditions": list(args.conditions),
        "models": list(args.models),
        "friendly_model_names": {model: MODEL_FRIENDLY_NAMES.get(model, model) for model in args.models},
        "runtime_model_aliases": {model: resolve_runtime_model(model) for model in args.models},
        "parser_models": PARSER_MODELS,
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
    if args.dry_run:
        print("Dry-run mode enabled: prompts will be constructed but not sent to the API.")
    if args.mock_from_reference:
        print("Mock-from-reference mode enabled: responses will be copied from ground truth.")

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
                "parser_cache": parser_cache,
                "parser_cache_path": parser_cache_path,
                "parser_cache_lock": parser_cache_lock,
                "parser_locks": parser_locks,
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
