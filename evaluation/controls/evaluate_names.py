from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import argparse
import hashlib
import json
import queue
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import evaluate_positions as base  # noqa: E402


PROBES_PATH = ROOT / "probes.json"
OUTPUT_ROOT = ROOT / "results" / "followup" / "e1"
MODELS = base.MODELS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def identity_scaffold(probe: dict[str, Any]) -> str:
    entity_catalog = probe["entity_catalog"]
    panels: list[str] = []
    for panel_index, line in enumerate(probe["panel_percept_lines"], start=1):
        body = str(line).lower()
        facts: list[str] = []
        for kind in ("agent", "object", "container"):
            label = "character" if kind == "agent" else kind
            for color in entity_catalog.get(kind, []):
                if re.search(rf"\b{re.escape(str(color).lower())}\b", body):
                    facts.append(f"{color} {label}")
        if not facts:
            raise ValueError(f"No entities extracted for {probe['id']} panel {panel_index}")
        panels.append(f"Panel {panel_index}: " + "; ".join(facts) + ".")
    return " ".join(panels)


def build_prompt(probe: dict[str, Any], condition: str, scaffold: str | None) -> str:
    if condition == "A_fresh":
        return str(probe["question_A"])
    return " ".join(
        [
            "Study this image. Here is a panel-local entity inventory:",
            str(scaffold),
            str(probe["task_prompt"]),
            str(probe["answer_instruction"]),
        ]
    )


def run_condition(condition: str, probes: list[dict[str, Any]], scaffolds: dict[str, str], keys: list[str], workers: int) -> None:
    output_dir = OUTPUT_ROOT / ("fresh_a" if condition == "A_fresh" else "identity_only")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "E1_strengthened_controls",
        "condition": condition,
        "created_at": utc_now(),
        "source_probes": str(PROBES_PATH.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(PROBES_PATH.read_bytes()).hexdigest(),
        "probe_count": len(probes),
        "models": MODELS,
        "temperature": base.TEMPERATURE,
        "max_tokens": base.MAX_TOKENS,
        "identity_definition": "Panel-local entity names only; no positions, relations, events, visibility, absence, or beliefs.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if condition == "B_identity":
        (output_dir / "identity_scaffolds.json").write_text(json.dumps(scaffolds, indent=2), encoding="utf-8")

    result_path = output_dir / "results.jsonl"
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[(str(row["model"]), str(row["probe_id"]))] = row

    tasks = [
        (model, probe)
        for index in range(len(probes))
        for model in MODELS
        for probe in [probes[index]]
        if (model, str(probe["id"])) not in existing
    ]
    print(f"{condition}: {len(probes)} probes x {len(MODELS)} models = {len(probes) * len(MODELS)} calls; pending {len(tasks)}")
    if not tasks:
        return

    task_queue: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()
    for task in tasks:
        task_queue.put(task)
    worker_count = workers
    for _ in range(worker_count):
        task_queue.put(None)
    write_lock = threading.Lock()
    completed = 0

    def worker(worker_index: int) -> None:
        nonlocal completed
        import requests

        session = requests.Session()
        key = keys[worker_index % len(keys)]
        try:
            while True:
                item = task_queue.get()
                if item is None:
                    task_queue.task_done()
                    return
                model, probe = item
                scaffold = None if condition == "A_fresh" else scaffolds[str(probe["id"])]
                prompt = build_prompt(probe, condition, scaffold)
                response, raw = base.call_model(model, prompt, ROOT / str(probe["image_path"]), key, session)
                row = {
                    "experiment": "E1_strengthened_controls",
                    "condition": condition,
                    "model": model,
                    "probe_id": str(probe["id"]),
                    "template": str(probe["template"]),
                    "stage": int(probe["stage"]),
                    "variant_index": int(probe["variant_index"]),
                    "image_path": str(probe["image_path"]),
                    "prompt": prompt,
                    "scaffold": scaffold,
                    "response": response,
                    "correct": bool(base.answer_correct(probe, response)),
                    "expected": str(probe["expected"]),
                    "answer_kind": str(probe["answer_kind"]),
                    "timestamp": utc_now(),
                    "raw_api": raw,
                }
                with write_lock:
                    with result_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    existing[(model, str(probe["id"]))] = row
                    completed += 1
                    if completed % 25 == 0 or completed == len(tasks):
                        print(f"{condition}: completed {completed}/{len(tasks)}")
                task_queue.task_done()
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(worker, index) for index in range(worker_count)]
        for future in futures:
            future.result()
    task_queue.join()
    print(f"{condition}: saved {len(existing)} records to {result_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strengthened E1 controls")
    parser.add_argument("--api-key-file", action="append", default=[])
    parser.add_argument("--parallel-workers", type=int, default=4)
    args = parser.parse_args()
    if args.parallel_workers < 1:
        raise SystemExit("--parallel-workers must be positive")
    probes = [p for p in base.load_json_array(PROBES_PATH) if int(p.get("stage", 0)) == 3]
    probes.sort(key=lambda probe: (str(probe["template"]), int(probe["variant_index"])))
    if len(probes) != 500:
        raise ValueError(f"Expected 500 Stage 3 probes, found {len(probes)}")
    scaffolds = {str(probe["id"]): identity_scaffold(probe) for probe in probes}
    keys = base.load_keys(args.api_key_file)
    run_condition("A_fresh", probes, scaffolds, keys, args.parallel_workers)
    run_condition("B_identity", probes, scaffolds, keys, args.parallel_workers)


if __name__ == "__main__":
    main()
