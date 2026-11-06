"""Run the optional Stage-3 E2 Ministral-8B control extension."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "evaluation" / "abstract2d")))
import evaluate as v5  # noqa: E402


OUT = ROOT / "results" / "followup" / "additional" / "e2_ministral_extension"
OUT_FILE = OUT / "results.jsonl"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    key_values = v5.load_api_keys(require=True)
    probes = v5.ensure_probe_set(50)
    probes = [probe for probe in probes if int(probe.get("stage", 0)) == 3]
    tasks = [
        task
        for task in v5.build_phase_tasks(
            "phase2", probes, ["mistralai/ministral-8b-2512"], variants=50
        )
    ]
    expected = {(str(task.probe["id"]), task.model, task.condition) for task in tasks}
    existing: set[tuple[str, str, str]] = set()
    if OUT_FILE.exists():
        for line in OUT_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            existing.add((str(row.get("probe_id")), str(row.get("model")), str(row.get("condition"))))
    pending = [task for task in tasks if (str(task.probe["id"]), task.model, task.condition) not in existing]
    print(f"Stage-3 Ministral extension: {len(tasks)} planned, {len(pending)} pending, 6 workers")
    if not pending:
        return

    task_queue: queue.Queue[Any] = queue.Queue()
    for task in pending:
        task_queue.put(task)
    write_lock = threading.Lock()
    progress = {"done": 0}

    def worker(worker_index: int) -> None:
        session = requests.Session()
        key = key_values[worker_index % len(key_values)]
        try:
            while True:
                try:
                    task = task_queue.get_nowait()
                except queue.Empty:
                    return
                try:
                    record = v5.execute_task(task, key, session=session)
                    with write_lock:
                        with OUT_FILE.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                        progress["done"] += 1
                        if progress["done"] % 25 == 0 or progress["done"] == len(pending):
                            print(f"completed {progress['done']}/{len(pending)}", flush=True)
                finally:
                    task_queue.task_done()
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(worker, index) for index in range(6)]
        for future in futures:
            future.result()
    task_queue.join()
    final_keys = set(existing)
    for line in OUT_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        final_keys.add((str(row.get("probe_id")), str(row.get("model")), str(row.get("condition"))))
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": "E2_Ministral8B_stage3_controls",
                "probe_count": len(probes),
                "models": ["mistralai/ministral-8b-2512"],
                "conditions": ["D_mismatch", "D_shuffled", "D_noise"],
                "calls_planned": len(tasks),
                "unique_keys_saved": len(final_keys),
                "complete": expected.issubset(final_keys),
                "api_key_count": len(key_values),
                "historical_results_unchanged": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
