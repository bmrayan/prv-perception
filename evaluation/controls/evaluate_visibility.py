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
from typing import Any


NEURIPS_CODE = REPO_ROOT / "evaluation" / "muma"
sys.path.insert(0, str(NEURIPS_CODE))

from evaluate_muma import (  # noqa: E402
    DEFAULT_FRAME_CACHE_DIR,
    DEFAULT_KEY_LIST,
    build_full_context_prompt,
    call_openrouter,
    extract_frame_paths,
    load_api_keys,
    parse_answer,
    resolve_runtime_model,
)


OUT_DIR = ROOT / "results" / "followup" / "derived_visibility"
PAIR_PATH = OUT_DIR / "scaffold_pairs.jsonl"
VIDEOS_DIR = BENCHMARK_ROOT / "MuMA-ToM" / "videos"
MODELS = [
    "openai/gpt-4o-2024-11-20",
    "google/gemini-2.5-pro",
    "anthropic/claude-4-sonnet-20250522",
    "qwen/qwen2.5-vl-72b-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "openai/gpt-4.1-mini",
    "mistralai/ministral-8b-2512",
]
CONDITIONS = ["B_full_vis", "B_vis_stripped"]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_pairs() -> list[dict[str, Any]]:
    rows = []
    with PAIR_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_completed(path: Path) -> set[tuple[str, str, str]]:
    completed: set[tuple[str, str, str]] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue


            if row.get("status") == "ok":
                completed.add((str(row.get("question_id")), str(row.get("model")), str(row.get("condition"))))
    return completed


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def run_one(
    pair: dict[str, Any],
    model: str,
    condition: str,
    api_key: str,
    frame_cache_dir: Path,
    frame_count: int,
) -> dict[str, Any]:
    question_id = str(pair["question_id"])
    video_id = str(pair["video_id"])
    video_path = VIDEOS_DIR / f"video_{video_id}.mp4"
    scaffold = pair["scaffold_full_vis"] if condition == "B_full_vis" else pair["scaffold_stripped"]
    prompt = build_full_context_prompt(scaffold, str(pair["question"]), with_video=True)
    image_paths = extract_frame_paths(video_path, video_id, frame_count, frame_cache_dir)
    response, raw_data = call_openrouter(
        resolve_runtime_model(model),
        prompt,
        api_key,
        image_paths=image_paths,
        max_tokens=250,
    )
    if response.strip().upper().startswith("ERROR"):
        parsed_answer = None
        status = "error"
    else:
        parsed_answer = parse_answer(response)
        status = "ok" if parsed_answer is not None else "unparsed"
    return {
        "timestamp_utc": now_iso(),
        "experiment": "E3_MuMA_derived_visibility_followup",
        "question_id": question_id,
        "video_id": video_id,
        "question_slot": str(pair["question_slot"]),
        "label": str(pair["label"]),
        "model": model,
        "runtime_model": resolve_runtime_model(model),
        "condition": condition,
        "gold_answer": str(pair["gold_answer"])[0].lower(),
        "question": pair["question"],
        "scaffold": scaffold,
        "visibility_note": pair["visibility_note"],
        "annotation_source": pair["annotation_source"],
        "prompt": prompt,
        "response": response,
        "parsed_answer": parsed_answer,
        "correct": parsed_answer == str(pair["gold_answer"])[0].lower(),
        "status": status,
        "image_count": len(image_paths),
        "raw_usage": raw_data.get("usage") if raw_data else None,
    }


def worker(
    tasks: "queue.Queue[tuple[dict[str, Any], str, str]]",
    keys: list[str],
    key_state: dict[str, int],
    key_lock: threading.Lock,
    output_path: Path,
    output_lock: threading.Lock,
    frame_cache_dir: Path,
    frame_count: int,
) -> None:
    while True:
        try:
            pair, model, condition = tasks.get_nowait()
        except queue.Empty:
            return
        try:
            row = None
            for attempt in range(3):
                with key_lock:
                    key = keys[key_state["value"] % len(keys)]
                    key_state["value"] += 1
                row = run_one(pair, model, condition, key, frame_cache_dir, frame_count)
                if row.get("status") != "error":
                    break
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
            assert row is not None
            append_jsonl(output_path, row, output_lock)
        except Exception as exc:  # noqa: BLE001
            append_jsonl(
                output_path,
                {
                    "timestamp_utc": now_iso(),
                    "experiment": "E3_MuMA_derived_visibility_followup",
                    "question_id": pair["question_id"],
                    "video_id": pair["video_id"],
                    "question_slot": pair["question_slot"],
                    "label": pair["label"],
                    "model": model,
                    "condition": condition,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                },
                output_lock,
            )
        finally:
            tasks.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paired MuMA visibility follow-up.")
    parser.add_argument("--run-name", default="muma_visibility_followup_20260727")
    parser.add_argument("--parallel-workers", type=int, default=8)
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--key-list-path", type=Path, default=DEFAULT_KEY_LIST)
    parser.add_argument("--extra-key-files", nargs="*", default=[])
    args = parser.parse_args()

    pairs = read_pairs()
    output_path = OUT_DIR / f"{args.run_name}.jsonl"
    frame_cache_dir = OUT_DIR / "frame_cache"
    frame_cache_dir.mkdir(parents=True, exist_ok=True)
    keys = load_api_keys(args.key_list_path, args.extra_key_files)
    completed = read_completed(output_path)

    tasks: "queue.Queue[tuple[dict[str, Any], str, str]]" = queue.Queue()
    pending = 0
    for pair in pairs:
        for model in MODELS:
            for condition in CONDITIONS:
                key = (str(pair["question_id"]), model, condition)
                if key not in completed:
                    tasks.put((pair, model, condition))
                    pending += 1

    manifest = {
        "experiment": "E3_MuMA_derived_visibility_followup",
        "run_name": args.run_name,
        "started_at_utc": now_iso(),
        "pair_count": len(pairs),
        "models": MODELS,
        "conditions": CONDITIONS,
        "frame_count": args.frame_count,
        "parallel_workers": args.parallel_workers,
        "key_count": len(keys),
        "pending_calls_at_start": pending,
        "completed_before_start": len(completed),
        "prompt_source": str(PAIR_PATH.relative_to(ROOT)),
    }
    (OUT_DIR / f"{args.run_name}_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Pairs: {len(pairs)}")
    print(f"Models: {len(MODELS)}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Pending calls: {pending}")
    print(f"API keys loaded: {len(keys)}")

    key_state = {"value": 0}
    key_lock = threading.Lock()
    output_lock = threading.Lock()
    threads = [
        threading.Thread(
            target=worker,
            args=(tasks, keys, key_state, key_lock, output_path, output_lock, frame_cache_dir, args.frame_count),
            daemon=True,
        )
        for _ in range(max(1, args.parallel_workers))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    manifest["finished_at_utc"] = now_iso()
    manifest["output_path"] = str(output_path.relative_to(ROOT))
    (OUT_DIR / f"{args.run_name}_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
