from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import argparse
import base64
import json
import os
import queue
import random
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import imageio_ffmpeg
import requests
from PIL import Image


DEFAULT_QUESTIONS_PATH = BENCHMARK_ROOT / "MuMA-ToM" / "questions.json"
DEFAULT_TEXTS_PATH = BENCHMARK_ROOT / "MuMA-ToM" / "texts.json"
DEFAULT_VIDEOS_DIR = BENCHMARK_ROOT / "MuMA-ToM" / "videos"
DEFAULT_KEY_LIST = Path(__file__).resolve().parent / "default_openrouter_key_files.txt"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "generated" / "muma" / "ActualPilot"
DEFAULT_FRAME_CACHE_DIR = ROOT / "results" / "generated" / "muma" / "_frame_cache"

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = [
    "openai/gpt-4o-2024-11-20",
    "google/gemini-2.5-pro",
    "anthropic/claude-4-sonnet-20250522",
    "qwen/qwen2.5-vl-72b-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "openai/gpt-4.1-mini-2025-04-14",
    "mistralai/ministral-8b-2512",
]
MODEL_NAME_ALIASES = {
    
    "qwen/qwen-2.5-vl-7b-instruct": "mistralai/ministral-8b-2512",
}
OPENROUTER_RUNTIME_MODEL_ALIASES = {
    "anthropic/claude-4-sonnet-20250522": "anthropic/claude-sonnet-4",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "meta-llama/llama-4-maverick",
    "openai/gpt-4.1-mini-2025-04-14": "openai/gpt-4.1-mini",
}
SUPPORTED_CONDITIONS = ["A", "B_auto", "B_full", "C"]


@dataclass(frozen=True)
class Task:
    model: str
    condition: str
    question_id: str


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_runtime_model(model: str) -> str:
    return OPENROUTER_RUNTIME_MODEL_ALIASES.get(model, model)


def canonicalize_model_name(model: str) -> str:
    return MODEL_NAME_ALIASES.get(model, model)


def read_api_key_from_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return text.split()[-1].strip() or None


def load_api_keys(key_list_path: Optional[Path], extra_key_files: Sequence[str]) -> List[str]:
    keys: List[str] = []
    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_key:
        keys.append(env_key)

    candidate_items: List[Tuple[Path, str]] = []
    if key_list_path and key_list_path.exists():
        candidate_items.append((key_list_path.parent, key_list_path.read_text(encoding="utf-8")))
    for raw_path in extra_key_files:
        path = Path(raw_path)
        if path.exists():
            candidate_items.append((path.parent, path.read_text(encoding="utf-8")))
        else:
            repo_relative = ROOT / raw_path
            if repo_relative.exists():
                candidate_items.append((repo_relative.parent, repo_relative.read_text(encoding="utf-8")))

    for base_dir, content in candidate_items:
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("sk-or-"):
                keys.append(line)
                continue
            key_path = (base_dir / line).resolve()
            key = read_api_key_from_file(key_path)
            if key:
                keys.append(key)

    unique: List[str] = []
    seen = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    if not unique:
        raise RuntimeError("No OpenRouter keys found.")
    return unique


def normalize_description(text: str) -> str:
    text = text.strip()
    if text.startswith("b'") and text.endswith("'"):
        return text[2:-1].encode("utf-8").decode("unicode_escape")
    return text


def parse_choice_letter(text: str) -> Optional[str]:
    match = re.match(r"\s*([ABC])\)", text.strip(), flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    lowered = text.strip().lower()
    if lowered.startswith(("a", "b", "c")):
        return lowered[0]
    return None


def parse_answer(response: str) -> Optional[str]:
    text = response.strip().lower()
    if text.startswith("error"):
        return None
    patterns = [
        r"final answer\s*[:\-]?\s*\(?([abc])\)?\b",
        r"answer\s*[:\-]?\s*\(?([abc])\)?\b",
        r"\(([abc])\)",
        r"\b([abc])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    stripped = re.sub(r"[^abc]", "", text)
    if stripped.startswith(("a", "b", "c")):
        return stripped[0]
    return None


def select_models(count: int, explicit_models: Optional[Sequence[str]] = None) -> List[str]:
    if explicit_models:
        selected: List[str] = []
        seen = set()
        for raw_model in explicit_models:
            model = canonicalize_model_name(str(raw_model).strip())
            if not model:
                continue
            if model in seen:
                continue
            seen.add(model)
            selected.append(model)
        if not selected:
            raise ValueError("Explicit model list is empty after normalization.")
        return selected
    count = max(1, min(count, len(MODELS)))
    return MODELS[:count]


def build_examples(questions_path: Path, texts_path: Path, videos_dir: Path) -> List[dict]:
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    texts = json.loads(texts_path.read_text(encoding="utf-8"))

    examples: List[dict] = []
    for video_id in sorted(questions, key=int):
        entry = questions[video_id]
        description = normalize_description(str(texts.get(video_id) or entry.get("description", "")))
        video_path = videos_dir / f"video_{video_id}.mp4"
        for question_slot in sorted(entry["questions"], key=int):
            question_text = str(entry["questions"][question_slot]).strip()
            answer_text = str(entry["answers"][question_slot]).strip()
            answer_letter = parse_choice_letter(answer_text)
            if answer_letter is None:
                raise ValueError(f"Could not parse gold answer for video {video_id}, question {question_slot}")
            label = str(entry["labels"][question_slot]).strip()
            examples.append(
                {
                    "video_id": str(video_id),
                    "video_path": video_path,
                    "question_slot": str(question_slot),
                    "question_id": f"muma_v{int(video_id):04d}_q{int(question_slot)}",
                    "description": description,
                    "question": question_text,
                    "answer_text": answer_text,
                    "answer": answer_letter,
                    "label": label,
                }
            )
    return examples


def select_examples_by_video(examples: Sequence[dict], sample_videos: int, seed: int) -> List[dict]:
    video_ids = sorted({str(example["video_id"]) for example in examples}, key=int)
    if sample_videos <= 0 or sample_videos >= len(video_ids):
        selected_ids = set(video_ids)
    else:
        rng = random.Random(seed)
        selected_ids = set(rng.sample(video_ids, sample_videos))
    selected = [example for example in examples if str(example["video_id"]) in selected_ids]
    selected.sort(key=lambda row: (int(row["video_id"]), int(row["question_slot"])))
    return selected


def build_minimal_prompt(question: str) -> str:
    return f"{question}\nAnswer with only a, b, or c."


def build_video_only_prompt(question: str) -> str:
    return (
        "You are given sampled frames from a multi-agent household interaction in temporal order. "
        "Use only the frames to answer the question. Answer with only a, b, or c.\n\n"
        f"Question:\n{question}\n"
    )


def build_video_parser_prompt() -> str:
    return (
        "You are given sampled frames from a multi-agent household interaction in temporal order. "
        "Describe only observable facts relevant to later reasoning: agent locations, object interactions, "
        "container openings/closings, object placements, speaking turns, and temporal order. "
        "Do not infer hidden beliefs, goals, intentions, or answer any question."
    )


def build_full_context_prompt(description: str, question: str, with_video: bool) -> str:
    prefix = (
        "Use both the provided frames and the textual interaction summary to answer the multiple-choice "
        "question. Answer with only a, b, or c."
        if with_video
        else "Use the textual interaction summary to answer the multiple-choice question. Answer with only a, b, or c."
    )
    return f"{prefix}\n\nInteraction summary:\n{description}\n\nQuestion:\n{question}\n"


def build_scaffold_prompt(scaffold: str, question: str) -> str:
    return (
        "Use the automatically generated observable-action scaffold below as grounded context for the "
        "multi-agent Theory-of-Mind question. Do not add hidden beliefs or goals unless supported by the "
        "scaffold. Answer with only a, b, or c.\n\n"
        f"Scaffold:\n{scaffold}\n\n"
        f"Question:\n{question}\n"
    )


def compute_frame_indices(frame_total: int, frame_count: int) -> List[int]:
    if frame_total <= 1 or frame_count <= 1:
        return [0]
    capped_last = frame_total - 1
    indices = [
        int(round(step * capped_last / max(frame_count - 1, 1)))
        for step in range(frame_count)
    ]
    return sorted({max(0, min(capped_last, idx)) for idx in indices})


def extract_frame_paths(
    video_path: Path,
    video_id: str,
    frame_count: int,
    frame_cache_dir: Path,
) -> List[Path]:
    cache_dir = frame_cache_dir / f"video_{video_id}" / f"frames_{frame_count:02d}"
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths = [cache_dir / name for name in manifest.get("files", [])]
        if paths and all(path.exists() for path in paths):
            return paths

    cache_dir.mkdir(parents=True, exist_ok=True)
    frame_total, _ = imageio_ffmpeg.count_frames_and_secs(str(video_path))
    indices = compute_frame_indices(int(frame_total), frame_count)

    reader = imageio.get_reader(str(video_path), format="ffmpeg")
    paths: List[Path] = []
    try:
        for order, frame_idx in enumerate(indices):
            frame = reader.get_data(int(frame_idx))
            out_path = cache_dir / f"frame_{order:02d}_src{int(frame_idx):04d}.png"
            Image.fromarray(frame).save(out_path)
            paths.append(out_path)
    finally:
        reader.close()

    manifest = {
        "generated_at_utc": now_iso(),
        "video_path": str(video_path),
        "frame_count": frame_count,
        "source_indices": indices,
        "files": [path.name for path in paths],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths


def encode_image(image_path: Path) -> str:
    with image_path.open("rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def build_openrouter_content(prompt: str, image_paths: Sequence[Path]) -> List[dict]:
    content: List[dict] = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image(image_path)}",
                    "detail": "low",
                },
            }
        )
    return content


def call_openrouter(
    model: str,
    prompt: str,
    api_key: str,
    image_paths: Optional[Sequence[Path]] = None,
    session: Optional[requests.Session] = None,
    max_tokens: int = 250,
    max_retries: int = 3,
) -> Tuple[str, Optional[dict]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": build_openrouter_content(prompt, image_paths or []),
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    requester = session or requests
    for attempt in range(max_retries):
        try:
            response = requester.post(API_URL, headers=headers, json=payload, timeout=240)
            if response.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            if response.status_code >= 500:
                time.sleep(8 * (attempt + 1))
                continue
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                return f"ERROR_API: {data['error'].get('message', 'unknown error')}", data
            choices = data.get("choices") or []
            if not choices:
                return f"ERROR_BAD_RESPONSE: {json.dumps(data)[:300]}", data
            return str(choices[0]["message"]["content"]).strip(), data
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries - 1:
                return f"ERROR: {str(exc)[:180]}", None
            time.sleep(5 * (attempt + 1))
    return "ERROR: max retries", None


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_completed_task_keys(path: Path) -> set[Tuple[str, str, str]]:
    completed = set()
    for row in read_jsonl(path):
        completed.add((str(row["model"]), str(row["condition"]), str(row["question_id"])))
    return completed


def load_scaffold_cache(path: Path, parser_model: str) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    for row in read_jsonl(path):
        if str(row.get("parser_model")) == parser_model:
            cache[str(row["video_id"])] = str(row["scaffold"])
    return cache


def _safe_pcr(a: float, b: float, c: float) -> Optional[float]:
    denom = c - a
    if abs(denom) < 1e-12:
        return None
    return (b - a) / denom


def _accuracy(block: Dict[str, int]) -> float:
    return block["correct"] / block["total"] if block["total"] else 0.0


def summarize_results(results_path: Path) -> dict:
    rows = read_jsonl(results_path)
    overall = {"correct": 0, "total": 0}
    by_condition: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    by_model_condition: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    by_label_condition: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    for row in rows:
        if str(row.get("status")) != "ok":
            continue
        correct = int(bool(row.get("correct")))
        condition = str(row["condition"])
        model = str(row["model"])
        label = str(row["label"])
        overall["total"] += 1
        overall["correct"] += correct
        by_condition[condition]["total"] += 1
        by_condition[condition]["correct"] += correct
        by_model_condition[(model, condition)]["total"] += 1
        by_model_condition[(model, condition)]["correct"] += correct
        by_label_condition[(label, condition)]["total"] += 1
        by_label_condition[(label, condition)]["correct"] += correct

    by_condition_payload = {
        condition: {**block, "accuracy": _accuracy(block)}
        for condition, block in sorted(by_condition.items())
    }
    condition_accuracy = {condition: payload["accuracy"] for condition, payload in by_condition_payload.items()}
    summary = {
        "generated_at_utc": now_iso(),
        "experiment_set": "MuMA-ToM pilot",
        "overall": {**overall, "accuracy": _accuracy(overall)},
        "by_condition": by_condition_payload,
        "pcr": {
            "B_auto": _safe_pcr(condition_accuracy.get("A", 0.0), condition_accuracy.get("B_auto", 0.0), condition_accuracy.get("C", 0.0))
            if {"A", "B_auto", "C"}.issubset(condition_accuracy)
            else None,
            "B_full": _safe_pcr(condition_accuracy.get("A", 0.0), condition_accuracy.get("B_full", 0.0), condition_accuracy.get("C", 0.0))
            if {"A", "B_full", "C"}.issubset(condition_accuracy)
            else None,
        },
        "by_model_condition": {
            f"{model}::{condition}": {**block, "accuracy": _accuracy(block)}
            for (model, condition), block in sorted(by_model_condition.items())
        },
        "by_label_condition": {
            f"{label}::{condition}": {**block, "accuracy": _accuracy(block)}
            for (label, condition), block in sorted(by_label_condition.items())
        },
    }
    return summary


def summary_lines(summary: dict) -> List[str]:
    def fmt(value: Optional[float]) -> str:
        return "nan" if value is None else f"{value:.3f}"

    lines = [
        summary["experiment_set"],
        "",
        f"Overall: {summary['overall']['correct']}/{summary['overall']['total']} = {100.0 * summary['overall']['accuracy']:.1f}%",
        "",
        "By condition",
    ]
    for condition, block in summary["by_condition"].items():
        lines.append(f"  {condition}: {block['correct']}/{block['total']} = {100.0 * block['accuracy']:.1f}%")
    lines.extend(
        [
            "",
            f"PCR: B_auto={fmt(summary['pcr']['B_auto'])}, B_full={fmt(summary['pcr']['B_full'])}",
            "",
            "By label and condition",
        ]
    )
    for key, block in summary["by_label_condition"].items():
        lines.append(f"  {key}: {block['correct']}/{block['total']} = {100.0 * block['accuracy']:.1f}%")
    return lines


def build_tasks(
    examples: Sequence[dict],
    models: Sequence[str],
    conditions: Sequence[str],
    completed: set[Tuple[str, str, str]],
) -> List[Task]:
    tasks: List[Task] = []
    for example in examples:
        for model in models:
            for condition in conditions:
                key = (model, condition, str(example["question_id"]))
                if key not in completed:
                    tasks.append(Task(model=model, condition=condition, question_id=str(example["question_id"])))
    return tasks


def execute_task(
    task: Task,
    example: dict,
    api_key: str,
    videos_dir: Path,
    frame_count: int,
    frame_cache_dir: Path,
    parser_model: str,
    parser_cache: Dict[str, str],
    parser_cache_path: Path,
    parser_cache_lock: threading.Lock,
    session: requests.Session,
) -> dict:
    runtime_model = resolve_runtime_model(task.model)
    runtime_parser_model = resolve_runtime_model(parser_model)
    record = {
        "timestamp_utc": now_iso(),
        "question_id": str(example["question_id"]),
        "video_id": str(example["video_id"]),
        "question_slot": str(example["question_slot"]),
        "label": str(example["label"]),
        "model": task.model,
        "runtime_model": runtime_model,
        "condition": task.condition,
        "gold_answer": str(example["answer"]),
    }

    question = str(example["question"])
    description = str(example["description"])
    video_path = videos_dir / f"video_{example['video_id']}.mp4"
    image_paths: List[Path] = []

    if task.condition == "C":
        prompt = build_full_context_prompt(description, question, with_video=False)
    else:
        image_paths = extract_frame_paths(video_path, str(example["video_id"]), frame_count, frame_cache_dir)
        if task.condition == "A":
            prompt = build_video_only_prompt(question)
        elif task.condition == "B_full":
            prompt = build_full_context_prompt(description, question, with_video=True)
        elif task.condition == "B_auto":
            video_key = str(example["video_id"])
            scaffold = parser_cache.get(video_key)
            if scaffold is None:
                parser_prompt = build_video_parser_prompt()
                scaffold, _ = call_openrouter(
                    runtime_parser_model,
                    parser_prompt,
                    api_key,
                    image_paths=image_paths,
                    session=session,
                    max_tokens=450,
                )
                with parser_cache_lock:
                    if video_key not in parser_cache:
                        parser_cache[video_key] = scaffold
                        append_jsonl(
                            parser_cache_path,
                            {
                                "timestamp_utc": now_iso(),
                                "video_id": video_key,
                                "parser_model": parser_model,
                                "runtime_parser_model": runtime_parser_model,
                                "scaffold": scaffold,
                            },
                        )
            prompt = build_scaffold_prompt(parser_cache.get(video_key, ""), question)
        else:
            raise ValueError(f"Unsupported condition: {task.condition}")

    response, raw_data = call_openrouter(
        runtime_model,
        prompt,
        api_key,
        image_paths=image_paths,
        session=session,
        max_tokens=250,
    )
    if response.strip().upper().startswith("ERROR"):
        parsed_answer = None
        status = "error"
    else:
        parsed_answer = parse_answer(response)
        status = "ok" if parsed_answer is not None else "unparsed"
    record.update(
        {
            "status": status,
            "response": response,
            "parsed_answer": parsed_answer,
            "correct": parsed_answer == str(example["answer"]),
            "question": question,
            "description": description if task.condition in {"B_full", "C"} else None,
            "image_count": len(image_paths),
            "raw_usage": raw_data.get("usage") if raw_data else None,
        }
    )
    return record


def worker_loop(
    task_queue: "queue.Queue[Task]",
    example_index: Dict[str, dict],
    api_keys: Sequence[str],
    videos_dir: Path,
    frame_count: int,
    frame_cache_dir: Path,
    parser_model: str,
    parser_cache: Dict[str, str],
    parser_cache_path: Path,
    parser_cache_lock: threading.Lock,
    results_path: Path,
    results_lock: threading.Lock,
    key_counter: Dict[str, int],
    key_counter_lock: threading.Lock,
) -> None:
    session = requests.Session()
    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            return
        with key_counter_lock:
            idx = key_counter["value"]
            key_counter["value"] = (idx + 1) % len(api_keys)
        api_key = api_keys[idx]
        example = example_index[task.question_id]
        try:
            record = execute_task(
                task,
                example,
                api_key,
                videos_dir,
                frame_count,
                frame_cache_dir,
                parser_model,
                parser_cache,
                parser_cache_path,
                parser_cache_lock,
                session,
            )
        except Exception as exc:  # noqa: BLE001
            record = {
                "timestamp_utc": now_iso(),
                "question_id": task.question_id,
                "video_id": str(example["video_id"]),
                "question_slot": str(example["question_slot"]),
                "label": str(example["label"]),
                "model": task.model,
                "runtime_model": resolve_runtime_model(task.model),
                "condition": task.condition,
                "gold_answer": str(example["answer"]),
                "status": "error",
                "response": f"{type(exc).__name__}: {str(exc)[:240]}",
            }
        with results_lock:
            append_jsonl(results_path, record)
        task_queue.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a MuMA-ToM pilot with MMToM-style conditions.")
    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--texts-path", type=Path, default=DEFAULT_TEXTS_PATH)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frame-cache-dir", type=Path, default=DEFAULT_FRAME_CACHE_DIR)
    parser.add_argument("--run-name", type=str, default=f"muma_tom_pilot_{time.strftime('%Y%m%d')}")
    parser.add_argument("--sample-videos", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--model-count", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=None, help="Explicit model IDs to run; overrides --model-count.")
    parser.add_argument("--parallel-workers", type=int, default=3)
    parser.add_argument("--parser-model", type=str, default="openai/gpt-4o-2024-11-20")
    parser.add_argument("--conditions", nargs="+", default=SUPPORTED_CONDITIONS)
    parser.add_argument("--key-list-path", type=Path, default=DEFAULT_KEY_LIST)
    parser.add_argument("--extra-key-files", nargs="*", default=[])
    args = parser.parse_args()

    conditions = []
    for condition in args.conditions:
        if condition not in SUPPORTED_CONDITIONS:
            raise ValueError(f"Unsupported condition: {condition}")
        if condition not in conditions:
            conditions.append(condition)

    examples = build_examples(args.questions_path, args.texts_path, args.videos_dir)
    selected = select_examples_by_video(examples, args.sample_videos, args.seed)
    selected_video_ids = sorted({str(example["video_id"]) for example in selected}, key=int)
    selected_models = select_models(args.model_count, args.models)
    api_keys = load_api_keys(args.key_list_path, args.extra_key_files)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.frame_cache_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / f"{args.run_name}.jsonl"
    summary_json_path = args.output_dir / f"{args.run_name}_summary.json"
    summary_txt_path = args.output_dir / f"{args.run_name}_summary.txt"
    config_path = args.output_dir / f"{args.run_name}_config.json"
    parser_cache_path = args.output_dir / f"{args.run_name}_parser_cache.jsonl"

    config = {
        "generated_at_utc": now_iso(),
        "run_name": args.run_name,
        "dataset": "MuMA-ToM",
        "sample_videos": args.sample_videos,
        "selected_video_ids": selected_video_ids,
        "question_count": len(selected),
        "frame_count": args.frame_count,
        "conditions": conditions,
        "models": selected_models,
        "runtime_model_aliases": {
            model: resolve_runtime_model(model)
            for model in selected_models + [args.parser_model]
        },
        "parser_model": args.parser_model,
        "seed": args.seed,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    completed = load_completed_task_keys(results_path)
    parser_cache = load_scaffold_cache(parser_cache_path, args.parser_model)
    tasks = build_tasks(selected, selected_models, conditions, completed)
    example_index = {str(example["question_id"]): example for example in selected}

    print(f"Selected {len(selected_video_ids)} videos, {len(selected)} questions")
    print(f"Models: {selected_models}")
    print(f"Conditions: {conditions}")
    print(f"Pending tasks: {len(tasks)}")

    task_queue: "queue.Queue[Task]" = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    results_lock = threading.Lock()
    parser_cache_lock = threading.Lock()
    key_counter = {"value": 0}
    key_counter_lock = threading.Lock()
    threads = []
    for _ in range(max(1, args.parallel_workers)):
        thread = threading.Thread(
            target=worker_loop,
            args=(
                task_queue,
                example_index,
                api_keys,
                args.videos_dir,
                args.frame_count,
                args.frame_cache_dir,
                args.parser_model,
                parser_cache,
                parser_cache_path,
                parser_cache_lock,
                results_path,
                results_lock,
                key_counter,
                key_counter_lock,
            ),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    summary = summarize_results(results_path)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_txt_path.write_text("\n".join(summary_lines(summary)) + "\n", encoding="utf-8")

    print("\n".join(summary_lines(summary)))
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
