from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import argparse
import base64
import hashlib
import json
import os
import queue
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import requests


PROBES_PATH = ROOT / "probes.json"
OUTPUT_DIR = ROOT / "results" / "followup" / "e1"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
TEMPERATURE = 0.0
MAX_TOKENS = 200
BOOTSTRAP_RESAMPLES = 10_000


MODELS = [
    "openai/gpt-4o",
    "google/gemini-2.5-pro",
    "anthropic/claude-4-sonnet-20250522",
    "qwen/qwen2.5-vl-72b-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "openai/gpt-4.1-mini-2025-04-14",
    "mistralai/ministral-8b-2512",
]

MODEL_SOURCE_SLUGS = {
    "openai/gpt-4o": "gpt-4o-2024-11-20",
    "google/gemini-2.5-pro": "gemini-2.5-pro-preview",
    "anthropic/claude-4-sonnet-20250522": "claude-sonnet-4",
    "qwen/qwen2.5-vl-72b-instruct": "qwen-2.5-vl-72b-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "llama-4-maverick",
    "openai/gpt-4.1-mini-2025-04-14": "gpt-4.1-mini",
    "mistralai/ministral-8b-2512": "ministral-8b-2512",
}

COLORS = {"blue", "red", "green", "purple", "orange", "teal", "yellow", "amber", "lime", "pink", "cyan"}
AGENT_COLORS = {"blue", "red", "green", "purple", "orange", "teal"}
OBJECT_COLORS = {"yellow", "amber", "lime", "pink", "cyan"}
CONTAINER_COLORS = {"light-blue", "coral", "brown", "mint", "violet", "peach"}

FORBIDDEN_TERMS = (
    "belief",
    "believe",
    "intention",
    "goal",
    "did not see",
    "didn't see",
    "was outside",
    "was absent",
    "moves",
    "moved",
    "places",
    "placed",
    "exits",
    "exited",
    "returns",
    "returned",
    "enters",
    "entered",
    "carries",
    "carried",
    "transfers",
    "transferred",
    "opens",
    "opened",
    "closes",
    "closed",
    "sees",
    "saw",
    "looking",
    "looks",
    "searches",
    "search",
    "knows",
)

RATE_LOCK = threading.Lock()
NEXT_ALLOWED: dict[str, float] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def read_key(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    return text.split()[-1]


def load_keys(paths: Iterable[str]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_key:
        keys.append(env_key)
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"API key file not found: {path}")
        key = read_key(path)
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    if not keys:
        raise RuntimeError("No OpenRouter API keys were supplied.")
    return keys


def split_clauses(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    return [part.strip(" .") for part in re.split(r"\.\s+", text) if part.strip(" .")]


def add_fact(facts: list[str], fact: str) -> None:
    fact = re.sub(r"\s+", " ", fact.strip(" ."))
    if fact and fact not in facts:
        facts.append(fact)


def static_facts_from_clause(clause: str) -> list[str]:
    """Extract panel-local entity facts without retaining event language."""
    clause = clause.strip(" .")
    facts: list[str] = []


    match = re.fullmatch(
        r"(?P<agent>[a-z-]+) character is in the (?P<room>left|right) room with the "
        r"(?P<object>[a-z-]+) object near the (?P<side>left|right) (?P<container>[a-z-]+) container",
        clause,
    )
    if match:
        add_fact(facts, f"{match['agent']} character is in the {match['room']} room")
        add_fact(facts, f"{match['object']} object is near the {match['side']} {match['container']} container")
        return facts

    match = re.fullmatch(
        r"(?P<agent>[a-z-]+) character and the (?P<object>[a-z-]+) object are in the (?P<room>left|right) room",
        clause,
    )
    if match:
        add_fact(facts, f"{match['agent']} character is in the {match['room']} room")
        add_fact(facts, f"{match['object']} object is in the {match['room']} room")
        return facts


    match = re.fullmatch(
        r"(?P<object>[a-z-]+) object is (?P<relation>near|in|placed in) the (?P<side>left|right) (?P<container>[a-z-]+) container",
        clause,
    )
    if match:
        relation = "near" if match["relation"] == "near" else "in"
        add_fact(facts, f"{match['object']} object is {relation} the {match['side']} {match['container']} container")
        return facts


    match = re.fullmatch(r"Only (?P<agent>[a-z-]+) character is (?:in the room|present)", clause)
    if match:
        add_fact(facts, f"{match['agent']} character is present in the room")
        return facts

    match = re.fullmatch(r"(?P<agents>[a-z-]+(?: and [a-z-]+)?) characters? (?:are|is) (?:both )?in the room", clause)
    if match:
        for agent in re.split(r" and ", match["agents"]):
            add_fact(facts, f"{agent} character is present in the room")
        return facts

    match = re.fullmatch(r"(?P<agent>[a-z-]+) character is in the (?P<room>left|right) room", clause)
    if match:
        add_fact(facts, f"{match['agent']} character is in the {match['room']} room")
        return facts

    match = re.fullmatch(r"(?P<agent>[a-z-]+) character is in the room", clause)
    if match:
        add_fact(facts, f"{match['agent']} character is present in the room")
        return facts

    match = re.fullmatch(r"(?P<agent>[a-z-]+) character remains(?: in the room)?", clause)
    if match:
        add_fact(facts, f"{match['agent']} character is present in the room")
        return facts


    match = re.match(r"(?P<agent>[a-z-]+) character (returns?|places)\b", clause)
    if match:
        add_fact(facts, f"{match['agent']} character is present in the room")
        return facts


    if "exits" not in clause and "exited" not in clause and "moves to the door" not in clause:
        for agent in sorted(AGENT_COLORS):
            if re.search(rf"\b{re.escape(agent)} character\b", clause):
                add_fact(facts, f"{agent} character is present in the room")
        for color in sorted(CONTAINER_COLORS):
            if re.search(rf"\b{re.escape(color)} (?:box|container)\b", clause):
                add_fact(facts, f"{color} container is present")
        for color in sorted(OBJECT_COLORS):
            if re.search(rf"\b{re.escape(color)} object\b", clause):
                add_fact(facts, f"{color} object is present")

    return facts


def build_static_scaffold(probe: dict[str, Any]) -> str:
    panels: list[str] = []
    for line in probe["panel_percept_lines"]:
        match = re.match(r"Panel\s+(\d+):\s*(.*)", str(line))
        if not match:
            raise ValueError(f"Unexpected panel line: {line!r}")
        panel_number, body = match.groups()
        facts: list[str] = []
        for clause in split_clauses(body):
            for fact in static_facts_from_clause(clause):
                add_fact(facts, fact)
        if not facts:
            raise ValueError(f"No static facts extracted for {probe['id']} panel {panel_number}: {line!r}")
        panels.append(f"Panel {panel_number}: " + "; ".join(facts) + ".")
    scaffold = " ".join(panels)
    lowered = normalize(scaffold)
    violations = [term for term in FORBIDDEN_TERMS if term in lowered]
    if violations:
        raise ValueError(f"Static scaffold {probe['id']} contains forbidden terms: {violations}")
    return scaffold


def build_prompt(probe: dict[str, Any], scaffold: str) -> str:
    return " ".join(
        [
            "Study this image. Here is a per-panel entity inventory:",
            scaffold,
            str(probe["task_prompt"]),
            str(probe["answer_instruction"]),
        ]
    )


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def wait_for_model(model: str) -> None:
    while True:
        with RATE_LOCK:
            ready = NEXT_ALLOWED.get(model, 0.0)
            now = time.time()
            if now >= ready:
                NEXT_ALLOWED[model] = now + 1.0
                return
            delay = min(ready - now, 5.0)
        time.sleep(max(delay, 0.05))


def call_model(model: str, prompt: str, image_path: Path, api_key: str, session: requests.Session) -> tuple[str, dict[str, Any] | None]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(image_path)}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = ""
    for attempt in range(4):
        try:
            wait_for_model(model)
            response = session.post(API_URL, headers=headers, json=payload, timeout=180)
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
                time.sleep(min(30 * (2**attempt), 180))
                continue
            response.raise_for_status()
            data = response.json()
            if data.get("error"):
                error = data["error"]
                return f"ERROR_API[{error.get('code', 'unknown')}]: {error.get('message', 'unknown error')}", data
            choices = data.get("choices") or []
            if not choices:
                return "ERROR_BAD_RESPONSE: no choices", data
            return str(choices[0]["message"].get("content", "")).strip(), data
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < 3:
                time.sleep(10 * (attempt + 1))
    return f"ERROR: {last_error[:240]}", None


def answer_correct(probe: dict[str, Any], response: str) -> bool:
    norm = normalize(response)
    kind = str(probe["answer_kind"])
    expected = str(probe["expected"])
    aliases = [str(alias).lower() for alias in probe.get("expected_aliases", [])]
    if kind == "yes_no":
        return norm.startswith(expected.lower())
    if kind == "number":
        return expected.lower() in norm[:4]
    if kind == "belief_split":
        if "belief" in norm:
            before, after = norm.split("belief", 1)
        else:
            midpoint = len(norm) // 2
            before, after = norm[:midpoint], norm[midpoint:]
        right_ok = any(alias in before for alias in ["right", "right box", "right container", str(probe.get("right_container_name", "")).lower()])
        left_ok = any(alias in after for alias in ["left", "left box", "left container", str(probe.get("left_container_name", "")).lower()])
        return right_ok and left_ok
    if kind == "choice_abc":
        return norm.startswith(expected.lower()) or any(alias in norm for alias in aliases)
    return any(alias in norm for alias in aliases)


def load_json_array(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Expected a JSON array at {path}")


def load_sharded_results(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("_format") != "sharded_json_array_v1":
        return load_json_array(path)
    rows: list[dict[str, Any]] = []
    for shard in manifest["shards"]:
        rows.extend(load_json_array(path.parent / shard))
    return rows


def load_baseline_results() -> dict[str, list[dict[str, Any]]]:
    full = load_sharded_results(ROOT / "results" / "reference" / "abstract" / "original" / "v5_phase1_results.json")
    small = load_json_array(ROOT / "results" / "reference" / "abstract" / "smalltier" / "v5_phase1_results.json")
    final: dict[str, list[dict[str, Any]]] = {}
    for model, source_slug in MODEL_SOURCE_SLUGS.items():
        source = small if source_slug == "ministral-8b-2512" else full
        final[model] = [
            row
            for row in source
            if int(row.get("stage", 0)) == 3 and str(row.get("model", "")) == source_slug
        ]
    return final


def bootstrap(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = float(np.mean(values))
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for index in range(BOOTSTRAP_RESAMPLES):
        draws[index] = float(np.mean(values[rng.integers(0, len(values), len(values))]))
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def pcr_ci(a: np.ndarray, b: np.ndarray, c: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = (float(np.mean(b)) - float(np.mean(a))) / (float(np.mean(c)) - float(np.mean(a)))
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for index in range(BOOTSTRAP_RESAMPLES):
        sample = rng.integers(0, len(a), len(a))
        denom = float(np.mean(c[sample])) - float(np.mean(a[sample]))
        draws[index] = np.nan if denom == 0 else (float(np.mean(b[sample])) - float(np.mean(a[sample]))) / denom
    return point, float(np.nanquantile(draws, 0.025)), float(np.nanquantile(draws, 0.975))


def write_analysis(rows: list[dict[str, Any]], probes: list[dict[str, Any]]) -> None:
    by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    baseline = load_baseline_results()
    per_model: list[dict[str, Any]] = []
    paired_a: list[float] = []
    paired_static: list[float] = []
    paired_c: list[float] = []
    paired_percept: list[float] = []
    paired_full: list[float] = []

    def accuracy(items: list[dict[str, Any]], condition: str) -> dict[str, Any]:
        vals = np.asarray([float(row["correct"]) for row in items if row.get("condition") == condition], dtype=float)
        point, low, high = bootstrap(vals, 100 + len(vals))
        return {"point": point, "low": low, "high": high, "n": int(len(vals))}

    for model in MODELS:
        static_rows = by_model.get(model, [])
        base_rows = baseline.get(model, [])
        combined: dict[str, list[dict[str, Any]]] = {}
        for condition in ("A", "B_percept", "B_full", "C"):
            combined[condition] = [row for row in base_rows if str(row.get("condition")) == condition]
        combined["B_static"] = static_rows
        stats = {condition: accuracy(combined[condition], condition) for condition in ("A", "B_static", "B_percept", "B_full", "C")}
        per_model.append({"model": model, "source_baseline_model": MODEL_SOURCE_SLUGS[model], "conditions": stats})
        by_id = {condition: {str(row["probe_id"]): float(row["correct"]) for row in items} for condition, items in combined.items()}
        common = sorted(set.intersection(*(set(mapping) for mapping in by_id.values())))
        paired_a.extend(by_id["A"][probe_id] for probe_id in common)
        paired_static.extend(by_id["B_static"][probe_id] for probe_id in common)
        paired_percept.extend(by_id["B_percept"][probe_id] for probe_id in common)
        paired_full.extend(by_id["B_full"][probe_id] for probe_id in common)
        paired_c.extend(by_id["C"][probe_id] for probe_id in common)

    a = np.asarray(paired_a, dtype=float)
    b_static = np.asarray(paired_static, dtype=float)
    b_percept = np.asarray(paired_percept, dtype=float)
    b_full = np.asarray(paired_full, dtype=float)
    c = np.asarray(paired_c, dtype=float)
    aggregate = {}
    for condition, values in (("A", a), ("B_static", b_static), ("B_percept", b_percept), ("B_full", b_full), ("C", c)):
        point, low, high = bootstrap(values, 1000 + len(condition))
        aggregate[condition] = {"point": point, "low": low, "high": high, "n": int(len(values))}
    aggregate["PCR_static"] = dict(zip(("point", "low", "high"), pcr_ci(a, b_static, c, 2001)))
    aggregate["PCR_percept"] = dict(zip(("point", "low", "high"), pcr_ci(a, b_percept, c, 2002)))
    aggregate["PCR_full"] = dict(zip(("point", "low", "high"), pcr_ci(a, b_full, c, 2003)))

    analysis = {
        "experiment": "E1_B_static",
        "generated_at": utc_now(),
        "definition": "Panel-local entity inventory: agent presence/room and object location only; no events, cross-panel relations, visibility/absence, or belief statements.",
        "probe_count": len(probes),
        "result_count": len(rows),
        "paired_count": int(len(a)),
        "models": MODELS,
        "aggregate": aggregate,
        "per_model": per_model,
        "baseline_sources": {model: MODEL_SOURCE_SLUGS[model] for model in MODELS},
    }
    (OUTPUT_DIR / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    lines = [
        "E1 B_static analysis",
        "",
        analysis["definition"],
        f"Paired model-probe observations: {len(a)}",
        "",
    ]
    for condition, stats in aggregate.items():
        if condition.startswith("PCR"):
            lines.append(f"{condition}: {stats['point']:.3f} [{stats['low']:.3f}, {stats['high']:.3f}]")
        else:
            lines.append(f"{condition}: {100 * stats['point']:.1f}% [{100 * stats['low']:.1f}, {100 * stats['high']:.1f}]")
    (OUTPUT_DIR / "analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    probes = [probe for probe in load_json_array(PROBES_PATH) if int(probe.get("stage", 0)) == 3]
    probes.sort(key=lambda probe: (str(probe["template"]), int(probe["variant_index"])))
    if len(probes) != 500:
        raise ValueError(f"Expected 500 Stage 3 probes, found {len(probes)}")

    scaffolds = {str(probe["id"]): build_static_scaffold(probe) for probe in probes}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "static_scaffolds.json").write_text(json.dumps(scaffolds, indent=2), encoding="utf-8")
    manifest = {
        "experiment": "E1_B_static",
        "created_at": utc_now(),
        "source_probes": str(PROBES_PATH.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(PROBES_PATH.read_bytes()).hexdigest(),
        "probe_count": len(probes),
        "models": MODELS,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "condition": "B_static",
        "scaffold_validation": {"forbidden_term_hits": 0, "scaffold_count": len(scaffolds)},
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result_path = OUTPUT_DIR / "results.jsonl"
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[(str(row["model"]), str(row["probe_id"]))] = row

    pending_by_model = {
        model: [
            probe
            for probe in probes
            if (model, str(probe["id"])) not in existing
        ]
        for model in MODELS
    }
    max_pending = max((len(items) for items in pending_by_model.values()), default=0)
    tasks = [
        (model, pending_by_model[model][index])
        for index in range(max_pending)
        for model in MODELS
        if index < len(pending_by_model[model])
    ]
    if args.dry_run:
        print(f"E1_B_static: {len(probes)} probes x {len(MODELS)} models = {len(probes) * len(MODELS)} calls")
        print(f"Pending calls: {len(tasks)}")
        print(f"Output: {OUTPUT_DIR}")
        return

    keys = load_keys(args.api_key_file)
    worker_count = min(args.parallel_workers, len(keys))
    print(f"E1_B_static: {len(probes)} probes x {len(MODELS)} models = {len(probes) * len(MODELS)} calls")
    print(f"Pending calls: {len(tasks)}; workers: {worker_count}")
    print(f"Output: {OUTPUT_DIR}")

    task_queue: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in range(worker_count):
        task_queue.put(None)
    write_lock = threading.Lock()
    completed = 0

    def worker(worker_index: int) -> None:
        nonlocal completed
        session = requests.Session()
        key = keys[worker_index % len(keys)]
        try:
            while True:
                item = task_queue.get()
                if item is None:
                    task_queue.task_done()
                    return
                model, probe = item
                scaffold = scaffolds[str(probe["id"])]
                prompt = build_prompt(probe, scaffold)
                image_path = ROOT / str(probe["image_path"])
                response, raw = call_model(model, prompt, image_path, key, session)
                row = {
                    "experiment": "E1_B_static",
                    "condition": "B_static",
                    "model": model,
                    "probe_id": str(probe["id"]),
                    "template": str(probe["template"]),
                    "stage": int(probe["stage"]),
                    "variant_index": int(probe["variant_index"]),
                    "image_path": str(probe["image_path"]),
                    "prompt": prompt,
                    "scaffold": scaffold,
                    "response": response,
                    "correct": bool(answer_correct(probe, response)),
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
                        print(f"Completed {completed}/{len(tasks)} pending calls")
                task_queue.task_done()
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(worker, index) for index in range(worker_count)]
        for future in futures:
            future.result()
    task_queue.join()

    all_rows = list(existing.values())
    all_rows.sort(key=lambda row: (MODELS.index(str(row["model"])), str(row["probe_id"])))
    write_analysis(all_rows, probes)
    print(f"Saved {len(all_rows)} records")
    print(f"Saved analysis to {OUTPUT_DIR / 'analysis.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 3 B_static scaffold ablation")
    parser.add_argument(
        "--api-key-file",
        action="append",
        required=False,
        default=[],
        help="Path to a local OpenRouter key file; repeat for multiple workers",
    )
    parser.add_argument("--parallel-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.parallel_workers < 1:
        raise SystemExit("--parallel-workers must be positive")
    run(args)


if __name__ == "__main__":
    main()
