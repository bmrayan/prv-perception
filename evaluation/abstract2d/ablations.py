
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path

import argparse
import base64
import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import requests


CONFIG = {


    "probes_path": str(ROOT / "probes.json"),


    "api_url": "https://openrouter.ai/api/v1/chat/completions",


    "keys_path": "keys.txt",


    "eval_models": [
        "openai/gpt-4o-2024-11-20",
        "google/gemini-2.5-pro-preview",
        "anthropic/claude-sonnet-4",
        "qwen/qwen-2.5-vl-72b-instruct",
        "meta-llama/llama-4-maverick",
        "openai/gpt-4.1-mini",
        
    ],


    "parser_model": "openai/gpt-4o-2024-11-20",


    "temperature": 0.0,
    "max_tokens": 200,
    "max_tokens_parser": 600,  


    "max_workers": 15,
    "exp3_max_workers": 4,
    "min_model_interval": 2.5,
    "retry_jitter_max": 3.0,


    "output_dir": str(ROOT / "results/generated/ablations"),


    "max_retries": 3,
    "backoff_base": 30,
}


def read_api_key_from_file(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return ""
    return text.split()[-1].strip()


def load_keys(path: str) -> list[str]:
    keys: list[str] = [os.environ["OPENROUTER_API_KEY"]] if os.environ.get("OPENROUTER_API_KEY") else []
    seen: set[str] = set(keys)
    if not Path(path).exists():
        if keys:
            return keys
        raise SystemExit("Set OPENROUTER_API_KEY or pass --keys.")
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            key = line
            if not line.startswith("sk-or-") and Path(line).exists():
                key = read_api_key_from_file(line)
            if not key or key in seen:
                continue
            seen.add(key)
            keys.append(key)
    return keys


def slugify_model(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")


def interleave_by_model(
    tasks_by_model: dict[str, list[tuple]],
    model_order: list[str],
) -> list[tuple]:
    """Round-robin tasks across models to avoid bursty same-model traffic."""
    interleaved: list[tuple] = []
    max_len = max((len(tasks) for tasks in tasks_by_model.values()), default=0)
    for idx in range(max_len):
        for model in model_order:
            tasks = tasks_by_model.get(model, [])
            if idx < len(tasks):
                interleaved.append(tasks[idx])
    return interleaved


MODEL_RATE_LOCK = threading.Lock()
MODEL_NEXT_ALLOWED: dict[str, float] = defaultdict(float)


def wait_for_model_slot(model: str):
    """Prevent bursty same-model traffic across threads."""
    while True:
        with MODEL_RATE_LOCK:
            now = time.time()
            ready_at = MODEL_NEXT_ALLOWED[model]
            if now >= ready_at:
                return
            sleep_for = max(0.05, min(ready_at - now, 5.0))
        time.sleep(sleep_for)


def push_model_cooldown(model: str, delay_s: float):
    with MODEL_RATE_LOCK:
        MODEL_NEXT_ALLOWED[model] = max(
            MODEL_NEXT_ALLOWED[model],
            time.time() + max(delay_s, 0.0),
        )


def load_probes(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        raw_probes = json.load(f)

    probes: list[dict] = []
    for probe in raw_probes:
        normalized = dict(probe)
        normalized["image_path"] = str(Path(path).resolve().parent / probe["image_path"])
        normalized["probe_id"] = probe.get("probe_id", probe.get("id"))
        normalized["expected_answer"] = probe.get("expected_answer", probe.get("expected"))
        normalized["answer_type"] = probe.get("answer_type", probe.get("answer_kind"))
        normalized["question"] = probe.get("question", probe.get("question_A", probe.get("task_prompt", "")))
        normalized["text_only_narration"] = probe.get(
            "text_only_narration", probe.get("text_narration", "")
        )
        probes.append(normalized)
    return probes


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def api_call(
    api_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 200,
) -> str:
    """Single API call with retry logic."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(CONFIG["max_retries"]):
        try:
            wait_for_model_slot(model)
            resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = max(float(retry_after), 1.0)
                    except ValueError:
                        wait = CONFIG["backoff_base"] * (2 ** attempt)
                else:
                    wait = CONFIG["backoff_base"] * (2 ** attempt)
                wait += random.uniform(0.0, CONFIG["retry_jitter_max"])
                push_model_cooldown(model, wait)
                print(f"  429 rate-limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                wait = 15 + random.uniform(0.0, CONFIG["retry_jitter_max"])
                push_model_cooldown(model, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            push_model_cooldown(model, CONFIG["min_model_interval"])
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == CONFIG["max_retries"] - 1:
                return f"ERROR: {e}"
            wait = 10 + random.uniform(0.0, CONFIG["retry_jitter_max"])
            push_model_cooldown(model, wait)
            time.sleep(wait)
    return "ERROR: max retries exceeded"


def grade_response(response: str, probe: dict) -> bool:
    """Deterministic grader matching main paper (§A.4)."""
    resp = response.lower().strip()
    resp = " ".join(resp.split())  
    atype = probe["answer_type"]
    expected = probe["expected_answer"].lower()
    aliases = [a.lower() for a in probe.get("expected_aliases", [expected])]

    if atype == "yes_no":
        return resp.startswith(expected)
    elif atype == "number":
        return expected in resp[:4]
    elif atype == "left_right":
        return any(a in resp for a in aliases)
    elif atype == "belief_split":
        split_idx = resp.find("belief")
        if split_idx == -1:
            split_idx = len(resp) // 2
        reality_aliases = [a.lower() for a in probe.get("actual_aliases", [])]
        belief_aliases = aliases
        reality_part = resp[:split_idx]
        belief_part = resp[split_idx:]
        return (
            any(a in reality_part for a in reality_aliases)
            and any(a in belief_part for a in belief_aliases)
        )
    else:
        return any(a in resp for a in aliases)


VLM_PARSER_SYSTEM = (
    "You are a precise visual scene describer. You will see a multi-panel "
    "image showing colored geometric shapes (circles, rectangles) on a white "
    "background. Panels are arranged left-to-right and separated by thin "
    "gray dividers.\n\n"
    "For EACH panel (numbered 1, 2, 3, ... from left to right), describe:\n"
    "- Which colored circles (characters/agents) are present and their "
    "approximate positions (left/right, upper/lower)\n"
    "- Which smaller colored circles (objects) are present and their positions\n"
    "- Which colored rectangles (containers) are present and their positions\n"
    "- Any changes from the previous panel: who entered, who exited, "
    "what moved\n\n"
    "Be precise about colors. Use the format:\n"
    "Panel N: [description]\n"
    "Changes: [what changed from Panel N-1]\n\n"
    "Do NOT interpret the scene or make inferences about beliefs or "
    "intentions. Only describe what you literally see in each panel."
)


def generate_vlm_scaffold(probe: dict, api_key: str, parser_model: str) -> str:
    """Use parser VLM to generate a scene description scaffold."""
    img_b64 = encode_image(probe["image_path"])
    messages = [
        {"role": "system", "content": VLM_PARSER_SYSTEM},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}"
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Describe each panel of this multi-panel image. "
                        "List all colored shapes, their types (large circle = "
                        "character, small circle = object, rectangle = container), "
                        "colors, and positions. Note any changes between "
                        "consecutive panels."
                    ),
                },
            ],
        },
    ]
    return api_call(
        CONFIG["api_url"],
        api_key,
        parser_model,
        messages,
        temperature=0.0,
        max_tokens=CONFIG["max_tokens_parser"],
    )


def build_vlm_scaffold_prompt(probe: dict, vlm_scaffold: str) -> list[dict]:
    """Build evaluation prompt using VLM-generated scaffold."""
    img_b64 = encode_image(probe["image_path"])
    text = (
        f"Study this image. Here is a structured scene description generated "
        f"by a visual analysis system:\n\n{vlm_scaffold}\n\n"
        f"{probe['question']}"
    )
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                },
                {"type": "text", "text": text},
            ],
        }
    ]


def run_experiment_vlm_parser(
    probes: list[dict],
    keys: list[str],
    parser_model: str | None = None,
    output_subdir: str | None = None,
):
    """
    Experiment 1: Generate VLM scaffolds, then evaluate all models.

    Phase 1: Parse all Stage 3 probes with GPT-4o (500 calls)
    Phase 2: Evaluate all 7 models with B_vlm (3,500 calls)
    Total: ~4,000 calls
    """
    parser_model = parser_model or CONFIG["parser_model"]
    parser_slug = slugify_model(parser_model)
    if output_subdir is None:
        if parser_model == CONFIG["parser_model"]:
            output_subdir = "exp1_vlm_parser"
        else:
            output_subdir = f"exp1_vlm_parser__{parser_slug}"
    out_dir = Path(CONFIG["output_dir"]) / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "experiment": "vlm_parser",
                "parser_model": parser_model,
                "parser_slug": parser_slug,
                "eval_models": CONFIG["eval_models"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    stage3 = sorted([p for p in probes if p["stage"] == 3], key=lambda p: p["probe_id"])
    scaffold_path = out_dir / "vlm_scaffolds.json"


    if scaffold_path.exists():
        print(f"Loading cached VLM scaffolds from {scaffold_path}")
        with open(scaffold_path) as f:
            vlm_scaffolds = json.load(f)
    else:
        print(f"Phase 1: Generating VLM scaffolds for {len(stage3)} probes...")
        vlm_scaffolds = {}
        key_cycle = keys * (len(stage3) // len(keys) + 1)

        with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as pool:
            futures = {}
            for i, probe in enumerate(stage3):
                fut = pool.submit(
                    generate_vlm_scaffold, probe, key_cycle[i], parser_model
                )
                futures[fut] = probe["probe_id"]

            for fut in as_completed(futures):
                pid = futures[fut]
                vlm_scaffolds[pid] = fut.result()
                if len(vlm_scaffolds) % 25 == 0:
                    with open(scaffold_path, "w", encoding="utf-8") as f:
                        json.dump(vlm_scaffolds, f)
                if len(vlm_scaffolds) % 50 == 0:
                    print(f"  Parsed {len(vlm_scaffolds)}/{len(stage3)}")

        with open(scaffold_path, "w", encoding="utf-8") as f:
            json.dump(vlm_scaffolds, f)
        print(f"  Saved {len(vlm_scaffolds)} scaffolds to {scaffold_path}")


    print(f"Phase 2: Evaluating {len(CONFIG['eval_models'])} models...")
    results_path = out_dir / "results.jsonl"


    existing = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                r = json.loads(line)
                existing.add((r["model"], r["probe_id"]))

    tasks = []
    for model in CONFIG["eval_models"]:
        for probe in stage3:
            pid = probe["probe_id"]
            if (model, pid) in existing:
                continue
            if pid not in vlm_scaffolds:
                continue
            tasks.append((model, probe, vlm_scaffolds[pid]))

    print(f"  {len(tasks)} calls to make ({len(existing)} cached)")
    def eval_one(idx_args):
        idx, (model, probe, scaffold) = idx_args
        key = keys[idx % len(keys)]
        messages = build_vlm_scaffold_prompt(probe, scaffold)
        response = api_call(
            CONFIG["api_url"], key, model, messages,
            CONFIG["temperature"], CONFIG["max_tokens"]
        )
        correct = grade_response(response, probe)
        return {
            "model": model,
            "parser_model": parser_model,
            "probe_id": probe["probe_id"],
            "template": probe["template"],
            "condition": "B_vlm",
            "response": response,
            "correct": correct,
        }

    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as pool:
        futs = [pool.submit(eval_one, (i, t)) for i, t in enumerate(tasks)]
        with open(results_path, "a") as f:
            for i, fut in enumerate(as_completed(futs)):
                result = fut.result()
                f.write(json.dumps(result) + "\n")
                f.flush()
                if (i + 1) % 100 == 0:
                    print(f"  Completed {i+1}/{len(tasks)}")

    print(f"  Done. Results in {results_path}")


VISIBILITY_PATTERNS = [
    r"[^.]*\b(was absent|is absent)\b[^.]*\.",
    r"[^.]*\b(did not see|did not observe|does not see|does not observe)\b[^.]*\.",
    r"[^.]*\b(was not present|is not present)\b[^.]*\.",
    r"[^.]*\b(unaware of|has no knowledge of)\b[^.]*\.",
    r"[^.]*\b(cannot see|could not see)\b[^.]*\.",
    r"[^.]*\b(left the room|left the scene|exited the room|exited the scene)\b[^.]*\.",
    r"[^.]*\b(not in the room|outside the room)\b[^.]*\.",
    r"[^.]*\b(while .* was gone|while .* was away)\b[^.]*\.",
    r"[^.]*\b(without .* knowing|without .* seeing)\b[^.]*\.",
]


def strip_visibility(scaffold_full: str) -> str:
    """
    Remove sentences containing explicit visibility/absence predicates
    from B_full scaffold, producing B_vis_stripped.

    Keeps: entity positions, colors, container contents, movements
    Removes: "X was absent and did not observe...", "while X was gone...", etc.
    """
    result = scaffold_full
    for pattern in VISIBILITY_PATTERNS:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    
    result = re.sub(r"\n\s*\n", "\n", result)
    result = re.sub(r"  +", " ", result)
    return result.strip()


def build_vis_stripped_prompt(probe: dict, scaffold_stripped: str) -> list[dict]:
    """Build prompt for B_vis_stripped condition."""
    img_b64 = encode_image(probe["image_path"])
    text = (
        f"Study this image. Here is a structured scene description:\n\n"
        f"{scaffold_stripped}\n\n"
        f"{probe['question']}"
    )
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                },
                {"type": "text", "text": text},
            ],
        }
    ]


def run_experiment_vis_stripped(probes: list[dict], keys: list[str]):
    """
    Experiment 2: B_full minus visibility statements.
    Total: 7 models × 500 probes = 3,500 calls
    """
    out_dir = Path(CONFIG["output_dir"]) / "exp2_vis_stripped"
    out_dir.mkdir(parents=True, exist_ok=True)

    stage3 = [p for p in probes if p["stage"] == 3]


    stripped_log = {}
    for probe in stage3:
        original = probe["scaffold_full"]
        stripped = strip_visibility(original)
        stripped_log[probe["probe_id"]] = {
            "original_len": len(original),
            "stripped_len": len(stripped),
            "chars_removed": len(original) - len(stripped),
            "stripped_scaffold": stripped,
        }

    with open(out_dir / "strip_log.json", "w") as f:
        json.dump(stripped_log, f, indent=2)


    removals = [v["chars_removed"] for v in stripped_log.values()]
    print(f"Visibility stripping: mean {np.mean(removals):.0f} chars removed "
          f"(min {min(removals)}, max {max(removals)})")

    results_path = out_dir / "results.jsonl"
    existing = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                r = json.loads(line)
                existing.add((r["model"], r["probe_id"]))

    tasks = []
    for model in CONFIG["eval_models"]:
        for probe in stage3:
            pid = probe["probe_id"]
            if (model, pid) in existing:
                continue
            tasks.append((model, probe, stripped_log[pid]["stripped_scaffold"]))

    print(f"Experiment 2: {len(tasks)} calls ({len(existing)} cached)")
    key_cycle = keys * (len(tasks) // len(keys) + 1)

    def eval_one(idx_args):
        idx, (model, probe, scaffold) = idx_args
        key = keys[idx % len(keys)]
        messages = build_vis_stripped_prompt(probe, scaffold)
        response = api_call(
            CONFIG["api_url"], key, model, messages,
            CONFIG["temperature"], CONFIG["max_tokens"]
        )
        correct = grade_response(response, probe)
        return {
            "model": model,
            "probe_id": probe["probe_id"],
            "template": probe["template"],
            "condition": "B_vis_stripped",
            "response": response,
            "correct": correct,
        }

    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as pool:
        futs = [pool.submit(eval_one, (i, t)) for i, t in enumerate(tasks)]
        with open(results_path, "a") as f:
            for i, fut in enumerate(as_completed(futs)):
                result = fut.result()
                f.write(json.dumps(result) + "\n")
                f.flush()
                if (i + 1) % 100 == 0:
                    print(f"  Completed {i+1}/{len(tasks)}")

    print(f"  Done. Results in {results_path}")


def paraphrase_narration_restructured(narration: str) -> str:
    """Restructure narration: reverse panel order in text."""
    sentences = [s.strip() for s in narration.split(".") if s.strip()]
    
    panels = defaultdict(list)
    other = []
    for s in sentences:
        match = re.match(r"(Panel \d+)", s, re.IGNORECASE)
        if match:
            panels[match.group(1)].append(s)
        else:
            other.append(s)
    
    reversed_panels = []
    for key in sorted(panels.keys(), reverse=True):
        reversed_panels.extend(panels[key])
    result = ". ".join(reversed_panels + other) + "."
    return result


def paraphrase_narration_vocabulary(narration: str) -> str:
    """Synonym substitution on narration."""
    subs = {
        "character": "figure",
        "container": "box",
        "object": "item",
        "places": "puts",
        "moves": "transfers",
        "exits": "leaves",
        "returns": "comes back",
        "enters": "goes into",
        "left side": "left portion",
        "right side": "right portion",
        "is located": "sits",
    }
    result = narration
    for old, new in subs.items():
        result = re.sub(re.escape(old), new, result, flags=re.IGNORECASE)
    return result


def paraphrase_narration_adversarial(narration: str) -> str:
    """
    Adversarial ToM modification (Ullman-style): insert a misleading
    sentence that would confuse a model relying on surface patterns.
    """
    
    sentences = narration.split(".", 1)
    adversarial_insert = (
        " Note that all characters could potentially see everything "
        "that happens in the scene"
    )
    if len(sentences) > 1:
        return sentences[0] + "." + adversarial_insert + "." + sentences[1]
    return narration + adversarial_insert + "."


PARAPHRASE_VARIANTS = {
    "restructured": paraphrase_narration_restructured,
    "vocabulary": paraphrase_narration_vocabulary,
    "adversarial": paraphrase_narration_adversarial,
}


def build_condition_c_prompt(probe: dict, narration: str) -> list[dict]:
    """Text-only prompt (Condition C) with potentially paraphrased narration."""
    return [
        {
            "role": "user",
            "content": f"{narration}\n\n{probe['question']}",
        }
    ]


def build_condition_bfull_paraphrase_prompt(
    probe: dict, scaffold: str
) -> list[dict]:
    """B_full with paraphrased scaffold."""
    img_b64 = encode_image(probe["image_path"])
    text = (
        f"Study this image. Here is a structured scene description:\n\n"
        f"{scaffold}\n\n"
        f"{probe['question']}"
    )
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                },
                {"type": "text", "text": text},
            ],
        }
    ]


def run_experiment_paraphrases(probes: list[dict], keys: list[str]):
    """
    Experiment 3: Paraphrase robustness for Conditions C and B_full.

    3 paraphrase variants × 2 conditions × 7 models × 500 probes
    = 21,000 calls

    With 15 keys this is ~1,400 calls/key, very manageable.
    """
    out_dir = Path(CONFIG["output_dir"]) / "exp3_paraphrases"
    out_dir.mkdir(parents=True, exist_ok=True)

    stage3 = [p for p in probes if p["stage"] == 3]

    results_path = out_dir / "results.jsonl"
    existing = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                r = json.loads(line)
                existing.add((r["model"], r["probe_id"], r["condition"]))

    tasks_by_model = {model: [] for model in CONFIG["eval_models"]}
    for variant_name, variant_fn in PARAPHRASE_VARIANTS.items():
        for model in CONFIG["eval_models"]:
            for probe in stage3:
                pid = probe["probe_id"]


                cond_c = f"C_{variant_name}"
                if (model, pid, cond_c) not in existing:
                    narration_para = variant_fn(probe["text_only_narration"])
                    tasks_by_model[model].append((
                        model, probe, cond_c,
                        lambda p=probe, n=narration_para: build_condition_c_prompt(p, n)
                    ))


                cond_bf = f"Bfull_{variant_name}"
                if (model, pid, cond_bf) not in existing:
                    scaffold_para = variant_fn(probe["scaffold_full"])
                    tasks_by_model[model].append((
                        model, probe, cond_bf,
                        lambda p=probe, s=scaffold_para: build_condition_bfull_paraphrase_prompt(p, s)
                    ))

    tasks = interleave_by_model(tasks_by_model, CONFIG["eval_models"])
    worker_count = min(CONFIG["max_workers"], CONFIG["exp3_max_workers"], len(keys))
    print(
        f"Experiment 3: {len(tasks)} calls ({len(existing)} cached, "
        f"using {worker_count} workers)"
    )

    def eval_one(idx_args):
        idx, (model, probe, condition, prompt_fn) = idx_args
        key = keys[idx % len(keys)]
        messages = prompt_fn()
        response = api_call(
            CONFIG["api_url"], key, model, messages,
            CONFIG["temperature"], CONFIG["max_tokens"]
        )
        correct = grade_response(response, probe)
        return {
            "model": model,
            "probe_id": probe["probe_id"],
            "template": probe["template"],
            "condition": condition,
            "response": response,
            "correct": correct,
        }

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futs = [pool.submit(eval_one, (i, t)) for i, t in enumerate(tasks)]
        with open(results_path, "a") as f:
            for i, fut in enumerate(as_completed(futs)):
                result = fut.result()
                f.write(json.dumps(result) + "\n")
                f.flush()
                if (i + 1) % 200 == 0:
                    print(f"  Completed {i+1}/{len(tasks)}")

    print(f"  Done. Results in {results_path}")


def compute_reweighted_pcr(probes: list[dict], results: dict[str, dict]):
    """
    Compute PCR after reweighting templates to equalize answer balance.

    Each template contributes equally regardless of how many probes it has,
    neutralizing any answer-distribution bias.

    Args:
        probes: list of Stage 3 probe dicts
        results: nested dict[condition][model][probe_id] -> bool
    """
    templates = sorted(set(p["template"] for p in probes))
    n_templates = len(templates)

    conditions = ["A", "B_auto", "B_percept", "B_full", "C"]
    models = CONFIG["eval_models"]


    template_acc = {c: {} for c in conditions}
    for cond in conditions:
        for tmpl in templates:
            scores = []
            for model in models:
                tmpl_probes = [
                    p for p in probes if p["template"] == tmpl
                ]
                for p in tmpl_probes:
                    pid = p["probe_id"]
                    if pid in results.get(cond, {}).get(model, {}):
                        scores.append(
                            float(results[cond][model][pid])
                        )
            template_acc[cond][tmpl] = np.mean(scores) if scores else 0.0


    reweighted = {}
    for cond in conditions:
        reweighted[cond] = np.mean(
            [template_acc[cond][t] for t in templates]
        )


    gap = reweighted["C"] - reweighted["A"]
    if abs(gap) < 1e-9:
        print("WARNING: C-A gap is ~0, PCR undefined")
        return

    print("\n=== Template-Reweighted PCR (Stage 3) ===")
    print(f"  A (visual-only):  {reweighted['A']*100:.1f}%")
    print(f"  B_auto:           {reweighted['B_auto']*100:.1f}%")
    print(f"  B_percept:        {reweighted['B_percept']*100:.1f}%")
    print(f"  B_full:           {reweighted['B_full']*100:.1f}%")
    print(f"  C (text-only):    {reweighted['C']*100:.1f}%")
    print(f"  Gap (C - A):      {gap*100:.1f} pp")
    for x in ["B_auto", "B_percept", "B_full"]:
        pcr = (reweighted[x] - reweighted["A"]) / gap
        print(f"  PCR_{x.split('_')[-1]:8s} = {pcr:.3f}")


    n_boot = 10000
    rng = np.random.RandomState(42)
    pcr_boot = {x: [] for x in ["B_auto", "B_percept", "B_full"]}

    for _ in range(n_boot):
        
        boot_templates = rng.choice(templates, size=n_templates, replace=True)
        boot_acc = {}
        for cond in conditions:
            boot_acc[cond] = np.mean(
                [template_acc[cond][t] for t in boot_templates]
            )
        boot_gap = boot_acc["C"] - boot_acc["A"]
        if abs(boot_gap) < 1e-9:
            continue
        for x in ["B_auto", "B_percept", "B_full"]:
            pcr_boot[x].append(
                (boot_acc[x] - boot_acc["A"]) / boot_gap
            )

    print("\n  95% Bootstrap CIs (template-reweighted):")
    for x in ["B_auto", "B_percept", "B_full"]:
        vals = np.array(pcr_boot[x])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"  PCR_{x.split('_')[-1]:8s}: [{lo:.2f}, {hi:.2f}]")


def analyze_results():
    """Load and analyze all rebuttal experiment results."""
    base = Path(CONFIG["output_dir"])
    probes = [p for p in load_probes(CONFIG["probes_path"]) if p["stage"] == 3]


    exp1_path = base / "exp1_vlm_parser" / "results.jsonl"
    if exp1_path.exists():
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: B_vlm (VLM-as-parser)")
        print("=" * 60)
        results = []
        with open(exp1_path) as f:
            for line in f:
                results.append(json.loads(line))

        by_model = defaultdict(list)
        for r in results:
            by_model[r["model"]].append(r["correct"])

        print(f"\n{'Model':<45} {'Acc':>6} {'n':>5}")
        print("-" * 58)
        for model in CONFIG["eval_models"]:
            if model in by_model:
                acc = np.mean(by_model[model]) * 100
                n = len(by_model[model])
                print(f"  {model:<43} {acc:5.1f}% {n:5d}")

        all_correct = [r["correct"] for r in results]
        print(f"\n  Overall B_vlm: {np.mean(all_correct)*100:.1f}%")
        print(f"  (Compare: B_auto=70.3%, B_percept=83.4%, B_full=94.5%)")


    exp2_path = base / "exp2_vis_stripped" / "results.jsonl"
    if exp2_path.exists():
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: B_vis_stripped (B_full minus visibility)")
        print("=" * 60)
        results = []
        with open(exp2_path) as f:
            for line in f:
                results.append(json.loads(line))

        by_model = defaultdict(list)
        for r in results:
            by_model[r["model"]].append(r["correct"])

        print(f"\n{'Model':<45} {'Acc':>6} {'n':>5}")
        print("-" * 58)
        for model in CONFIG["eval_models"]:
            if model in by_model:
                acc = np.mean(by_model[model]) * 100
                n = len(by_model[model])
                print(f"  {model:<43} {acc:5.1f}% {n:5d}")

        all_correct = [r["correct"] for r in results]
        print(f"\n  Overall B_vis_stripped: {np.mean(all_correct)*100:.1f}%")
        print(f"  (Compare: B_percept=83.4%, B_full=94.5%)")
        print(f"  If B_vis_stripped ≈ B_full: visibility statements are redundant")
        print(f"  If B_vis_stripped ≈ B_percept: the lift is perceptual content")


    exp3_path = base / "exp3_paraphrases" / "results.jsonl"
    if exp3_path.exists():
        print("\n" + "=" * 60)
        print("EXPERIMENT 3: C and B_full paraphrase robustness")
        print("=" * 60)
        results = []
        with open(exp3_path) as f:
            for line in f:
                results.append(json.loads(line))

        by_cond = defaultdict(list)
        for r in results:
            by_cond[r["condition"]].append(r["correct"])

        print(f"\n{'Condition':<25} {'Acc':>6} {'n':>6}")
        print("-" * 40)
        for cond in sorted(by_cond.keys()):
            acc = np.mean(by_cond[cond]) * 100
            n = len(by_cond[cond])
            print(f"  {cond:<23} {acc:5.1f}% {n:6d}")

        print("\nCompare with the matched model panel in analysis/reproduce.py.")


    v5_path = ROOT / "results/reference/abstract/original/v5_phase1_results.json"
    if v5_path.exists():
        try:
            from evaluate import load_results as load_v5_results

            v5_results = load_v5_results(v5_path)
            baseline = defaultdict(lambda: defaultdict(dict))
            allowed = {"A", "B_auto", "B_percept", "B_full", "C"}
            for row in v5_results:
                if int(row.get("stage", -1)) != 3:
                    continue
                condition = row.get("condition")
                if condition not in allowed:
                    continue
                model_name = row.get("model_full") or row.get("model")
                baseline[condition][model_name][row["probe_id"]] = bool(row["correct"])

            print("\n" + "=" * 60)
            print("ANALYSIS 4: Template-Reweighted PCR")
            print("=" * 60)
            compute_reweighted_pcr(probes, baseline)
        except Exception as exc:  # noqa: BLE001
            print(f"\nWARNING: could not compute reweighted PCR from V5 results: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Scaffold ablations")
    parser.add_argument(
        "--experiment",
        choices=["vlm_parser", "vis_stripped", "paraphrases", "analyze", "all"],
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--keys", default=CONFIG["keys_path"],
        help="Path to file with API keys (one per line)",
    )
    parser.add_argument(
        "--probes", default=CONFIG["probes_path"],
        help="Path to probes JSON",
    )
    parser.add_argument(
        "--max-workers", type=int, default=CONFIG["max_workers"],
        help="Maximum concurrent worker threads",
    )
    parser.add_argument(
        "--parser-model", default=CONFIG["parser_model"],
        help="Model ID to use as the VLM parser for --experiment vlm_parser",
    )
    parser.add_argument(
        "--output-subdir", default=None,
        help="Optional subdirectory under results/rebuttal for parser outputs",
    )
    args = parser.parse_args()

    if args.experiment == "analyze":
        analyze_results()
        return

    keys = load_keys(args.keys)
    print(f"Loaded {len(keys)} API keys")
    CONFIG["max_workers"] = min(max(1, args.max_workers), max(1, len(keys)))
    print(f"Using {CONFIG['max_workers']} workers")

    probes = load_probes(args.probes)
    print(f"Loaded {len(probes)} probes")

    if args.experiment in ("vlm_parser", "all"):
        run_experiment_vlm_parser(
            probes,
            keys,
            parser_model=args.parser_model,
            output_subdir=args.output_subdir,
        )

    if args.experiment in ("vis_stripped", "all"):
        run_experiment_vis_stripped(probes, keys)

    if args.experiment in ("paraphrases", "all"):
        run_experiment_paraphrases(probes, keys)

    if args.experiment in ("analyze", "all"):
        analyze_results()


if __name__ == "__main__":
    main()
