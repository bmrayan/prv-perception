
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path

import argparse
import base64
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    sys.exit("pip install requests")


MODELS = [
    ("openai/gpt-4o-2024-11-20", "GPT-4o"),
    ("google/gemini-2.5-pro-preview", "Gemini 2.5 Pro"),
    ("anthropic/claude-sonnet-4", "Claude Sonnet 4"),
    ("qwen/qwen-2.5-vl-72b-instruct", "Qwen-72B"),
    ("meta-llama/llama-4-maverick", "Llama-4 Mav."),
    ("openai/gpt-4.1-mini", "GPT-4.1-mini"),

]


REF_A = {
    "GPT-4o": 76.4, "Gemini 2.5 Pro": 71.8,
    "Claude Sonnet 4": 78.8, "Qwen-72B": 79.2,
    "Llama-4 Mav.": 87.0, "GPT-4.1-mini": 70.2,
    "Qwen-7B": 59.4,
}
REF_C = {
    "GPT-4o": 91.6, "Gemini 2.5 Pro": 95.4,
    "Claude Sonnet 4": 93.2, "Qwen-72B": 99.4,
    "Llama-4 Mav.": 100.0, "GPT-4.1-mini": 94.8,
    "Qwen-7B": 90.2,
}

API_URL = "https://openrouter.ai/api/v1/chat/completions"


PROMPT_COT_SYSTEM = (
    "You are analyzing a multi-panel image showing colored "
    "circles (characters), smaller circles (objects), and "
    "rectangles (containers). Think step by step:\n"
    "1. For each panel, describe who is present, where they "
    "are, and what objects and containers you see.\n"
    "2. Note any changes between consecutive panels: who "
    "enters, who exits, what moves.\n"
    "3. Track which characters were present or absent during "
    "each change.\n"
    "4. Based on your observations, answer the question.\n"
    "Show your reasoning first, then write FINAL ANSWER: "
    "followed by your answer on the last line."
)

PROMPT_SELF_SCAFFOLD_GENERATE = (
    "You are looking at a multi-panel image showing colored "
    "circles (characters), smaller circles (objects), and "
    "rectangles (containers). There is no text in the image.\n\n"
    "Describe the scene panel by panel. For each panel, state:\n"
    "- Which colored circles (characters) are present and "
    "where they are positioned (left/right, top/bottom)\n"
    "- Which smaller circles (objects) are visible and their "
    "positions\n"
    "- Which rectangles (containers) are present and their "
    "positions\n"
    "- Any changes from the previous panel: who entered, who "
    "exited, what moved\n\n"
    "Be precise about colors. Do NOT interpret intentions or "
    "beliefs. Only describe what you literally see."
)

PROMPT_SELF_SCAFFOLD_REASON = (
    "Based on the following scene description and the image, "
    "answer the question. The scene description was generated "
    "by analyzing the image panel by panel.\n\n"
    "SCENE DESCRIPTION:\n{scaffold}\n\n"
    "QUESTION: {question}\n\n"
    "Answer concisely."
)


class APIKeyPool:
    """Round-robin API key pool with per-key rate limiting."""

    def __init__(self, keys):
        self.keys = list(keys)
        self.idx = 0
        self.lock = threading.Lock()
        
        self.last_call = {k: 0.0 for k in self.keys}
        self.min_interval = 0.5  

    def get_key(self):
        with self.lock:
            key = self.keys[self.idx % len(self.keys)]
            self.idx += 1
            
            now = time.time()
            elapsed = now - self.last_call[key]
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call[key] = time.time()
            return key


def api_call(model_id, messages, api_key, max_tokens=200,
             temperature=0, max_retries=4):
    """Make a single API call with retry logic."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                API_URL, headers=headers, json=payload, timeout=180
            )
            if resp.status_code == 429:
                wait = 15 * (2 ** attempt)
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                time.sleep(10 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return text
        except Exception as e:
            if attempt == max_retries - 1:
                return f"ERROR: {e}"
            time.sleep(5 * (attempt + 1))
    return "ERROR: max retries"


def grade(response, expected, answer_type, aliases=None):
    """Grade a response. For CoT, check last line."""
    resp = response.lower().strip()
    
    if "final answer:" in resp:
        resp = resp.split("final answer:")[-1].strip()
    else:
        
        lines = [l.strip() for l in resp.split("\n") if l.strip()]
        if lines:
            resp = lines[-1]

    check_targets = [expected.lower()]
    if aliases:
        check_targets = [a.lower() for a in aliases]

    if answer_type == "yes_no":
        return resp.startswith(expected.lower())
    elif answer_type == "number":
        return expected in resp[:10]
    elif answer_type in ("left_right", "color", "agent"):
        return any(t in resp for t in check_targets)
    else:
        return any(t in resp for t in check_targets)


def run_exp_cot(probe, model_id, model_name, key_pool):
    """Exp A: CoT self-scaffolding."""
    key = key_pool.get_key()
    messages = [
        {"role": "system", "content": PROMPT_COT_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{probe['image_b64']}"
            }},
            {"type": "text", "text": probe["question"]},
        ]},
    ]
    response = api_call(model_id, messages, key, max_tokens=600)
    correct = grade(response, probe["expected"],
                    probe["answer_type"], probe.get("aliases"))
    return {
        "experiment": "cot",
        "model": model_name,
        "template": probe["template"],
        "variant": probe["variant"],
        "response": response[:800],
        "correct": correct,
    }


def run_exp_self_scaffold(probe, model_id, model_name, key_pool):
    """Exp B: Generate description, then reason with it."""
    
    key1 = key_pool.get_key()
    gen_messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{probe['image_b64']}"
            }},
            {"type": "text", "text": PROMPT_SELF_SCAFFOLD_GENERATE},
        ]},
    ]
    scaffold = api_call(model_id, gen_messages, key1, max_tokens=600)

    if scaffold.startswith("ERROR"):
        return {
            "experiment": "self_scaffold",
            "model": model_name,
            "template": probe["template"],
            "variant": probe["variant"],
            "scaffold": scaffold[:200],
            "response": "ERROR: scaffold generation failed",
            "correct": False,
        }


    key2 = key_pool.get_key()
    reason_prompt = PROMPT_SELF_SCAFFOLD_REASON.format(
        scaffold=scaffold, question=probe["question"]
    )
    reason_messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{probe['image_b64']}"
            }},
            {"type": "text", "text": reason_prompt},
        ]},
    ]
    response = api_call(model_id, reason_messages, key2, max_tokens=200)
    correct = grade(response, probe["expected"],
                    probe["answer_type"], probe.get("aliases"))
    return {
        "experiment": "self_scaffold",
        "model": model_name,
        "template": probe["template"],
        "variant": probe["variant"],
        "scaffold": scaffold[:500],
        "response": response[:500],
        "correct": correct,
    }


def run_exp_disambig_c(probe, model_id, model_name, key_pool):
    """Exp C: Text-only with disambiguated narration."""
    
    narration = probe.get("narration_disambig")
    if not narration:
        narration = probe.get("narration_full", "")
    if not narration:
        return None  

    key = key_pool.get_key()
    question = probe["question"]
    text = f"{narration}\n\n{question}"
    messages = [
        {"role": "user", "content": text},
    ]
    response = api_call(model_id, messages, key, max_tokens=200)
    correct = grade(response, probe["expected"],
                    probe["answer_type"], probe.get("aliases"))
    return {
        "experiment": "disambig_c",
        "model": model_name,
        "template": probe["template"],
        "variant": probe["variant"],
        "response": response[:500],
        "correct": correct,
    }


def load_probes(probes_dir):
    """Load Stage 3 probes."""
    path = Path(probes_dir)
    jsonl_path = path / "stage3_probes.jsonl"
    if jsonl_path.exists():
        probes = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                probes.append(json.loads(line))
        return probes

    if path.is_file() and path.suffix == ".json":
        root = path.resolve().parent
        raw_probes = json.loads(path.read_text(encoding="utf-8"))
        probes = []
        for probe in raw_probes:
            if probe.get("stage") != 3:
                continue
            image_path = root / probe["image_path"]
            image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
            probes.append(
                {
                    "template": probe["template"],
                    "variant": probe.get("variant", probe.get("variant_index", 0)),
                    "image_b64": image_b64,
                    "question": probe.get("question_A", probe.get("question", "")),
                    "expected": probe["expected"],
                    "answer_type": probe.get("answer_kind", probe.get("answer_type")),
                    "aliases": probe.get("expected_aliases", [probe["expected"]]),
                    "narration_full": probe.get("text_narration", ""),
                    "narration_disambig": probe.get("narration_disambig", ""),
                }
            )
        return probes

    sys.exit(
        f"ERROR: could not find Stage 3 probes at {probes_dir}.\n"
        f"Supported inputs:\n"
        f"  - a directory containing stage3_probes.jsonl\n"
        f"  - a JSON file like stimuli_v5/probes.json\n"
    )


def load_done(output_dir):
    """Load completed results for resume."""
    done = set()
    for exp in ["cot", "self_scaffold", "disambig_c"]:
        path = Path(output_dir) / f"{exp}_results.jsonl"
        if path.exists():
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    done.add((r["experiment"], r["model"],
                              r["template"], r["variant"]))
    return done


def build_task_queue(probes, experiments, done):
    """Build list of (experiment, probe, model) tasks."""
    tasks = []
    for exp_name in experiments:
        for model_id, model_name in MODELS:
            for probe in probes:
                key = (exp_name, model_name,
                       probe["template"], probe["variant"])
                if key not in done:
                    tasks.append((exp_name, probe,
                                  model_id, model_name))
    return tasks


def run_task(task, key_pool):
    """Dispatch a single task to the right experiment runner."""
    exp_name, probe, model_id, model_name = task
    if exp_name == "cot":
        return run_exp_cot(probe, model_id, model_name, key_pool)
    elif exp_name == "self_scaffold":
        return run_exp_self_scaffold(
            probe, model_id, model_name, key_pool)
    elif exp_name == "disambig_c":
        return run_exp_disambig_c(
            probe, model_id, model_name, key_pool)


def print_summary(output_dir):
    """Print results summary for pasting into paper."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (paste into paper)")
    print("=" * 60)

    for exp in ["cot", "self_scaffold", "disambig_c"]:
        path = Path(output_dir) / f"{exp}_results.jsonl"
        if not path.exists():
            continue

        results = {}
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                m = r["model"]
                if m not in results:
                    results[m] = {"c": 0, "t": 0}
                results[m]["t"] += 1
                if r["correct"]:
                    results[m]["c"] += 1

        cond_label = {
            "cot": "A_CoT",
            "self_scaffold": "A_self",
            "disambig_c": "C_disambig",
        }[exp]
        ref = REF_A if exp != "disambig_c" else REF_C

        print(f"\n── Experiment: {exp} ({cond_label}) ──")
        print(f"{'Model':<20} {cond_label:>10} {'Ref':>8}"
              f" {'Delta':>8}")
        print("-" * 50)

        all_c, all_t = 0, 0
        for _, model_name in MODELS:
            if model_name in results:
                r = results[model_name]
                acc = r["c"] / r["t"] * 100
                ref_val = ref.get(model_name, 0)
                delta = acc - ref_val
                print(f"{model_name:<20} {acc:>9.1f}%"
                      f" {ref_val:>7.1f}%"
                      f" {delta:>+7.1f}")
                all_c += r["c"]
                all_t += r["t"]

        if all_t > 0:
            agg = all_c / all_t * 100
            ref_agg = (74.7 if exp != "disambig_c" else 94.9)
            print("-" * 50)
            print(f"{'Mean':<20} {agg:>9.1f}%"
                  f" {ref_agg:>7.1f}%"
                  f" {agg - ref_agg:>+7.1f}")


    print("\n" + "=" * 60)
    print("LATEX TABLE ROWS (copy into paper)")
    print("=" * 60)
    for exp in ["cot", "self_scaffold"]:
        path = Path(output_dir) / f"{exp}_results.jsonl"
        if not path.exists():
            continue
        results = {}
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                m = r["model"]
                if m not in results:
                    results[m] = {"c": 0, "t": 0}
                results[m]["t"] += 1
                if r["correct"]:
                    results[m]["c"] += 1

        cond = "A_\\mathrm{CoT}" if exp == "cot" else "A_\\mathrm{self}"
        print(f"\n% {exp} table rows:")
        for _, model_name in MODELS:
            if model_name in results:
                r = results[model_name]
                acc = r["c"] / r["t"] * 100
                ref_val = REF_A.get(model_name, 0)
                delta = acc - ref_val
                sign = "+" if delta >= 0 else ""
                color_cmd = "\\up" if delta >= 0 else "\\dn"
                print(f"% {model_name:<18} & {acc:.1f}"
                      f"{color_cmd}{{{sign}{delta:.1f}}} "
                      f"& {ref_val:.1f} \\\\")


def main():
    parser = argparse.ArgumentParser(
        description="Run reasoning baselines"
    )
    parser.add_argument("--keys_file", default=None,
                        help="File with one API key per line")
    parser.add_argument("--probes_dir", default=str(ROOT / "probes.json"),
                        help="Dir with stage3_probes.jsonl")
    parser.add_argument("--output_dir", default=str(ROOT / "results/generated/reasoning"),
                        help="Output directory")
    parser.add_argument("--experiments", nargs="+",
                        default=["cot", "self_scaffold"],
                        choices=["cot", "self_scaffold",
                                 "disambig_c"],
                        help="Which experiments to run")
    parser.add_argument("--workers", type=int, default=20,
                        help="Number of parallel workers")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results")
    args = parser.parse_args()


    keys = [os.environ["OPENROUTER_API_KEY"]] if os.environ.get("OPENROUTER_API_KEY") else []
    if args.keys_file:
        for raw in Path(args.keys_file).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("sk-or-"):
                keys.append(line)
            else:
                path = Path(line)
                if path.exists():
                    text = path.read_text(encoding="utf-8").strip()
                    if text:
                        keys.append(text.split()[-1].strip())
    if not keys:
        raise SystemExit("Set OPENROUTER_API_KEY or pass --keys_file.")
    print(f"Loaded {len(keys)} API keys")
    key_pool = APIKeyPool(keys)


    probes = load_probes(args.probes_dir)
    print(f"Loaded {len(probes)} Stage 3 probes")


    os.makedirs(args.output_dir, exist_ok=True)


    done = load_done(args.output_dir) if args.resume else set()
    if done:
        print(f"Resuming: {len(done)} tasks already completed")


    tasks = build_task_queue(probes, args.experiments, done)
    print(f"Tasks to run: {len(tasks)}")
    print(f"Workers: {args.workers}")

    if not tasks:
        print("Nothing to do!")
        print_summary(args.output_dir)
        return


    writers = {}
    for exp in args.experiments:
        path = Path(args.output_dir) / f"{exp}_results.jsonl"
        writers[exp] = open(path, "a")


    completed = 0
    errors = 0
    lock = threading.Lock()
    t0 = time.time()

    def process_task(task):
        nonlocal completed, errors
        result = run_task(task, key_pool)
        if result is None:
            return
        with lock:
            exp_name = result["experiment"]
            writers[exp_name].write(json.dumps(result) + "\n")
            writers[exp_name].flush()
            completed += 1
            if result.get("response", "").startswith("ERROR"):
                errors += 1
            if completed % 100 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed * 3600
                remaining = (len(tasks) - completed) / (
                    completed / elapsed) if completed > 0 else 0
                print(f"  [{completed}/{len(tasks)}] "
                      f"{rate:.0f}/hr, "
                      f"~{remaining/60:.0f}min left, "
                      f"{errors} errors")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_task, t) for t in tasks]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"  Task error: {e}")


    for w in writers.values():
        w.close()

    elapsed = time.time() - t0
    print(f"\nDone! {completed} tasks in {elapsed/60:.1f} min "
          f"({errors} errors)")


    print_summary(args.output_dir)


if __name__ == "__main__":
    main()
