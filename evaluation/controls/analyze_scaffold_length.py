from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np


OUTPUT_DIR = ROOT / "results" / "followup" / "e4"
PROBES_PATH = ROOT / "probes.json"
V5_DIR = ROOT / "results" / "reference" / "abstract" / "original"
E1_RESULTS = ROOT / "results" / "followup" / "e1" / "results.jsonl"
E2_RESULTS = ROOT / "results" / "followup" / "e2" / "results.jsonl"
VLM_DIR = ROOT / "results" / "reference" / "ablations" / "exp1_vlm_parser"
VIS_STRIPPED_DIR = ROOT / "results" / "reference" / "ablations" / "exp2_vis_stripped"
SMALLTIER_PHASE1 = ROOT / "results" / "reference" / "abstract" / "smalltier" / "v5_phase1_results.json"
CONDITIONS = ("A", "B_auto", "B_vlm", "B_static", "B_percept", "B_full", "B_vis_stripped", "C", "D_mismatch", "D_shuffled", "D_noise")
BOOTSTRAP_RESAMPLES = 10_000


def load_sharded(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for shard in manifest["shards"]:
        rows.extend(json.loads((path.parent / str(shard)).read_text(encoding="utf-8")))
    return rows


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokens(text: str) -> int:
    return len(str(text).split())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


MODEL_KEYS = {
    "openai/gpt-4o": "gpt-4o-2024-11-20",
    "google/gemini-2.5-pro": "gemini-2.5-pro-preview",
    "anthropic/claude-4-sonnet-20250522": "claude-sonnet-4",
    "qwen/qwen2.5-vl-72b-instruct": "qwen-2.5-vl-72b-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "llama-4-maverick",
    "openai/gpt-4.1-mini-2025-04-14": "gpt-4.1-mini",
    "mistralai/ministral-8b-2512": "ministral-8b-2512",
}


def model_key(model: str) -> str:
    if model in MODEL_KEYS:
        return MODEL_KEYS[model]
    return model.split("/", 1)[1] if "/" in model else model


def bootstrap_accuracy(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    point = float(np.mean(values))
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, BOOTSTRAP_RESAMPLES, 500):
        count = min(500, BOOTSTRAP_RESAMPLES - start)
        indexes = rng.integers(0, values.shape[1], size=(count, values.shape[1]))
        samples.append(np.mean(values[:, indexes], axis=(0, 2)))
    joined = np.concatenate(samples)
    low, high = np.percentile(joined, [2.5, 97.5])
    return point, float(low), float(high)


def scaffold_from_prompt(row: dict[str, Any], probe: dict[str, Any]) -> str:
    prompt = str(row.get("prompt", ""))
    task = str(probe["task_prompt"])
    if task in prompt:
        return prompt.split(task, 1)[0]
    return prompt


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    probes = {str(p["id"]): p for p in json.loads(PROBES_PATH.read_text(encoding="utf-8")) if int(p["stage"]) == 3}

    phase1 = [row for row in load_sharded(V5_DIR / "v5_phase1_results.json") if int(row["stage"]) == 3]
    phase2 = [row for row in load_sharded(V5_DIR / "v5_phase2_results.json") if int(row["stage"]) == 3]
    e1 = load_jsonl(E1_RESULTS)
    e2 = load_jsonl(E2_RESULTS)
    vlm_rows = load_jsonl(VLM_DIR / "results.jsonl")
    vlm_scaffolds = json.loads((VLM_DIR / "vlm_scaffolds.json").read_text(encoding="utf-8"))
    strip_log = json.loads((VIS_STRIPPED_DIR / "strip_log.json").read_text(encoding="utf-8"))
    stripped_rows = load_jsonl(VIS_STRIPPED_DIR / "results.jsonl")
    smalltier = json.loads(SMALLTIER_PHASE1.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for row in phase1:
        condition = str(row["condition"])
        if condition not in {"A", "B_auto", "B_percept", "B_full", "C"}:
            continue
        probe = probes[str(row["probe_id"])]
        scaffold = scaffold_from_prompt(row, probe) if condition != "A" else ""
        rows.append({"condition": condition, "model": str(row["model"]), "model_key": model_key(str(row["model"])), "probe_id": str(row["probe_id"]), "correct": int(bool(row["correct"])), "scaffold_tokens": tokens(scaffold)})

    for row in phase2:
        condition = str(row["condition"])
        if condition not in {"D_mismatch", "D_shuffled", "D_noise"}:
            continue
        probe = probes[str(row["probe_id"])]
        scaffold = scaffold_from_prompt(row, probe)
        rows.append({"condition": condition, "model": str(row["model"]), "model_key": model_key(str(row["model"])), "probe_id": str(row["probe_id"]), "correct": int(bool(row["correct"])), "scaffold_tokens": tokens(scaffold)})

    for row in vlm_rows:
        probe_id = str(row["probe_id"])
        rows.append({"condition": "B_vlm", "model": str(row["model"]), "model_key": model_key(str(row["model"])), "probe_id": probe_id, "correct": int(bool(row["correct"])), "scaffold_tokens": tokens(vlm_scaffolds[probe_id])})

    for row in e1:
        rows.append({"condition": "B_static", "model": str(row["model"]), "model_key": model_key(str(row["model"])), "probe_id": str(row["probe_id"]), "correct": int(bool(row["correct"])), "scaffold_tokens": tokens(str(row.get("scaffold", "")))})

    for row in stripped_rows:
        probe_id = str(row["probe_id"])
        rows.append({"condition": "B_vis_stripped", "model": str(row["model"]), "model_key": model_key(str(row["model"])), "probe_id": probe_id, "correct": int(bool(row["correct"])), "scaffold_tokens": tokens(str(strip_log[probe_id]["stripped_scaffold"]))})


    a_keys = {(row["model_key"], row["probe_id"]): row["correct"] for row in rows if row["condition"] == "A"}
    for row in smalltier:
        if int(row["stage"]) == 3 and str(row["condition"]) == "A" and str(row["model"]) == "ministral-8b-2512":
            a_keys.setdefault((str(row["model"]), str(row["probe_id"])), int(bool(row["correct"])))
    for row in rows:
        key = (row["model_key"], row["probe_id"])
        row["gain_vs_A"] = row["correct"] - a_keys[key] if row["condition"] != "A" and key in a_keys else (0 if row["condition"] == "A" else None)

    summary: list[dict[str, Any]] = []
    for index, condition in enumerate(CONDITIONS):
        subset = [row for row in rows if row["condition"] == condition]
        keys = sorted({(row["model_key"], row["probe_id"]) for row in subset})
        lookup = {(row["model_key"], row["probe_id"]): row for row in subset}
        a_common = sorted(set(keys) & set(a_keys))
        matrix = np.array([[lookup[key]["correct"] for key in keys]], dtype=float)
        point, low, high = bootstrap_accuracy(matrix, 12000 + index)
        gains = [lookup[key]["correct"] - a_keys[key] for key in a_common] if condition != "A" else [0]
        summary.append({
            "condition": condition,
            "n_rows": len(subset),
            "n_unique_model_probe": len(keys),
            "mean_scaffold_tokens": float(np.mean([row["scaffold_tokens"] for row in subset])) if subset else 0.0,
            "accuracy": point,
            "low": low,
            "high": high,
            "gain_vs_A": float(np.mean(gains)) if gains else 0.0,
            "paired_n": len(a_common),
        })


    with (OUTPUT_DIR / "scaffold_lengths.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: (CONDITIONS.index(item["condition"]), item["model"], item["probe_id"])):
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    with (OUTPUT_DIR / "condition_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    analysis = {
        "experiment": "E4_scaffold_length",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "definition": "Mean whitespace-token scaffold length from logged prompts/scaffolds and paired accuracy gain over A.",
        "stage": 3,
        "summary": summary,
        "sources": {
            "v5_phase1": "results_v5/v5_phase1_results.json",
            "v5_phase2": "results_v5/v5_phase2_results.json",
            "e1": "e1/results/results.jsonl",
            "e2": "e2/results/results.jsonl",
            "vlm_parser": "results/rebuttal/exp1_vlm_parser",
            "visibility_stripped": "results/rebuttal/exp2_vis_stripped",
        },
    }
    (OUTPUT_DIR / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    lines = ["E4 scaffold length versus accuracy gain", "", "Stage 3 logged conditions; gains are paired against A on common model–probe keys.", ""]
    for row in summary:
        lines.append(f"{row['condition']}: {row['mean_scaffold_tokens']:.1f} tokens | {100 * row['accuracy']:.1f}% [{100 * row['low']:.1f}, {100 * row['high']:.1f}] | gain {100 * row['gain_vs_A']:+.1f} pp | n={row['n_rows']}")
    (OUTPUT_DIR / "analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=180)
        for row in summary:
            ax.scatter(row["mean_scaffold_tokens"], 100 * row["gain_vs_A"], s=65, zorder=3)
            ax.annotate(row["condition"], (row["mean_scaffold_tokens"], 100 * row["gain_vs_A"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_xlabel("Mean scaffold length (whitespace tokens)")
        ax.set_ylabel("Accuracy gain versus A (percentage points)")
        ax.set_title("Scaffold length and accuracy gain")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "scaffold_length_vs_gain.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        (OUTPUT_DIR / "plot_error.txt").write_text(str(exc), encoding="utf-8")

    manifest = {
        "experiment": "E4_scaffold_length",
        "stage": 3,
        "conditions": list(CONDITIONS),
        "row_count": len(rows),
        "source_hashes": {"probes": sha256(PROBES_PATH), "e1_results": sha256(E1_RESULTS), "e2_results": sha256(E2_RESULTS)},
        "new_api_calls": 0,
        "token_definition": "Whitespace-separated tokens from the logged scaffold portion of each prompt; parser scaffolds use the saved scaffold text.",
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote E4 analysis for {len(rows)} logged records to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
