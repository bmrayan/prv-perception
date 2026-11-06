from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from typing import Any


OUT_DIR = ROOT / "results" / "followup" / "derived_visibility"
DEFAULT_RESULTS = OUT_DIR / "muma_visibility_followup_20260727.jsonl"

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


def load_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    malformed = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
    return rows, malformed


def exact_mcnemar_p(b: int, c: int) -> float | None:
    """Two-sided exact McNemar p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return None
    denominator = 2**n
    observed = min(b, c)
    probability = 0.0
    for k in range(observed + 1):
        probability += math.comb(n, k) / denominator
    for k in range(n - observed, n + 1):
        probability += math.comb(n, k) / denominator
    return min(1.0, probability)


def accuracy(items: list[dict[str, Any]], condition: str) -> float:
    values = [int(row[condition]) for row in items]
    return sum(values) / len(values) if values else float("nan")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def bootstrap_video_ci(
    paired: list[dict[str, Any]],
    iterations: int = 5000,
    seed: int = 20260727,
) -> dict[str, dict[str, float]]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in paired:
        by_video[item["video_id"]].append(item)
    videos = sorted(by_video)
    if not videos:
        return {}
    rng = random.Random(seed)
    full_values: list[float] = []
    stripped_values: list[float] = []
    differences: list[float] = []
    for _ in range(iterations):
        sampled = [videos[rng.randrange(len(videos))] for _ in videos]
        sample = [item for video in sampled for item in by_video[video]]
        full = accuracy(sample, "full_correct")
        stripped = accuracy(sample, "stripped_correct")
        full_values.append(full)
        stripped_values.append(stripped)
        differences.append(full - stripped)
    return {
        "B_full_vis_accuracy": {
            "lower": percentile(full_values, 0.025),
            "upper": percentile(full_values, 0.975),
        },
        "B_vis_stripped_accuracy": {
            "lower": percentile(stripped_values, 0.025),
            "upper": percentile(stripped_values, 0.975),
        },
        "paired_difference": {
            "lower": percentile(differences, 0.025),
            "upper": percentile(differences, 0.975),
        },
    }


def make_paired(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    statuses = Counter(str(row.get("status")) for row in rows)
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (str(row.get("question_id")), str(row.get("model")), str(row.get("condition")))


        by_key.setdefault(key, row)

    paired: list[dict[str, Any]] = []
    for question_id in sorted({key[0] for key in by_key}):
        for model in MODELS:
            full = by_key.get((question_id, model, "B_full_vis"))
            stripped = by_key.get((question_id, model, "B_vis_stripped"))
            if not full or not stripped:
                continue
            paired.append(
                {
                    "question_id": question_id,
                    "video_id": str(full.get("video_id")),
                    "model": model,
                    "full_correct": bool(full.get("correct")),
                    "stripped_correct": bool(stripped.get("correct")),
                    "gold_answer": full.get("gold_answer"),
                    "full_parsed_answer": full.get("parsed_answer"),
                    "stripped_parsed_answer": stripped.get("parsed_answer"),
                }
            )
    return paired, statuses


def pct(value: float | None) -> str:
    return "NA" if value is None or math.isnan(value) else f"{100 * value:.2f}%"


def write_outputs(
    results_path: Path,
    raw_row_count: int,
    paired: list[dict[str, Any]],
    statuses: Counter[str],
    malformed: int,
) -> None:
    manifest_path = OUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    expected = int(manifest.get("calls_planned", 0))
    expected_pairs = int(manifest.get("eligible_question_count", 0)) * len(MODELS)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in paired:
        by_model[item["model"]].append(item)

    overall_b = sum(int(item["full_correct"]) for item in paired)
    overall_s = sum(int(item["stripped_correct"]) for item in paired)
    discordant_full_only = sum(item["full_correct"] and not item["stripped_correct"] for item in paired)
    discordant_stripped_only = sum(not item["full_correct"] and item["stripped_correct"] for item in paired)

    model_rows: list[dict[str, Any]] = []
    for model in MODELS:
        items = by_model[model]
        b = sum(int(item["full_correct"]) for item in items)
        s = sum(int(item["stripped_correct"]) for item in items)
        model_rows.append(
            {
                "model": model,
                "n_paired": len(items),
                "B_full_vis_correct": b,
                "B_full_vis_accuracy": b / len(items) if items else None,
                "B_vis_stripped_correct": s,
                "B_vis_stripped_accuracy": s / len(items) if items else None,
                "difference_full_minus_stripped": (b - s) / len(items) if items else None,
            }
        )

    bootstrap = bootstrap_video_ci(paired) if paired else {}
    analysis = {
        "experiment": "E3_MuMA_derived_visibility_followup",
        "results_path": str(results_path.relative_to(ROOT)),
        "historical_results_unchanged": True,
        "derived_followup_note": "B_full_vis appends one derived temporal visibility sentence to the original MuMA interaction summary; B_vis_stripped is the unchanged original summary. This is not a reconstruction of the historical MuMA B_full condition.",
        "preparation": {
            "eligible_videos": manifest.get("eligible_video_count"),
            "eligible_questions": manifest.get("eligible_question_count"),
            "models": len(MODELS),
            "conditions": CONDITIONS,
            "calls_planned": expected,
        },
        "raw_status_counts": dict(statuses),
        "malformed_lines": malformed,
        "successful_unique_keys": len({(item["question_id"], item["model"], condition) for item in paired for condition in CONDITIONS}),
        "paired_question_model_count": len(paired),
        "expected_question_model_pairs": expected_pairs,
        "complete_matrix": len(paired) == expected_pairs,
        "overall": {
            "n_paired": len(paired),
            "B_full_vis_correct": overall_b,
            "B_full_vis_accuracy": overall_b / len(paired) if paired else None,
            "B_vis_stripped_correct": overall_s,
            "B_vis_stripped_accuracy": overall_s / len(paired) if paired else None,
            "difference_full_minus_stripped": (overall_b - overall_s) / len(paired) if paired else None,
            "discordant_full_only": discordant_full_only,
            "discordant_stripped_only": discordant_stripped_only,
            "mcnemar_exact_two_sided_p": exact_mcnemar_p(discordant_full_only, discordant_stripped_only),
        },
        "by_model": model_rows,
        "video_cluster_bootstrap_95_ci": bootstrap,
    }
    (OUT_DIR / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    with (OUT_DIR / "by_model.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=model_rows[0].keys() if model_rows else ["model"])
        writer.writeheader()
        writer.writerows(model_rows)

    lines = [
        "E3 MuMA derived visibility follow-up",
        "",
        "This is a new, derived follow-up. It does not alter historical MuMA results.",
        "B_full_vis = original MuMA summary + one derived temporal visibility sentence.",
        "B_vis_stripped = unchanged original MuMA summary.",
        "Only status=ok rows and question/model pairs with both conditions are analyzed.",
        "Video-cluster bootstrap intervals resample eligible videos (5,000 replicates).",
        "",
        f"Raw rows: {raw_row_count}",
        f"Status counts: {dict(statuses)}",
        f"Malformed JSONL lines: {malformed}",
        f"Paired question/model observations: {len(paired)} / {expected_pairs}",
        f"Complete matrix: {analysis['complete_matrix']}",
        "",
        "Overall",
        f"B_full_vis: {overall_b}/{len(paired)} = {pct(analysis['overall']['B_full_vis_accuracy'])}",
        f"B_vis_stripped: {overall_s}/{len(paired)} = {pct(analysis['overall']['B_vis_stripped_accuracy'])}",
        f"Difference (full - stripped): {pct(analysis['overall']['difference_full_minus_stripped'])}",
        f"Discordant pairs full-only / stripped-only: {discordant_full_only} / {discordant_stripped_only}",
        f"Exact McNemar two-sided p: {analysis['overall']['mcnemar_exact_two_sided_p']}",
        "",
        "Per model",
        "model | n | B_full_vis | B_vis_stripped | difference",
        "--- | ---: | ---: | ---: | ---:",
    ]
    for item in model_rows:
        lines.append(
            f"{item['model']} | {item['n_paired']} | {pct(item['B_full_vis_accuracy'])} | {pct(item['B_vis_stripped_accuracy'])} | {pct(item['difference_full_minus_stripped'])}"
        )
    lines += [
        "",
        "Interpretation boundary",
        "The result tests whether an explicitly supplied derived visibility sentence changes performance within this restricted eligible subset. It is additive evidence and cannot retroactively change what the historical MuMA B_full prompts contained.",
    ]
    (OUT_DIR / "analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    rows, malformed = load_rows(args.results)
    paired, statuses = make_paired(rows)
    write_outputs(args.results, len(rows), paired, statuses, malformed)
    analysis_path = OUT_DIR / "analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["raw_row_count"] = len(rows)
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    text_path = OUT_DIR / "analysis.txt"
    print(json.dumps({
        "raw_rows": len(rows),
        "statuses": dict(statuses),
        "paired": len(paired),
        "expected_pairs": analysis["expected_question_model_pairs"],
        "complete_matrix": analysis["complete_matrix"],
    }, indent=2))


if __name__ == "__main__":
    main()
