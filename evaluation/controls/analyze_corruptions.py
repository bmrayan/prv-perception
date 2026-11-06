from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np


OUT = ROOT / "results" / "followup" / "e2"
PROBES = {str(p["id"]): p for p in json.loads((ROOT / "probes.json").read_text(encoding="utf-8")) if int(p["stage"]) == 3}
MODELS = [
    "gpt-4o-2024-11-20", "gemini-2.5-pro-preview", "claude-sonnet-4",
    "qwen-2.5-vl-72b-instruct", "llama-4-maverick", "gpt-4.1-mini", "qwen-2.5-vl-7b-instruct",
]
CONDITIONS = ["A", "B_percept", "C", "D_mismatch", "D_shuffled", "D_noise"]
BOOTSTRAP = 10_000


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_sharded(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for shard in manifest["shards"]:
        rows.extend(json.loads((path.parent / str(shard)).read_text(encoding="utf-8")))
    return rows


def matrix(rows: list[dict[str, Any]], condition: str) -> np.ndarray:
    probe_ids = sorted(PROBES)
    lookup = {(str(r["model"]), str(r["probe_id"])): int(bool(r["correct"])) for r in rows if str(r["condition"]) == condition}
    return np.array([[lookup[(model, probe)] for probe in probe_ids] for model in MODELS], dtype=float)


def bootstrap_accuracy(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    point = float(values.mean())
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, BOOTSTRAP, 500):
        count = min(500, BOOTSTRAP - start)
        index = rng.integers(0, values.shape[1], size=(count, values.shape[1]))
        samples.append(values[:, index].mean(axis=(0, 2)))
    joined = np.concatenate(samples)
    low, high = np.percentile(joined, [2.5, 97.5])
    return point, float(low), float(high)


def bootstrap_pcr(a: np.ndarray, b: np.ndarray, c: np.ndarray, seed: int) -> tuple[float, float, float]:
    point = float((b.mean() - a.mean()) / (c.mean() - a.mean()))
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, BOOTSTRAP, 500):
        count = min(500, BOOTSTRAP - start)
        index = rng.integers(0, a.shape[1], size=(count, a.shape[1]))
        aa = a[:, index].mean(axis=(0, 2)); bb = b[:, index].mean(axis=(0, 2)); cc = c[:, index].mean(axis=(0, 2))
        samples.append((bb - aa) / (cc - aa))
    joined = np.concatenate(samples)
    low, high = np.percentile(joined, [2.5, 97.5])
    return point, float(low), float(high)


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    av = a.reshape(-1); bv = b.reshape(-1)
    b_only = int(np.sum((av == 1) & (bv == 0)))
    c_only = int(np.sum((av == 0) & (bv == 1)))
    discordant = b_only + c_only
    chi_sq = 0.0 if discordant == 0 else (abs(b_only - c_only) - 1.0) ** 2 / discordant
    p = 1.0 if discordant == 0 else math.erfc(math.sqrt(max(chi_sq, 0.0) / 2.0))
    return {"n_pairs": int(av.size), "a_only_correct": b_only, "b_only_correct": c_only, "chi_sq": chi_sq, "p": p}


def main() -> None:
    e2 = load_jsonl(OUT / "results.jsonl")
    phase1 = [r for r in load_sharded(ROOT / "results" / "reference" / "abstract" / "original" / "v5_phase1_results.json") if int(r["stage"]) == 3 and str(r["condition"]) in {"B_percept", "C"}]
    source_rows = e2 + phase1
    mats = {condition: matrix(source_rows, condition) for condition in CONDITIONS}
    accuracy = {}
    for index, condition in enumerate(CONDITIONS):
        point, low, high = bootstrap_accuracy(mats[condition], 15000 + index)
        accuracy[condition] = {"point": point, "low": low, "high": high, "n": int(mats[condition].size)}

    pcr = {}
    for index, condition in enumerate(["D_mismatch", "D_shuffled", "D_noise"]):
        point, low, high = bootstrap_pcr(mats["A"], mats[condition], mats["C"], 16000 + index)
        pcr[condition] = {"point": point, "low": low, "high": high}

    paired = {}
    for condition in ["D_mismatch", "D_shuffled", "D_noise"]:
        paired[f"A_vs_{condition}"] = mcnemar(mats["A"], mats[condition])
        paired[f"B_percept_vs_{condition}"] = mcnemar(mats["B_percept"], mats[condition])

    length_diffs = []
    same_template = 0
    for probe_id, probe in PROBES.items():
        partner = PROBES.get(str(probe.get("mismatch_partner_id")))
        if partner is None:
            continue
        length_diffs.append(abs(len(str(probe["scaffold_percept"]).split()) - len(str(partner["scaffold_percept"]).split())))
        same_template += int(str(probe["template"]) == str(partner["template"]))
    length_match = {
        "n": len(length_diffs),
        "same_template": same_template,
        "mean_abs_token_difference": float(np.mean(length_diffs)),
        "max_abs_token_difference": int(max(length_diffs)),
        "exact_matches": int(sum(diff == 0 for diff in length_diffs)),
    }

    template_accuracy: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for template in sorted({str(p["template"]) for p in PROBES.values()}):
        ids = {pid for pid, p in PROBES.items() if str(p["template"]) == template}
        for condition in CONDITIONS:
            values = [int(bool(r["correct"])) for r in source_rows if str(r["condition"]) == condition and str(r["probe_id"]) in ids]
            template_accuracy[template][condition] = {"accuracy": float(np.mean(values)), "n": len(values)}

    per_model = {}
    for model_index, model in enumerate(MODELS):
        per_model[model] = {}
        for condition in CONDITIONS:
            values = mats[condition][model_index]
            per_model[model][condition] = float(values.mean())
        for condition in ["D_mismatch", "D_shuffled", "D_noise"]:
            per_model[model][f"PCR_{condition}"] = float((mats[condition][model_index].mean() - mats["A"][model_index].mean()) / (mats["C"][model_index].mean() - mats["A"][model_index].mean()))

    stronger = {
        "experiment": "E2_controls_strengthened",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "accuracy": accuracy,
        "PCR": pcr,
        "paired_tests": paired,
        "length_match_audit": length_match,
        "per_model": per_model,
        "template_accuracy": dict(template_accuracy),
        "source": ["e2/results/results.jsonl", "results_v5/v5_phase1_results.json"],
    }
    (OUT / "analysis_stronger.json").write_text(json.dumps(stronger, indent=2), encoding="utf-8")
    lines = [
        "E2 strengthened analysis",
        "",
        f"Length-matched mismatch audit: {length_match['exact_matches']}/{length_match['n']} exact scaffold-length matches; same-template partners {length_match['same_template']}/{length_match['n']}.",
        "",
        "Aggregate accuracy:",
    ]
    for condition, row in accuracy.items():
        lines.append(f"  {condition}: {100 * row['point']:.1f}% [{100 * row['low']:.1f}, {100 * row['high']:.1f}]")
    lines.append("")
    lines.append("PCR relative to A and C:")
    for condition, row in pcr.items():
        lines.append(f"  {condition}: {row['point']:.3f} [{row['low']:.3f}, {row['high']:.3f}]")
    lines.append("")
    lines.append("Paired tests:")
    for name, row in paired.items():
        lines.append(f"  {name}: chi^2={row['chi_sq']:.3f}, p={row['p']:.3g}, n={row['n_pairs']}")
    (OUT / "analysis_stronger.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote strengthened E2 analysis to {OUT}")


if __name__ == "__main__":
    main()
