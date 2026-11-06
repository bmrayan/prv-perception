from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np


PROBES_PATH = ROOT / "probes.json"
V5_RESULTS = ROOT / "results" / "reference" / "abstract" / "original"
OUTPUT_DIR = ROOT / "results" / "followup" / "e2"
STAGE = 3
CONDITIONS = ("A", "D_mismatch", "D_shuffled", "D_noise")
BOOTSTRAP_RESAMPLES = 10_000
MODELS = (
    "gpt-4o-2024-11-20",
    "gemini-2.5-pro-preview",
    "claude-sonnet-4",
    "qwen-2.5-vl-72b-instruct",
    "llama-4-maverick",
    "gpt-4.1-mini",
    "qwen-2.5-vl-7b-instruct",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sharded_manifest(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("_format") != "sharded_json_array_v1":
        raise ValueError(f"Unexpected result format in {path}")
    rows: list[dict[str, Any]] = []
    for shard in manifest["shards"]:
        shard_rows = json.loads((path.parent / str(shard)).read_text(encoding="utf-8"))
        if not isinstance(shard_rows, list):
            raise ValueError(f"Expected an array in {shard}")
        rows.extend(shard_rows)
    if len(rows) != int(manifest["total_records"]):
        raise ValueError(f"Shard count mismatch for {path}: {len(rows)} != {manifest['total_records']}")
    return rows


def percentile_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    point = float(np.mean(values))
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for start in range(0, BOOTSTRAP_RESAMPLES, 500):
        count = min(500, BOOTSTRAP_RESAMPLES - start)
        indexes = rng.integers(0, values.shape[1], size=(count, values.shape[1]))
        samples.append(np.mean(values[:, indexes], axis=(0, 2)))
    bootstrap = np.concatenate(samples)
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return point, float(low), float(high)


def pair_map(rows: list[dict[str, Any]], condition: str) -> dict[tuple[str, str], int]:
    return {
        (str(row["model"]), str(row["probe_id"])): int(bool(row["correct"]))
        for row in rows
        if str(row["condition"]) == condition
    }


def mcnemar(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], condition_b: str) -> dict[str, Any]:
    a = pair_map(rows_a, "A")
    b = pair_map(rows_b, condition_b)
    common = sorted(set(a) & set(b))
    discordant_a_only = sum(a[key] == 1 and b[key] == 0 for key in common)
    discordant_b_only = sum(a[key] == 0 and b[key] == 1 for key in common)
    total = discordant_a_only + discordant_b_only
    chi_sq = 0.0 if total == 0 else (abs(discordant_a_only - discordant_b_only) - 1.0) ** 2 / total
    p_value = 1.0 if total == 0 else math.erfc(math.sqrt(max(chi_sq, 0.0) / 2.0))
    return {
        "comparison": f"A vs {condition_b}",
        "n_pairs": len(common),
        "A_only_correct": discordant_a_only,
        "condition_only_correct": discordant_b_only,
        "chi_sq": chi_sq,
        "p": p_value,
        "p_bonf_v5": min(p_value * 12.0, 1.0),
        "bonferroni_family_size_v5": 12,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase1 = load_sharded_manifest(V5_RESULTS / "v5_phase1_results.json")
    phase2 = load_sharded_manifest(V5_RESULTS / "v5_phase2_results.json")
    selected = [
        row
        for row in phase1 + phase2
        if int(row["stage"]) == STAGE and str(row["condition"]) in CONDITIONS
    ]
    error_count = sum(str(row.get("response", "")).startswith("ERROR") for row in selected)

    counts = Counter((str(row["condition"]), str(row["model"])) for row in selected)
    expected = len(MODELS) * 500
    for condition in CONDITIONS:
        for model in MODELS:
            if counts[(condition, model)] != 500:
                raise ValueError(f"Expected 500 rows for {condition}/{model}, got {counts[(condition, model)]}")

    result_path = OUTPUT_DIR / "results.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for row in sorted(selected, key=lambda item: (CONDITIONS.index(str(item["condition"])), str(item["model"]), str(item["probe_id"]))):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    probe_ids = sorted({str(row["probe_id"]) for row in selected if str(row["condition"]) == "A"})
    matrices: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        lookup = {(str(row["model"]), str(row["probe_id"])): int(bool(row["correct"])) for row in selected if str(row["condition"]) == condition}
        matrices[condition] = np.array([[lookup[(model, probe_id)] for probe_id in probe_ids] for model in MODELS], dtype=float)

    accuracy: dict[str, dict[str, Any]] = {}
    for index, condition in enumerate(CONDITIONS):
        point, low, high = percentile_ci(matrices[condition], seed=9100 + index)
        accuracy[condition] = {"point": point, "low": low, "high": high, "n": int(matrices[condition].size)}

    phase1_stage3 = [row for row in phase1 if int(row["stage"]) == STAGE]
    phase2_stage3 = [row for row in phase2 if int(row["stage"]) == STAGE]
    analysis = {
        "experiment": "E2_controls",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "D_mismatch": "Scaffold copied from a different probe",
            "D_shuffled": "Correct scaffold with panel sentences shuffled",
            "D_noise": "Correct scaffold with entity names reassigned",
        },
        "stage": STAGE,
        "probe_count": len(probe_ids),
        "models": list(MODELS),
        "result_count": len(selected),
        "error_count": error_count,
        "accuracy": accuracy,
        "delta_vs_A": {condition: accuracy[condition]["point"] - accuracy["A"]["point"] for condition in CONDITIONS},
        "mcnemar": [mcnemar(phase1_stage3, phase2_stage3, condition) for condition in CONDITIONS[1:]],
        "source_analysis_reference": "results_v5/v5_analysis.json",
    }
    (OUTPUT_DIR / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    lines = [
        "E2 control analysis",
        "",
        "Stage 3 only; 500 probes x 7 models per condition.",
        "Bootstrap intervals use 10,000 probe resamples.",
        f"Inherited source-run error rows counted as incorrect: {error_count}.",
        "",
    ]
    for condition in CONDITIONS:
        row = accuracy[condition]
        delta = "baseline" if condition == "A" else f"{100 * analysis['delta_vs_A'][condition]:+.1f} pp vs A"
        lines.append(f"{condition}: {100 * row['point']:.1f}% [{100 * row['low']:.1f}, {100 * row['high']:.1f}] ({delta})")
    lines.extend(["", "McNemar paired tests; Bonferroni values use the original V5 family size of 12."])
    for test in analysis["mcnemar"]:
        lines.append(
            f"{test['comparison']}: chi^2={test['chi_sq']:.3f}, p={test['p']:.3g}, "
            f"p_bonf_v5={test['p_bonf_v5']:.3g}, n={test['n_pairs']}"
        )
    (OUTPUT_DIR / "analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "experiment": "E2_controls",
        "source_phase1": "results_v5/v5_phase1_results.json",
        "source_phase2": "results_v5/v5_phase2_results.json",
        "source_probes": "stimuli_v5/probes.json",
        "source_hashes": {
            "phase1_manifest": sha256(V5_RESULTS / "v5_phase1_results.json"),
            "phase2_manifest": sha256(V5_RESULTS / "v5_phase2_results.json"),
            "probes": sha256(PROBES_PATH),
        },
        "stage": STAGE,
        "probe_count": len(probe_ids),
        "models": list(MODELS),
        "conditions": list(CONDITIONS),
        "result_count": len(selected),
        "error_count": error_count,
        "rows_per_condition": {condition: int(sum(str(row["condition"]) == condition for row in selected)) for condition in CONDITIONS},
        "api_calls": 0,
        "note": "E2 was already run in V5 phase 2; this bundle extracts and analyzes the completed records.",
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Packaged {len(selected)} E2 records into {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
