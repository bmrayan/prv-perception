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


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
import evaluate_positions as base  # noqa: E402


OUT = ROOT / "results" / "followup" / "e1"
PROBES = [p for p in base.load_json_array(ROOT / "probes.json") if int(p.get("stage", 0)) == 3]
PROBES.sort(key=lambda p: (str(p["template"]), int(p["variant_index"])))
PROBE_IDS = [str(p["id"]) for p in PROBES]
CONDITIONS = ["A_original", "A_fresh", "B_identity", "B_static", "B_percept", "B_full", "C"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def matrix(rows: list[dict[str, Any]], model: str, condition: str) -> np.ndarray:
    lookup = {(str(row["model"]), str(row["probe_id"])): int(bool(row["correct"])) for row in rows}
    return np.asarray([lookup[(model, probe_id)] for probe_id in PROBE_IDS], dtype=float)


def accuracy(values: np.ndarray, seed: int) -> dict[str, Any]:
    point, low, high = base.bootstrap(values, seed)
    return {"point": point, "low": low, "high": high, "n": int(values.size)}


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))
    discordant = a_only + b_only
    chi_sq = 0.0 if discordant == 0 else (abs(a_only - b_only) - 1.0) ** 2 / discordant
    p = 1.0 if discordant == 0 else math.erfc(math.sqrt(max(chi_sq, 0.0) / 2.0))
    return {"n_pairs": int(a.size), "a_only_correct": a_only, "b_only_correct": b_only, "chi_sq": chi_sq, "p": p}


def pcr_ci(a: np.ndarray, b: np.ndarray, c: np.ndarray, seed: int) -> dict[str, float]:
    point, low, high = base.pcr_ci(a, b, c, seed)
    return {"point": point, "low": low, "high": high}


def audit(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row.get("condition")) == condition]
    keys = [(str(row.get("model")), str(row.get("probe_id"))) for row in selected]
    errors = sum(1 for row in selected if row.get("response") is None or row.get("raw_api") is None)
    return {
        "rows": len(selected),
        "unique_model_probe_pairs": len(set(keys)),
        "duplicate_pairs": len(keys) - len(set(keys)),
        "error_rows": errors,
        "models": sorted({key[0] for key in keys}),
        "probes": len({key[1] for key in keys}),
    }


def main() -> None:
    if len(PROBES) != 500:
        raise ValueError(f"Expected 500 Stage 3 probes, found {len(PROBES)}")
    fresh_rows = load_jsonl(OUT / "fresh_a" / "results.jsonl")
    repairs = load_jsonl(OUT / "fresh_a" / "repairs.jsonl")
    repaired_keys = {(str(row["model"]), str(row["probe_id"])) for row in repairs}
    fresh_rows = [row for row in fresh_rows if (str(row["model"]), str(row["probe_id"])) not in repaired_keys] + repairs
    identity_rows = load_jsonl(OUT / "identity_only" / "results.jsonl")
    static_rows = load_jsonl(OUT / "results.jsonl")
    baseline = base.load_baseline_results()

    source_rows: dict[str, list[dict[str, Any]]] = {
        "A_fresh": fresh_rows,
        "B_identity": identity_rows,
        "B_static": static_rows,
    }
    for model in base.MODELS:
        for condition, source_condition in (("A_original", "A"), ("B_percept", "B_percept"), ("B_full", "B_full"), ("C", "C")):
            source_rows.setdefault(condition, []).extend(
                dict(row, model=model) for row in baseline[model] if str(row.get("condition")) == source_condition
            )

    matrices: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for model in base.MODELS:
        for condition in CONDITIONS:
            matrices[model][condition] = matrix(source_rows[condition], model, condition if condition in {"A_fresh", "B_identity", "B_static"} else {"A_original": "A", "B_percept": "B_percept", "B_full": "B_full", "C": "C"}[condition])

    aggregate: dict[str, Any] = {}
    for index, condition in enumerate(CONDITIONS):
        values = np.concatenate([matrices[model][condition] for model in base.MODELS])
        aggregate[condition] = accuracy(values, 24000 + index)
    pcr = {}
    for index, condition in enumerate(["B_identity", "B_static", "B_percept", "B_full"]):
        a = np.concatenate([matrices[model]["A_fresh"] for model in base.MODELS])
        b = np.concatenate([matrices[model][condition] for model in base.MODELS])
        c = np.concatenate([matrices[model]["C"] for model in base.MODELS])
        pcr[condition] = pcr_ci(a, b, c, 24100 + index)

    paired = {}
    for condition in ["A_original", "B_identity", "B_static", "B_percept", "B_full", "C"]:
        a = np.concatenate([matrices[model]["A_fresh"] for model in base.MODELS])
        b = np.concatenate([matrices[model][condition] for model in base.MODELS])
        paired[f"A_fresh_vs_{condition}"] = mcnemar(a, b)

    per_model: dict[str, Any] = {}
    for model in base.MODELS:
        per_model[model] = {condition: float(matrices[model][condition].mean()) for condition in CONDITIONS}
        per_model[model].update({f"PCR_{condition}": pcr_ci(matrices[model]["A_fresh"], matrices[model][condition], matrices[model]["C"], 24200 + index) for index, condition in enumerate(["B_identity", "B_static", "B_percept", "B_full"])})

    identity_terms = ("position", "left", "right", "above", "below", "behind", "in front", "visible", "see", "believe", "thinks", "action", "moves", "moved", "not there", "absent")
    scaffold_rows = json.loads((OUT / "identity_only" / "identity_scaffolds.json").read_text(encoding="utf-8"))
    forbidden_hits = {term: sum(term in str(scaffold).lower() for scaffold in scaffold_rows.values()) for term in identity_terms}

    stronger = {
        "experiment": "E1_strengthened_controls",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "definition": "A_fresh reruns the exact A prompt with the submitted model identifiers. B_identity adds only panel-local entity names; it omits positions, relations, events, visibility, absence, and beliefs.",
        "aggregate": aggregate,
        "PCR_relative_to_A_fresh_and_C": pcr,
        "paired_tests": paired,
        "per_model": per_model,
        "audit": {"A_fresh": audit(fresh_rows, "A_fresh"), "B_identity": audit(identity_rows, "B_identity"), "B_static": audit(static_rows, "B_static")},
        "identity_scaffold_forbidden_term_hits": forbidden_hits,
        "sources": ["e1/results/fresh_a/results.jsonl", "e1/results/fresh_a/repairs.jsonl", "e1/results/identity_only/results.jsonl", "e1/results/results.jsonl", "results_v5/v5_phase1_results.json"],
    }
    (OUT / "analysis_strengthened.json").write_text(json.dumps(stronger, indent=2), encoding="utf-8")

    lines = [
        "E1 strengthened controls analysis",
        "",
        stronger["definition"],
        "",
        "Aggregate accuracy:",
    ]
    for condition, row in aggregate.items():
        lines.append(f"  {condition}: {100 * row['point']:.1f}% [{100 * row['low']:.1f}, {100 * row['high']:.1f}]")
    lines += ["", "PCR relative to A_fresh and C:"]
    for condition, row in pcr.items():
        lines.append(f"  {condition}: {row['point']:.3f} [{row['low']:.3f}, {row['high']:.3f}]")
    lines += ["", "Paired McNemar tests:"]
    for name, row in paired.items():
        lines.append(f"  {name}: chi^2={row['chi_sq']:.3f}, p={row['p']:.3g}, n={row['n_pairs']}")
    lines += ["", "Completeness audit:"]
    for condition, row in stronger["audit"].items():
        lines.append(f"  {condition}: {row['rows']} rows, {row['unique_model_probe_pairs']} unique pairs, {row['error_rows']} error rows")
    lines.append(f"  Identity scaffold forbidden-term hits: {sum(forbidden_hits.values())}")
    (OUT / "analysis_strengthened.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote strengthened E1 analysis to {OUT}")


if __name__ == "__main__":
    main()
