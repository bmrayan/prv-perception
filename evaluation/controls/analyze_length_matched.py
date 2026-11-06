from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy.stats import rankdata


OUT = ROOT / "results" / "followup" / "e4"
CONDITIONS = ["A", "B_auto", "B_vlm", "B_static", "B_percept", "B_full", "B_vis_stripped", "C", "D_mismatch", "D_shuffled", "D_noise"]


def ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, including ties, for a tie-aware Spearman estimate."""
    return rankdata(values, method="average")


def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    if len(x) <= 1 or np.std(x) == 0 or np.std(y) == 0:
        pearson = None
    else:
        pearson = float(np.corrcoef(x, y)[0, 1])
    rx, ry = ranks(x), ranks(y)
    if len(x) <= 1 or np.std(rx) == 0 or np.std(ry) == 0:
        spearman = None
    else:
        spearman = float(np.corrcoef(rx, ry)[0, 1])
    return {"pearson_r": pearson, "spearman_r": spearman, "n": int(len(x))}


def permutation_p(x: np.ndarray, y: np.ndarray, statistic: str, seed: int, iterations: int = 10_000) -> float | None:
    observed = correlation(x, y)[statistic]
    if observed is None:
        return None
    base_x = ranks(x) if statistic == "spearman_r" else x.astype(float)
    base_y = ranks(y) if statistic == "spearman_r" else y.astype(float)
    x_centered = base_x - base_x.mean()
    x_norm = np.linalg.norm(x_centered)
    rng = np.random.default_rng(seed)
    extreme = 0
    batch_size = 1000
    for start in range(0, iterations, batch_size):
        size = min(batch_size, iterations - start)
        permutations = np.vstack([rng.permutation(len(base_y)) for _ in range(size)])
        shuffled = base_y[permutations]
        centered = shuffled - shuffled.mean(axis=1, keepdims=True)
        denominators = x_norm * np.linalg.norm(centered, axis=1)
        values = (centered @ x_centered) / denominators
        extreme += int(np.count_nonzero(np.abs(values) >= abs(observed)))
    return float((extreme + 1) / (iterations + 1))


def main() -> None:
    rows = [json.loads(line) for line in (OUT / "scaffold_lengths.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    base = json.loads((OUT / "analysis.json").read_text(encoding="utf-8"))
    summary = {str(row["condition"]): row for row in base["summary"]}

    condition_x = np.array([summary[c]["mean_scaffold_tokens"] for c in CONDITIONS], dtype=float)
    condition_y = np.array([summary[c]["gain_vs_A"] for c in CONDITIONS], dtype=float)
    condition_corr = correlation(condition_x, condition_y)
    condition_corr["pearson_p_permutation"] = permutation_p(condition_x, condition_y, "pearson_r", 17001)
    condition_corr["spearman_p_permutation"] = permutation_p(condition_x, condition_y, "spearman_r", 17002)

    within: dict[str, Any] = {}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        by_probe: dict[str, list[dict[str, Any]]] = {}
        for row in subset:
            by_probe.setdefault(str(row["probe_id"]), []).append(row)
        x = np.array([np.mean([r["scaffold_tokens"] for r in group]) for group in by_probe.values()], dtype=float)
        y = np.array([np.mean([r["correct"] for r in group]) for group in by_probe.values()], dtype=float)
        corr = correlation(x, y)
        if len(x) > 1:
            corr["pearson_p_permutation"] = permutation_p(x, y, "pearson_r", 18000 + CONDITIONS.index(condition), iterations=3000)
            corr["spearman_p_permutation"] = permutation_p(x, y, "spearman_r", 19000 + CONDITIONS.index(condition), iterations=3000)
        within[condition] = corr

    result = {
        "experiment": "E4_scaffold_length_strengthened",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "condition_level": condition_corr,
        "within_condition_probe_level": within,
        "interpretation": "The matched-content controls and condition-level comparison are descriptive; scaffold length is not treated as a causal variable.",
    }
    (OUT / "analysis_stronger.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "E4 strengthened analysis",
        "",
        f"Condition-level length/gain Pearson r={condition_corr['pearson_r']}, permutation p={condition_corr['pearson_p_permutation']}; "
        f"Spearman r={condition_corr['spearman_r']}, permutation p={condition_corr['spearman_p_permutation']}.",
        "",
        "Within-condition probe-level correlations (scaffold length versus mean accuracy):",
    ]
    for condition in CONDITIONS:
        row = within[condition]
        lines.append(f"  {condition}: Pearson r={row['pearson_r']}, p={row.get('pearson_p_permutation')}; Spearman r={row['spearman_r']}, p={row.get('spearman_p_permutation')}; n={row['n']}")
    (OUT / "analysis_stronger.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote strengthened E4 analysis to {OUT}")


if __name__ == "__main__":
    main()
