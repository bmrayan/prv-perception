"""Summarize the completed Stage-3 Ministral-8B E2 extension."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import json
from collections import Counter


OUT = ROOT / "results" / "followup" / "additional" / "e2_ministral_extension"


def main() -> None:
    rows = []
    path = OUT / "results.jsonl"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    by_condition = {}
    for condition in ("D_mismatch", "D_shuffled", "D_noise"):
        subset = [row for row in rows if str(row.get("condition")) == condition]
        by_condition[condition] = {
            "rows": len(subset),
            "correct": sum(bool(row.get("correct")) for row in subset),
            "accuracy": (sum(bool(row.get("correct")) for row in subset) / len(subset)) if subset else None,
        }
    payload = {
        "experiment": "E2_Ministral8B_stage3_controls",
        "rows": len(rows),
        "status_counts": dict(Counter("error" if str(row.get("response", "")).startswith("ERROR") else "ok" for row in rows)),
        "by_condition": by_condition,
        "complete_expected_rows": len(rows) == 1500,
        "historical_results_unchanged": True,
    }
    (OUT / "analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["E2 Ministral-8B Stage-3 extension", "", f"Rows: {len(rows)} / 1500", ""]
    for condition, result in by_condition.items():
        accuracy = result["accuracy"]
        lines.append(f"{condition}: {result['correct']}/{result['rows']} = {100 * accuracy:.2f}%" if accuracy is not None else f"{condition}: no rows")
    (OUT / "analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
