"""Prepare the saved-data follow-ups for E1, E2, E4, and reviewer 11Gb."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "followup" / "additional"
SIX_MODELS = [
    "gpt-4o-2024-11-20",
    "gemini-2.5-pro-preview",
    "claude-sonnet-4",
    "qwen-2.5-vl-72b-instruct",
    "llama-4-maverick",
    "gpt-4.1-mini",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_sharded(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for shard in manifest.get("shards", []):
        rows.extend(json.loads((path.parent / str(shard)).read_text(encoding="utf-8")))
    return rows


def bootstrap(values: list[int], seed: int, iterations: int = 10000) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if not len(array):
        return {"n": 0, "point": float("nan"), "low": float("nan"), "high": float("nan")}
    draws = np.mean(array[rng.integers(0, len(array), size=(iterations, len(array)))], axis=1)
    return {"n": int(len(array)), "point": float(array.mean()), "low": float(np.quantile(draws, 0.025)), "high": float(np.quantile(draws, 0.975))}


def e2_followup() -> dict[str, Any]:
    probes = {str(row["id"]): row for row in json.loads((ROOT / "data" / "abstract2d" / "probes.json").read_text(encoding="utf-8")) if int(row.get("stage", 0)) == 3}
    e2_rows = [row for row in load_jsonl(ROOT / "results" / "followup" / "e2" / "results.jsonl") if str(row.get("model")) in SIX_MODELS and int(row.get("stage", 0)) == 3]
    phase1 = [row for row in load_sharded(ROOT / "results" / "reference" / "abstract" / "original" / "v5_phase1_results.json") if str(row.get("model")) in SIX_MODELS and int(row.get("stage", 0)) == 3]
    rows_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in e2_rows + phase1:
        if str(row.get("condition")) in {"A", "B_percept", "C", "D_mismatch", "D_shuffled", "D_noise"}:
            rows_by_condition[str(row["condition"])].append(row)

    accuracy = {condition: bootstrap([int(bool(row.get("correct"))) for row in rows_by_condition[condition]], 9100 + index) for index, condition in enumerate(("A", "D_mismatch", "D_noise", "D_shuffled", "B_percept", "C"))}
    baseline = accuracy["A"]["point"]
    delta_vs_a = {condition: accuracy[condition]["point"] - baseline for condition in accuracy}

    expected = {pid: str(probe.get("expected", "")).lower() for pid, probe in probes.items()}
    agreement: dict[str, list[int]] = {"agree": [], "disagree": []}
    for pid, probe in probes.items():
        partner = probes.get(str(probe.get("mismatch_partner_id")))
        if partner is None:
            continue
        agreement["agree" if expected[pid] == str(partner.get("expected", "")).lower() else "disagree"].append(pid)
    mismatch_split: dict[str, Any] = {}
    for group, ids in agreement.items():
        allowed = set(ids)
        group_rows = [row for row in rows_by_condition["D_mismatch"] if str(row.get("probe_id")) in allowed]
        mismatch_split[group] = {
            "probe_count": len(allowed),
            "rows": len(group_rows),
            "accuracy": bootstrap([int(bool(row.get("correct"))) for row in group_rows], 9200 + len(group)),
            "by_model": {model: bootstrap([int(bool(row.get("correct"))) for row in group_rows if str(row.get("model")) == model], 9300 + index) for index, model in enumerate(SIX_MODELS)},
        }

    shuffled_rows = rows_by_condition["D_shuffled"]
    prefix_counts = Counter(len(re.findall(r"Panel\s+\d+\s*:", str(row.get("prompt", "")), flags=re.IGNORECASE)) for row in shuffled_rows)
    shuffled_audit = {
        "rows": len(shuffled_rows),
        "rows_with_panel_prefixes": sum(count for number, count in prefix_counts.items() if number > 0),
        "prefix_count_distribution": dict(sorted(prefix_counts.items())),
        "interpretation": "Panel labels are retained in the saved D_shuffled prompts; this control should not be described as removing all recoverable panel ordering cues.",
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "six_model_overlap": SIX_MODELS,
        "counts": {condition: len(rows_by_condition[condition]) for condition in accuracy},
        "accuracy": accuracy,
        "delta_vs_A": delta_vs_a,
        "mismatch_expected_answer_agreement": mismatch_split,
        "D_shuffled_prefix_audit": shuffled_audit,
        "reference_point_estimates_from_plan": {"A": 0.772, "D_mismatch": 0.753, "D_noise": 0.794, "D_shuffled": 0.796, "B_percept": 0.889, "C": 0.957},
    }


def e1_transcripts() -> dict[str, Any]:
    rows = [row for row in load_jsonl(ROOT / "results" / "followup" / "e1" / "results.jsonl") if str(row.get("condition")) == "B_static" and not bool(row.get("correct"))]
    selected: list[dict[str, Any]] = []
    for model in ("qwen/qwen2.5-vl-72b-instruct", "anthropic/claude-4-sonnet-20250522"):
        selected.extend([row for row in rows if str(row.get("model")) == model][:50])
    payload_rows = [{key: row.get(key) for key in ("model", "probe_id", "template", "expected", "response", "prompt", "scaffold", "correct")} for row in selected]
    return {"sample_size_per_model": 50, "models": ["qwen/qwen2.5-vl-72b-instruct", "anthropic/claude-4-sonnet-20250522"], "failure_transcripts": payload_rows, "note": "These are direct saved B_static failure transcripts for qualitative coding; no failure label is inferred from accuracy alone."}


def reviewer_11gb_transcript() -> dict[str, Any]:
    rows = [row for row in load_sharded(ROOT / "results" / "reference" / "abstract" / "original" / "v5_phase1_results.json") if int(row.get("stage", 0)) == 3 and str(row.get("template")) == "two_rooms" and str(row.get("condition")) == "C" and not bool(row.get("correct"))]
    if not rows:
        return {"found": False, "reason": "No saved Stage 3 two_rooms Condition-C failure row was found."}
    row = rows[0]
    return {"found": True, "model": row.get("model"), "probe_id": row.get("probe_id"), "expected": row.get("expected"), "response": row.get("response"), "prompt": row.get("prompt"), "raw_api": row.get("raw_api")}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    e2 = e2_followup()
    (OUT / "e2_six_model_followup.json").write_text(json.dumps(e2, indent=2), encoding="utf-8")
    e1 = e1_transcripts()
    (OUT / "e1_b_static_failure_transcripts.json").write_text(json.dumps(e1, indent=2), encoding="utf-8")
    reviewer = reviewer_11gb_transcript()
    (OUT / "reviewer_11gb_two_rooms_condition_c_failure.json").write_text(json.dumps(reviewer, indent=2), encoding="utf-8")

    e4_rows = [row for row in load_jsonl(ROOT / "results" / "followup" / "e4" / "scaffold_lengths.jsonl") if str(row.get("model")) in SIX_MODELS]
    e4_summary: dict[str, Any] = {}
    by_condition: dict[str, list[int]] = defaultdict(list)
    for row in e4_rows:
        by_condition[str(row.get("condition"))].append(int(bool(row.get("correct"))))
    a = float(np.mean(by_condition.get("A", [float("nan")])));
    for condition, values in sorted(by_condition.items()):
        stat = bootstrap(values, 9500 + len(e4_summary))
        stat["gain_vs_A"] = stat["point"] - a
        e4_summary[condition] = stat
    (OUT / "e4_six_model_gain_followup.json").write_text(json.dumps({"models": SIX_MODELS, "conditions": e4_summary, "note": "Recomputed from the saved E4 scaffold-length rows on the same six-model overlap."}, indent=2), encoding="utf-8")
    print("Deterministic follow-ups written.")


if __name__ == "__main__":
    main()
