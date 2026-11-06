"""Complete the remaining local README follow-up items."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "followup" / "additional"
MODELS = [
    "gpt-4o-2024-11-20",
    "gemini-2.5-pro-preview",
    "claude-sonnet-4",
    "qwen-2.5-vl-72b-instruct",
    "llama-4-maverick",
    "gpt-4.1-mini",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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
    rows = []
    for shard in manifest["shards"]:
        rows.extend(json.loads((path.parent / str(shard)).read_text(encoding="utf-8")))
    return rows


def exact_mcnemar(a: list[int], b: list[int]) -> dict[str, Any]:
    b_only = sum(x == 1 and y == 0 for x, y in zip(a, b))
    c_only = sum(x == 0 and y == 1 for x, y in zip(a, b))
    n = b_only + c_only
    if not n:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(min(b_only, c_only) + 1))
        tail += sum(math.comb(n, k) for k in range(n - min(b_only, c_only), n + 1))
        p = min(1.0, tail / (2**n))
    chi = 0.0 if not n else (abs(b_only - c_only) - 1) ** 2 / n
    return {"n_pairs": len(a), "A_only_correct": b_only, "condition_only_correct": c_only, "chi_sq_cc": chi, "exact_two_sided_p": p}


def e2_mcnemar_and_spotcheck() -> None:
    probes = {str(row["id"]): row for row in json.loads((ROOT / "data" / "abstract2d" / "probes.json").read_text(encoding="utf-8")) if int(row.get("stage", 0)) == 3}
    e2 = [row for row in load_jsonl(ROOT / "results" / "followup" / "e2" / "results.jsonl") if str(row.get("model")) in MODELS and int(row.get("stage", 0)) == 3]
    phase1 = [row for row in load_sharded(ROOT / "results" / "reference" / "abstract" / "original" / "v5_phase1_results.json") if str(row.get("model")) in MODELS and int(row.get("stage", 0)) == 3]
    rows = e2 + phase1
    by: dict[tuple[str, str, str], int] = {}
    for row in rows:
        condition = str(row.get("condition"))
        if condition in {"A", "D_mismatch", "D_shuffled", "D_noise"}:
            by[(str(row.get("model")), str(row.get("probe_id")), condition)] = int(bool(row.get("correct")))
    tests = {}
    for condition in ("D_mismatch", "D_shuffled", "D_noise"):
        a: list[int] = []
        b: list[int] = []
        for model in MODELS:
            for probe_id in sorted(probes):
                key_a = (model, probe_id, "A")
                key_b = (model, probe_id, condition)
                if key_a in by and key_b in by:
                    a.append(by[key_a]); b.append(by[key_b])
        tests[f"A_vs_{condition}"] = exact_mcnemar(a, b)

    phase2 = [row for row in load_sharded(ROOT / "results" / "reference" / "abstract" / "original" / "v5_phase2_results.json") if int(row.get("stage", 0)) == 3 and str(row.get("model")) in MODELS]
    spotcheck = []
    for condition in ("D_mismatch", "D_shuffled", "D_noise"):
        selected = [row for row in phase2 if str(row.get("condition")) == condition][:3]
        for row in selected:
            timestamp = str(row.get("timestamp", ""))
            try:
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                timestamp_ok = True
            except ValueError:
                timestamp_ok = False
            raw = row.get("raw_api")
            spotcheck.append({
                "condition": condition,
                "model": row.get("model"),
                "probe_id": row.get("probe_id"),
                "timestamp": timestamp,
                "timestamp_parseable": timestamp_ok,
                "response": row.get("response"),
                "correct": row.get("correct"),
                "raw_api_present": isinstance(raw, dict),
                "raw_api_has_usage": isinstance(raw, dict) and isinstance(raw.get("usage"), dict),
                "prompt_length": len(str(row.get("prompt", ""))),
            })
    audit = {
        "generated_at": datetime.now().isoformat(),
        "six_model_mcnemar": tests,
        "phase2_spotcheck": spotcheck,
        "spotcheck_all_timestamps_parseable": all(row["timestamp_parseable"] for row in spotcheck),
        "spotcheck_all_raw_usage_present": all(row["raw_api_has_usage"] for row in spotcheck),
    }
    (OUT / "e2_six_model_mcnemar_and_provenance.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def code_e1_failure_modes() -> None:
    source = json.loads((OUT / "e1_b_static_failure_transcripts.json").read_text(encoding="utf-8"))
    coded = []
    for row in source["failure_transcripts"]:
        response = str(row.get("response", ""))
        expected = str(row.get("expected", "")).lower()
        normalized = re.sub(r"\s+", " ", response.strip().lower())
        if expected in {"left", "right"} and normalized.startswith(("left", "right")) and not normalized.startswith(expected):
            mode = "final-state substitution / reality bias"
            evidence = "The response gives the opposite final location directly, without preserving the absent agent's last observed state."
        elif expected == "different" and (not normalized.endswith("different") or len(normalized) > 200):
            mode = "overlong or truncated answer before the required categorical decision"
            evidence = "The transcript traces observations but does not finish with the required categorical answer."
        else:
            mode = "other scaffold-limited error"
            evidence = "The saved response is incorrect but does not meet the two dominant transcript patterns above."
        coded.append(dict(row, failure_mode=mode, evidence=evidence))
    summary = {}
    for model in source["models"]:
        subset = [row for row in coded if row["model"] == model]
        summary[model] = dict(Counter(row["failure_mode"] for row in subset))
    (OUT / "e1_b_static_failure_modes.json").write_text(json.dumps({"summary": summary, "coded_transcripts": coded, "coding_basis": "Transcript text and required answer format; labels are descriptive, not inferred from aggregate accuracy."}, indent=2), encoding="utf-8")


def main() -> None:
    e2_mcnemar_and_spotcheck()
    code_e1_failure_modes()
    print("Completed E2 McNemar/provenance and E1 transcript coding.")


if __name__ == "__main__":
    main()
