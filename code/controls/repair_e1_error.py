from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
import run_e1_static_scaffold as base  # noqa: E402


PROBE_ID = "s3_sally_anne_reality_014"
MODEL = "google/gemini-2.5-pro"
OUT = ROOT / "results" / "followup" / "e1" / "fresh_a" / "repairs.jsonl"
PROBES = [p for p in base.load_json_array(ROOT / "data" / "abstract2d" / "probes.json") if str(p["id"]) == PROBE_ID]


def main() -> None:
    if len(PROBES) != 1:
        raise ValueError(f"Expected one probe, found {len(PROBES)}")
    probe = PROBES[0]
    keys = base.load_keys(["code/config/keys/API_Key1.txt"])
    import requests

    prompt = str(probe["question_A"])
    with requests.Session() as session:
        response, raw = base.call_model(MODEL, prompt, ROOT / str(probe["image_path"]), keys[0], session)
    row = {
        "experiment": "E1_strengthened_controls",
        "condition": "A_fresh",
        "model": MODEL,
        "probe_id": PROBE_ID,
        "template": str(probe["template"]),
        "stage": int(probe["stage"]),
        "variant_index": int(probe["variant_index"]),
        "image_path": str(probe["image_path"]),
        "prompt": prompt,
        "scaffold": None,
        "response": response,
        "correct": bool(base.answer_correct(probe, response)),
        "expected": str(probe["expected"]),
        "answer_kind": str(probe["answer_kind"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_api": raw,
        "repair_source": "transient network-error retry",
    }
    OUT.write_text(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote repair for {MODEL} / {PROBE_ID}; correct={row['correct']}; raw_api={'present' if raw is not None else 'missing'}")


if __name__ == "__main__":
    main()
