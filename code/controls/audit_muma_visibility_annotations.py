from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "MuMA-ToM"
OUTPUT_DIR = ROOT / "results" / "followup" / "e3"
PATTERNS = {
    "visible": r"\bvisible\b",
    "visibility": r"\bvisibility\b",
    "see_or_saw": r"\b(?:see|saw|seen)\b",
    "watch_or_witness": r"\b(?:watch|watched|witness|witnessed)\b",
    "observe": r"\b(?:observe|observed|observation)\b",
    "presence_absence": r"\b(?:present|absent|absence)\b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    texts = json.loads((DATA_DIR / "texts.json").read_text(encoding="utf-8"))
    questions = json.loads((DATA_DIR / "questions.json").read_text(encoding="utf-8"))
    description_dir = DATA_DIR / "full_episode_descriptions"
    descriptions = {path.stem.removeprefix("episode_"): path.read_text(encoding="utf-8") for path in description_dir.glob("episode_*.txt")}
    combined = {video_id: f"{texts.get(video_id, '')}\n{descriptions.get(video_id, '')}" for video_id in sorted(texts)}

    pattern_counts = {
        name: sum(bool(re.search(pattern, text, flags=re.IGNORECASE)) for text in combined.values())
        for name, pattern in PATTERNS.items()
    }
    explicit_visibility_fields = []
    for video_id, record in questions.items():
        for key in record:
            if any(term in str(key).lower() for term in ("visib", "absen", "witness", "access", "seen")):
                explicit_visibility_fields.append(f"questions.json:{video_id}:{key}")

    report = {
        "experiment": "E3_MuMA_visibility_stripped",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "videos": len(combined),
        "questions": sum(len(record.get("questions", {})) for record in questions.values()),
        "description_files": len(descriptions),
        "explicit_visibility_annotation_fields": explicit_visibility_fields,
        "description_pattern_video_counts": pattern_counts,
        "can_run_as_specified": bool(explicit_visibility_fields),
        "conclusion": "The local MuMA-ToM package has no explicit visibility/absence annotation field. Removing occasional dialogue terms would not be the specified visibility-stripping ablation.",
        "source_hashes": {
            "questions": sha256(DATA_DIR / "questions.json"),
            "texts": sha256(DATA_DIR / "texts.json"),
        },
    }
    (OUTPUT_DIR / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "E3 MuMA-ToM visibility annotation audit",
        "",
        f"Videos: {report['videos']}",
        f"Questions: {report['questions']}",
        f"Description files: {report['description_files']}",
        "",
        "Description-level cue counts by video:",
    ]
    lines.extend(f"  {name}: {count}" for name, count in pattern_counts.items())
    lines.extend([
        "",
        f"Explicit visibility annotation fields: {len(explicit_visibility_fields)}",
        "",
        report["conclusion"],
    ])
    (OUTPUT_DIR / "audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps({
        "experiment": "E3_MuMA_visibility_stripped",
        "status": "not_run",
        "reason": "Required visibility/absence annotations are not present in the local MuMA-ToM data.",
        "new_api_calls": 0,
        "source": "data/MuMA-ToM",
    }, indent=2), encoding="utf-8")
    print(f"E3 audit written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
