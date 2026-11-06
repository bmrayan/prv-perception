from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import concurrent.futures
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

import requests


DATA_DIR = BENCHMARK_ROOT / "MuMA-ToM"
OUT_DIR = ROOT / "results" / "followup" / "derived_visibility"
QUESTIONS_PATH = DATA_DIR / "questions.json"
TEXTS_PATH = DATA_DIR / "texts.json"
DESCRIPTION_DIR = DATA_DIR / "full_episode_descriptions"

G2_PATTERNS = {
    "absent": r"[^.]*\b(?:was absent|is absent)\b[^.]*\.",
    "not_see": r"[^.]*\b(?:did not see|did not observe|does not see|does not observe)\b[^.]*\.",
    "not_present": r"[^.]*\b(?:was not present|is not present)\b[^.]*\.",
    "unaware": r"[^.]*\b(?:unaware of|has no knowledge of)\b[^.]*\.",
    "cannot_see": r"[^.]*\b(?:cannot see|could not see)\b[^.]*\.",
    "exit": r"[^.]*\b(?:left the room|left the scene|exited the room|exited the scene)\b[^.]*\.",
    "not_in_room": r"[^.]*\b(?:not in the room|outside the room)\b[^.]*\.",
    "gone_away": r"[^.]*\bwhile .* was (?:gone|away)\b[^.]*\.",
    "without_knowing": r"[^.]*\bwithout .* (?:knowing|seeing)\b[^.]*\.",
}

PERSON_MARKER_RE = re.compile(
    r"\b(?:Later(?: on)?|Meanwhile|Afterward|After that|After this|"
    r"Subsequently|Shortly after),?\s+([A-Z][a-z]+)\b"
)
PERSON_FALLBACK_RE = re.compile(
    r"[.!?]\s+([A-Z][a-z]+)\s+"
    r"(?:later|walked|went|entered|returned|grabbed|headed|also|moved|"
    r"carried|approached|picked)\b"
)
COMMON_CAPITALIZED = {
    "After", "Afterward", "Later", "Meanwhile", "Next", "Then", "Finally",
    "Shortly", "Subsequently", "This", "That", "He", "She", "The", "It",
    "As", "While", "Only", "One", "Two", "A", "An", "For", "Following",
    "Once", "First", "Second",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def find_agents(description: str) -> tuple[str | None, str | None, int | None]:
    text = normalize(description)
    first_match = re.match(r"([A-Z][a-z]+)\b", text)
    if not first_match:
        return None, None, None
    first = first_match.group(1)

    for match in PERSON_MARKER_RE.finditer(text):
        candidate = match.group(1)
        if candidate not in COMMON_CAPITALIZED and candidate != first:
            return first, candidate, match.start()
    for match in PERSON_FALLBACK_RE.finditer(text):
        candidate = match.group(1)
        if candidate not in COMMON_CAPITALIZED and candidate != first:
            return first, candidate, match.start()
    return first, None, None


def derive_annotation(video_id: str, description: str) -> dict[str, Any]:
    text = normalize(description)
    first, second, second_position = find_agents(text)
    prefix = text[:second_position] if second_position is not None else text
    away_match = re.search(
        r"\b(?:bathroom|left for|went to the bathroom|walked to the bathroom|"
        r"walked towards the bathroom|headed to the bathroom|"
        r"walked toward the bathroom)\b",
        prefix,
        flags=re.IGNORECASE,
    )
    later_text = text[second_position:] if second_position is not None else ""
    action_match = re.search(
        r"\b(?:grabbed|moved|placed|opened|carried|picked|took|retrieved|"
        r"put|removed)\b",
        later_text,
        flags=re.IGNORECASE,
    )
    eligible = bool(int(video_id) < 4000 and first and second and away_match and action_match)

    note = None
    if eligible:
        note = (
            f"Visibility annotation: {first} was away when {second} later manipulated "
            f"the object; {first} did not see that manipulation."
        )

    return {
        "video_id": video_id,
        "eligible": eligible,
        "subset": "structured_sequential" if eligible else "not_eligible",
        "first_agent": first,
        "second_agent": second,
        "second_agent_position": second_position,
        "away_evidence": away_match.group(0) if away_match else None,
        "later_action_evidence": action_match.group(0) if action_match else None,
        "visibility_note": note,
        "source_description": text,
        "scaffold_stripped": text,
        "scaffold_full_vis": f"{text} {note}" if note else text,
    }


def fetch_official_boxes(video_id: str) -> dict[str, Any]:
    url = f"https://raw.githubusercontent.com/SCAI-JHU/MuMA-ToM/main/Files/visual_data/{video_id}/boxes.json"
    response = requests.get(url, timeout=(15, 60), headers={"User-Agent": "PerceptionToM-E3"})
    response.raise_for_status()
    payload = response.json()
    video_payload = payload.get(video_id) or next(iter(payload.values()))
    character_counts = []
    track_ids: set[str] = set()
    for objects in video_payload.values():
        chars = [
            str(object_id)
            for object_id, obj in objects.items()
            if str(obj.get("class_name", "")).lower() == "character"
        ]
        character_counts.append(len(chars))
        track_ids.update(chars)
    return {
        "video_id": video_id,
        "source_url": url,
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "frame_count": len(video_payload),
        "character_count_min": min(character_counts) if character_counts else 0,
        "character_count_max": max(character_counts) if character_counts else 0,
        "character_count_mean": (sum(character_counts) / len(character_counts)) if character_counts else 0.0,
        "character_track_ids": sorted(track_ids),
    }


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    texts = json.loads(TEXTS_PATH.read_text(encoding="utf-8"))
    descriptions = {
        path.stem.removeprefix("episode_"): normalize(path.read_text(encoding="utf-8"))
        for path in DESCRIPTION_DIR.glob("episode_*.txt")
    }

    annotations = {}
    for video_id in sorted(questions, key=int):
        description = descriptions.get(video_id) or normalize(str(texts.get(video_id, "")))
        annotations[video_id] = derive_annotation(video_id, description)

    eligible_ids = [video_id for video_id, item in annotations.items() if item["eligible"]]
    official = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_official_boxes, video_id): video_id for video_id in eligible_ids}
        for future in concurrent.futures.as_completed(futures):
            video_id = futures[future]
            official[video_id] = future.result()

    rows = []
    for video_id in eligible_ids:
        entry = questions[video_id]
        item = annotations[video_id]
        for slot in sorted(entry["questions"], key=int):
            question_id = f"muma_v{int(video_id):04d}_q{int(slot)}"
            rows.append(
                {
                    "question_id": question_id,
                    "video_id": video_id,
                    "question_slot": str(slot),
                    "question": str(entry["questions"][slot]).strip(),
                    "gold_answer": str(entry["answers"][slot]).strip(),
                    "label": str(entry["labels"][slot]).strip(),
                    "scaffold_stripped": item["scaffold_stripped"],
                    "scaffold_full_vis": item["scaffold_full_vis"],
                    "visibility_note": item["visibility_note"],
                    "annotation_source": "derived_from_full_episode_description_with_official_boxes_audit",
                }
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "annotations.json").write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    (OUT_DIR / "official_boxes_audit.json").write_text(json.dumps(official, indent=2), encoding="utf-8")
    with (OUT_DIR / "scaffold_pairs.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    manifest = {
        "experiment": "E3_MuMA_derived_visibility_followup",
        "prepared_at_utc": now_iso(),
        "status": "prepared_not_run",
        "source": "data/MuMA-ToM/full_episode_descriptions plus official MuMA visual_data boxes audit",
        "questions_path_sha256": sha256(QUESTIONS_PATH),
        "texts_path_sha256": sha256(TEXTS_PATH),
        "description_file_count": len(descriptions),
        "video_count": len(questions),
        "eligible_video_count": len(eligible_ids),
        "eligible_question_count": len(rows),
        "official_boxes_audit_count": len(official),
        "conditions": ["B_full_vis", "B_vis_stripped"],
        "models": [
            "openai/gpt-4o-2024-11-20",
            "google/gemini-2.5-pro",
            "anthropic/claude-4-sonnet-20250522",
            "qwen/qwen2.5-vl-72b-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "openai/gpt-4.1-mini",
            "mistralai/ministral-8b-2512",
        ],
        "calls_planned": len(rows) * 7 * 2,
        "visibility_policy": "Include only episodes with explicit pre-second-agent away evidence and a later manipulation; all other episodes are excluded from the primary intervention.",
        "stripped_definition": "The original benchmark interaction summary, unchanged.",
        "full_definition": "The original benchmark interaction summary plus one derived temporal visibility sentence.",
        "g2_pattern_hits_in_full": {
            name: sum(bool(re.search(pattern, item["scaffold_full_vis"], flags=re.IGNORECASE)) for item in annotations.values() if item["eligible"])
            for name, pattern in G2_PATTERNS.items()
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "E3 MuMA derived visibility follow-up preparation",
        "",
        f"All MuMA videos: {len(questions)}",
        f"High-confidence eligible videos: {len(eligible_ids)}",
        f"Eligible questions: {len(rows)}",
        f"Official boxes audits: {len(official)}",
        f"Planned calls: {manifest['calls_planned']}",
        "",
        "G.2 hits in eligible full scaffolds:",
    ]
    lines.extend(f"  {name}: {count}" for name, count in manifest["g2_pattern_hits_in_full"].items())
    lines.extend([
        "",
        "This is a derived follow-up condition, not a reconstruction of the historical MuMA B_full prompts.",
    ])
    (OUT_DIR / "preparation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
