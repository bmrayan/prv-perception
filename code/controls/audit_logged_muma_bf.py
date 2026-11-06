from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results" / "reference" / "muma"
OUT = ROOT / "results" / "followup" / "e3"

PATTERNS = {
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


def scaffold_from_row(row: dict[str, Any]) -> str:
    description = str(row.get("description") or "").strip()
    if description:
        return description
    prompt = str(row.get("prompt") or "")
    marker = "Interaction summary:\n"
    if marker in prompt:
        text = prompt.split(marker, 1)[1]
        return text.split("\n\nQuestion:", 1)[0].strip()
    return ""


def main() -> None:
    rows = 0
    sources = Counter()
    models = Counter()
    scaffolds: dict[str, set[str]] = defaultdict(set)
    pattern_hits = Counter()
    hit_examples: dict[str, dict[str, Any]] = {}

    files = [
        path
        for path in RESULT_ROOT.rglob("*.jsonl")
        if "parser_cache" not in path.name and "_frame_cache" not in str(path)
    ]
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(row.get("condition")) != "B_full":
                    continue
                rows += 1
                sources[path.name] += 1
                models[str(row.get("model"))] += 1
                question_id = str(row.get("question_id"))
                scaffold = scaffold_from_row(row)
                scaffolds[question_id].add(scaffold)

    nonempty = {question_id: next(iter(values)) for question_id, values in scaffolds.items() if any(values)}
    duplicate_variants = {
        question_id: sorted(values)
        for question_id, values in scaffolds.items()
        if len(values) > 1
    }
    for question_id, scaffold in nonempty.items():
        for name, pattern in PATTERNS.items():
            match = re.search(pattern, scaffold, flags=re.IGNORECASE)
            if match:
                pattern_hits[name] += 1
                hit_examples.setdefault(
                    name,
                    {"question_id": question_id, "match": match.group(0), "scaffold": scaffold},
                )

    local_questions = json.loads((ROOT / "data" / "MuMA-ToM" / "questions.json").read_text(encoding="utf-8"))
    annotation_keys = sorted({key for entry in local_questions.values() for key in entry if key not in {"description", "questions", "answers", "labels"}})
    local_full_descriptions = len(list((ROOT / "data" / "MuMA-ToM" / "full_episode_descriptions").glob("*.txt")))

    payload = {
        "experiment": "E3_logged_MuMA_ToM_B_full_visibility_audit",
        "source_rows": rows,
        "unique_questions": len(scaffolds),
        "nonempty_unique_scaffolds": len(nonempty),
        "source_file_counts": dict(sorted(sources.items())),
        "model_row_counts": dict(sorted(models.items())),
        "questions_with_multiple_scaffold_variants": len(duplicate_variants),
        "pattern_hits_by_question": dict(sorted(pattern_hits.items())),
        "questions_with_any_pattern_hit": len({question_id for question_id, scaffold in nonempty.items() if any(re.search(pattern, scaffold, flags=re.IGNORECASE) for pattern in PATTERNS.values())}),
        "pattern_definitions": PATTERNS,
        "hit_examples": hit_examples,
        "local_annotation_keys_beyond_standard_fields": annotation_keys,
        "local_full_episode_description_files": local_full_descriptions,
        "decision": "do_not_run_original_E3_strip_rerun" if not pattern_hits else "prepare_E3_strip_rerun",
        "reason": "The saved B_full scaffold text contains no sentence matching any of the nine G.2 visibility/absence patterns." if not pattern_hits else "At least one saved B_full scaffold contains removable G.2 visibility/absence content.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logged_bf_visibility_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "E3 logged MuMA-ToM B_full visibility audit",
        "",
        f"Saved B_full rows: {rows}",
        f"Unique questions: {len(scaffolds)}",
        f"Non-empty unique scaffolds: {len(nonempty)}",
        f"Questions with multiple scaffold variants: {len(duplicate_variants)}",
        "",
        "G.2 pattern hits by unique question:",
    ]
    lines.extend(f"  {name}: {pattern_hits.get(name, 0)}" for name in PATTERNS)
    lines.extend(
        [
            f"Questions with any G.2 hit: {payload['questions_with_any_pattern_hit']}",
            "",
            f"Local full-episode description files: {local_full_descriptions}",
            f"Local annotation keys beyond standard question fields: {', '.join(annotation_keys) if annotation_keys else 'none'}",
            "",
            f"Decision: {payload['decision']}",
            f"Reason: {payload['reason']}",
        ]
    )
    (OUT / "logged_bf_visibility_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
