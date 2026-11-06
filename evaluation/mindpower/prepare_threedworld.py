from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import argparse
from collections import Counter

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from mindpower.common import (
    DATA_ROOT,
    DEFAULT_PREP_DIR,
    build_manifest_rows,
    extract_storyboard_bundle,
    now_iso,
    relative_to_root,
    write_json,
    write_manifest_csv,
    write_manifest_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the MindPower ThreeDWorld phase-1 manifest and storyboard contact sheets."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--prep-dir", type=Path, default=DEFAULT_PREP_DIR)
    args = parser.parse_args()

    args.prep_dir.mkdir(parents=True, exist_ok=True)
    rows = build_manifest_rows(args.data_root)
    storyboard_rows = []
    for row in rows:
        video_path = args.data_root / row["sample_id"] / Path(row["video_path"]).name
        storyboard = extract_storyboard_bundle(video_path, row["sample_id"], prep_dir=args.prep_dir)
        enriched = dict(row)
        enriched["storyboard_path"] = storyboard["contact_sheet_path"]
        enriched["sampled_frame_paths"] = storyboard["frame_paths"]
        enriched["sampled_timestamps_sec"] = storyboard["timestamps_sec"]
        enriched["frame_total"] = storyboard["frame_total"]
        enriched["duration_sec"] = storyboard["duration_sec"]
        storyboard_rows.append(enriched)

    manifest_jsonl = args.prep_dir / "threedworld_phase1_manifest.jsonl"
    manifest_csv = args.prep_dir / "threedworld_phase1_manifest.csv"
    write_manifest_jsonl(manifest_jsonl, storyboard_rows)
    write_manifest_csv(manifest_csv, storyboard_rows)

    scenario_counts = Counter(row["scenario_family"] for row in storyboard_rows)
    validation = {
        "generated_at_utc": now_iso(),
        "data_root": relative_to_root(args.data_root),
        "prep_dir": relative_to_root(args.prep_dir),
        "manifest_jsonl": relative_to_root(manifest_jsonl),
        "manifest_csv": relative_to_root(manifest_csv),
        "sample_count": len(storyboard_rows),
        "scenario_family_counts": dict(sorted(scenario_counts.items())),
        "all_have_storyboards": all(Path(args.prep_dir / "storyboards" / f"sample_{int(row['sample_id']):02d}" / "storyboard_contact_sheet.png").exists() for row in storyboard_rows),
        "all_have_12_sampled_frames": all(len(row["sampled_frame_paths"]) == 12 for row in storyboard_rows),
        "all_have_required_references": all(
            row.get("description_ref")
            and row.get("belief_ref")
            and row.get("desire_ref")
            and row.get("intention_ref")
            and row.get("decision_ref")
            and row.get("action_ref")
            for row in storyboard_rows
        ),
    }
    write_json(args.prep_dir / "threedworld_phase1_validation.json", validation)

    print(f"Wrote manifest: {manifest_jsonl}")
    print(f"Wrote CSV: {manifest_csv}")
    print(f"Samples: {validation['sample_count']}")
    print(f"Scenario families: {validation['scenario_family_counts']}")
    print(f"All storyboards present: {validation['all_have_storyboards']}")
    print(f"All sampled frame counts are 12: {validation['all_have_12_sampled_frames']}")
    print(f"All references present: {validation['all_have_required_references']}")


if __name__ == "__main__":
    main()
