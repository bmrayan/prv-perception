from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import argparse
from collections import Counter

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from mindpower.benchmark_remainder import (
    DEFAULT_REMAINDER_PREP_DIR,
    THREEDWORLD_ROOT,
    VIRTUALHOME_ROOT,
    build_remainder_rows,
    build_virtualhome_render_jobs,
    filter_text_eval_rows,
    filter_virtualhome_script_rows,
    relative_to_root,
    write_json,
    write_jsonl,
    write_union_csv,
)
from mindpower.common import now_iso


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare manifests for the remaining MindPower benchmark data.")
    parser.add_argument("--threedworld-root", type=Path, default=THREEDWORLD_ROOT)
    parser.add_argument("--virtualhome-root", type=Path, default=VIRTUALHOME_ROOT)
    parser.add_argument("--prep-dir", type=Path, default=DEFAULT_REMAINDER_PREP_DIR)
    parser.add_argument("--include-script-context-in-text-manifest", action="store_true")
    args = parser.parse_args()

    args.prep_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped_virtualhome = build_remainder_rows(
        threedworld_root=args.threedworld_root,
        virtualhome_root=args.virtualhome_root,
    )
    text_rows = filter_text_eval_rows(rows, include_script_context=args.include_script_context_in_text_manifest)
    script_rows = filter_virtualhome_script_rows(rows)
    render_jobs = build_virtualhome_render_jobs(rows, args.prep_dir / "virtualhome_renders")

    combined_jsonl = args.prep_dir / "remainder_manifest.jsonl"
    combined_csv = args.prep_dir / "remainder_manifest.csv"
    text_jsonl = args.prep_dir / "remainder_text_eval_manifest.jsonl"
    text_csv = args.prep_dir / "remainder_text_eval_manifest.csv"
    script_jsonl = args.prep_dir / "virtualhome_script_manifest.jsonl"
    script_csv = args.prep_dir / "virtualhome_script_manifest.csv"
    jobs_jsonl = args.prep_dir / "virtualhome_render_jobs.jsonl"
    jobs_csv = args.prep_dir / "virtualhome_render_jobs.csv"

    write_jsonl(combined_jsonl, rows)
    write_union_csv(combined_csv, rows)
    write_jsonl(text_jsonl, text_rows)
    write_union_csv(text_csv, text_rows)
    write_jsonl(script_jsonl, script_rows)
    write_union_csv(script_csv, script_rows)
    write_jsonl(jobs_jsonl, render_jobs)
    write_union_csv(jobs_csv, render_jobs)

    source_counts = Counter(str(row["source_group"]) for row in rows)
    input_mode_counts = Counter(str(row["input_mode"]) for row in rows)
    validation = {
        "generated_at_utc": now_iso(),
        "threedworld_root": relative_to_root(args.threedworld_root),
        "virtualhome_root": relative_to_root(args.virtualhome_root),
        "prep_dir": relative_to_root(args.prep_dir),
        "combined_manifest_jsonl": relative_to_root(combined_jsonl),
        "text_manifest_jsonl": relative_to_root(text_jsonl),
        "virtualhome_script_manifest_jsonl": relative_to_root(script_jsonl),
        "virtualhome_render_jobs_jsonl": relative_to_root(jobs_jsonl),
        "row_count_total": len(rows),
        "row_count_text_eval": len(text_rows),
        "row_count_virtualhome_script": len(script_rows),
        "source_group_counts": dict(sorted(source_counts.items())),
        "input_mode_counts": dict(sorted(input_mode_counts.items())),
        "skipped_virtualhome_sample_ids": skipped_virtualhome,
        "all_have_references": all(
            row.get("description_ref")
            and row.get("belief_ref")
            and row.get("desire_ref")
            and row.get("intention_ref")
            and row.get("decision_ref")
            and row.get("action_ref")
            for row in rows
        ),
    }
    write_json(args.prep_dir / "remainder_validation.json", validation)

    print(f"Wrote combined manifest: {combined_jsonl}")
    print(f"Wrote text-eval manifest: {text_jsonl}")
    print(f"Wrote script manifest: {script_jsonl}")
    print(f"Wrote render jobs: {jobs_jsonl}")
    print(f"Rows total: {validation['row_count_total']}")
    print(f"Text-eval rows: {validation['row_count_text_eval']}")
    print(f"VirtualHome script rows: {validation['row_count_virtualhome_script']}")
    print(f"Source groups: {validation['source_group_counts']}")
    print(f"Skipped VirtualHome samples: {len(skipped_virtualhome)}")


if __name__ == "__main__":
    main()
