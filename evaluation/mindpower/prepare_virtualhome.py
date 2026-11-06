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
    build_virtualhome_visual_eval_rows,
    filter_virtualhome_script_rows,
    write_jsonl,
    write_union_csv,
)
from mindpower.common import now_iso, read_jsonl, relative_to_root, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the storyboard-backed VirtualHome visual eval manifest.")
    parser.add_argument(
        "--remainder-manifest-path",
        type=Path,
        default=DEFAULT_REMAINDER_PREP_DIR / "remainder_manifest.jsonl",
    )
    parser.add_argument(
        "--render-results-paths",
        nargs="+",
        type=Path,
        required=True,
        help="One or more render_results.jsonl files to merge, newest reruns included.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_REMAINDER_PREP_DIR / "virtualhome_visual_eval_manifest.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_REMAINDER_PREP_DIR / "virtualhome_visual_eval_manifest.csv",
    )
    parser.add_argument(
        "--dropped-json",
        type=Path,
        default=DEFAULT_REMAINDER_PREP_DIR / "virtualhome_visual_eval_dropped.json",
    )
    parser.add_argument(
        "--validation-json",
        type=Path,
        default=DEFAULT_REMAINDER_PREP_DIR / "virtualhome_visual_eval_validation.json",
    )
    args = parser.parse_args()

    remainder_rows = read_jsonl(args.remainder_manifest_path)
    script_rows = filter_virtualhome_script_rows(remainder_rows)
    render_rows = []
    for path in args.render_results_paths:
        render_rows.extend(read_jsonl(path))

    visual_rows, dropped_rows = build_virtualhome_visual_eval_rows(script_rows, render_rows)
    write_jsonl(args.output_jsonl, visual_rows)
    write_union_csv(args.output_csv, visual_rows)
    write_json(args.dropped_json, {"dropped_rows": dropped_rows})

    render_status_counts = Counter(str(row.get("render_status") or "unknown") for row in visual_rows)
    validation = {
        "generated_at_utc": now_iso(),
        "remainder_manifest_path": relative_to_root(args.remainder_manifest_path),
        "render_results_paths": [relative_to_root(path) for path in args.render_results_paths],
        "output_jsonl": relative_to_root(args.output_jsonl),
        "output_csv": relative_to_root(args.output_csv),
        "dropped_json": relative_to_root(args.dropped_json),
        "row_count_input_script": len(script_rows),
        "row_count_visual_eval": len(visual_rows),
        "row_count_dropped": len(dropped_rows),
        "render_status_counts": dict(sorted(render_status_counts.items())),
        "dropped_reason_counts": dict(sorted(Counter(row["reason"] for row in dropped_rows).items())),
    }
    write_json(args.validation_json, validation)

    print(f"Wrote visual manifest: {args.output_jsonl}")
    print(f"Visual rows: {validation['row_count_visual_eval']}")
    print(f"Dropped rows: {validation['row_count_dropped']}")
    print(f"Render statuses: {validation['render_status_counts']}")


if __name__ == "__main__":
    main()
