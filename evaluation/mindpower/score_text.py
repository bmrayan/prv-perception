from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import argparse
from typing import Dict

import pandas as pd
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from mindpower.benchmark_remainder import DEFAULT_REMAINDER_PREP_DIR, DEFAULT_REMAINDER_SCORE_DIR
from mindpower.common import now_iso, read_jsonl, relative_to_root, write_json
from mindpower.score import (
    add_action_metrics,
    add_text_metrics,
    aggregate_model_condition_table,
    build_frontier_sample_means,
    build_row_table,
)


def load_manifest_index(path: Path) -> Dict[str, dict]:
    rows = read_jsonl(path)
    return {str(row["sample_id"]): row for row in rows}


def aggregate_source_group_table(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "perception_bertscore",
        "perception_sent_sim",
        "belief_bertscore",
        "belief_sent_sim",
        "desire_bertscore",
        "desire_sent_sim",
        "intention_bertscore",
        "intention_sent_sim",
        "decision_bertscore",
        "decision_sent_sim",
        "action_sr",
        "action_ac",
        "action_mean",
        "overall_layer_mean",
    ]
    grouped = frame.groupby(["source_group", "condition"], as_index=False)[numeric_cols].mean()
    grouped["format_validity_rate"] = (
        frame.groupby(["source_group", "condition"])["format_valid"].mean().values * 100.0
    )
    return grouped.sort_values(["source_group", "condition"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the text-backed remainder of the MindPower benchmark.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_REMAINDER_PREP_DIR / "remainder_text_eval_manifest.jsonl",
    )
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REMAINDER_SCORE_DIR / "remainder_text_eval")
    parser.add_argument("--sentence-model-name", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--bertscore-model", type=str, default=None)
    parser.add_argument("--bertscore-rescale-with-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_index = load_manifest_index(args.manifest_path)
    run_rows = read_jsonl(args.results_path)
    if not run_rows:
        raise SystemExit(f"No run rows found in {args.results_path}")
    frame = build_row_table(manifest_index, run_rows)
    frame["source_group"] = frame["sample_id"].map(lambda sid: str(manifest_index[sid].get("source_group", "unknown")))
    frame = add_text_metrics(
        frame,
        sentence_model_name=args.sentence_model_name,
        bertscore_model=args.bertscore_model,
        bertscore_rescale_with_baseline=bool(args.bertscore_rescale_with_baseline),
        batch_size=args.batch_size,
    )
    frame = add_action_metrics(frame)
    if "bpc_overall" not in frame.columns:
        frame["bpc_overall"] = np.nan

    row_scores_path = args.output_dir / "row_scores.csv"
    by_model_condition_path = args.output_dir / "official_metrics_by_model_condition.csv"
    frontier_condition_path = args.output_dir / "official_metrics_frontier_condition.csv"
    source_group_path = args.output_dir / "source_group_condition_table.csv"
    summary_path = args.output_dir / "score_summary.json"

    frame.to_csv(row_scores_path, index=False)
    by_model = aggregate_model_condition_table(frame)
    by_model.to_csv(by_model_condition_path, index=False)
    frontier = aggregate_model_condition_table(build_frontier_sample_means(frame))
    frontier.to_csv(frontier_condition_path, index=False)
    source_group = aggregate_source_group_table(frame)
    source_group.to_csv(source_group_path, index=False)

    summary = {
        "generated_at_utc": now_iso(),
        "manifest_path": relative_to_root(args.manifest_path),
        "results_path": relative_to_root(args.results_path),
        "output_dir": relative_to_root(args.output_dir),
        "row_count": int(len(frame)),
        "unique_samples": int(frame["sample_id"].nunique()),
        "unique_models": int(frame["friendly_model_name"].nunique()),
        "conditions_present": sorted(frame["condition"].dropna().unique().tolist()),
        "status_counts": frame["status"].value_counts().sort_index().to_dict(),
        "format_validity_rate_overall": float(frame["format_valid"].mean() * 100.0),
        "bertscore_rescale_with_baseline": bool(args.bertscore_rescale_with_baseline),
        "outputs": {
            "row_scores": relative_to_root(row_scores_path),
            "official_metrics_by_model_condition": relative_to_root(by_model_condition_path),
            "official_metrics_frontier_condition": relative_to_root(frontier_condition_path),
            "source_group_condition_table": relative_to_root(source_group_path),
        },
    }
    write_json(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
