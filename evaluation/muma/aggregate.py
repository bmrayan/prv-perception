"""
Aggregate multiple disjoint MuMA-ToM frontier runs into one paper-ready bundle.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import argparse
import csv
import json
from typing import Dict, List, Sequence, Tuple

from evaluate_muma import summary_lines, summarize_results


DEFAULT_RUNS = [
    ROOT / "results" / "generated" / "muma" / "ActualPilot" / "muma_pilot15_frontier_20260426",
    ROOT / "results" / "generated" / "muma" / "Additional50" / "muma_additional50_frontier_20260426",
    ROOT / "results" / "generated" / "muma" / "Remaining160" / "muma_remaining160_frontier_20260427",
]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "generated" / "muma" / "FullFrontier"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def base_paths(run_roots: Sequence[Path]) -> List[Tuple[Path, Path, Path]]:
    resolved: List[Tuple[Path, Path, Path]] = []
    for root in run_roots:
        resolved.append((root.with_suffix(".jsonl"), root.with_name(root.name + "_config.json"), root))
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate disjoint MuMA-ToM runs.")
    parser.add_argument("--run-roots", nargs="*", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", type=str, default="muma_all225_frontier_20260427")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / f"{args.run_name}.jsonl"
    output_config = args.output_dir / f"{args.run_name}_config.json"
    output_summary_json = args.output_dir / f"{args.run_name}_summary.json"
    output_summary_txt = args.output_dir / f"{args.run_name}_summary.txt"
    output_by_condition_csv = args.output_dir / f"{args.run_name}_by_condition.csv"
    output_by_model_csv = args.output_dir / f"{args.run_name}_by_model_condition.csv"
    output_by_label_csv = args.output_dir / f"{args.run_name}_by_label_condition.csv"

    all_rows: List[dict] = []
    all_video_ids: List[str] = []
    run_payloads = []
    seen_video_ids = set()

    for jsonl_path, config_path, run_root in base_paths(args.run_roots):
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Missing run rows: {jsonl_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Missing run config: {config_path}")

        config = load_json(config_path)
        video_ids = [str(vid) for vid in config["selected_video_ids"]]
        overlap = sorted(set(video_ids) & seen_video_ids, key=int)
        if overlap:
            raise RuntimeError(f"Overlapping MuMA video ids across runs: {overlap[:10]}")
        seen_video_ids.update(video_ids)
        all_video_ids.extend(video_ids)
        rows = read_jsonl(jsonl_path)
        all_rows.extend(rows)
        run_payloads.append(
            {
                "run_root": str(run_root),
                "jsonl_path": str(jsonl_path),
                "config_path": str(config_path),
                "video_count": len(video_ids),
                "question_count": int(config["question_count"]),
                "models": config["models"],
                "conditions": config["conditions"],
            }
        )

    write_jsonl(output_jsonl, all_rows)

    merged_config = {
        "dataset": "MuMA-ToM",
        "aggregate_run": True,
        "run_name": args.run_name,
        "source_runs": run_payloads,
        "selected_video_ids": sorted(all_video_ids, key=int),
        "selected_video_count": len(all_video_ids),
        "row_count": len(all_rows),
    }
    output_config.write_text(json.dumps(merged_config, indent=2), encoding="utf-8")

    summary = summarize_results(output_jsonl)
    summary["aggregate_run"] = True
    summary["source_runs"] = run_payloads
    output_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_summary_txt.write_text("\n".join(summary_lines(summary)) + "\n", encoding="utf-8")

    write_csv(
        output_by_condition_csv,
        ["condition", "correct", "total", "accuracy"],
        [
            [condition, block["correct"], block["total"], block["accuracy"]]
            for condition, block in summary["by_condition"].items()
        ],
    )
    write_csv(
        output_by_model_csv,
        ["model", "condition", "correct", "total", "accuracy"],
        [
            [key.split("::", 1)[0], key.split("::", 1)[1], block["correct"], block["total"], block["accuracy"]]
            for key, block in summary["by_model_condition"].items()
        ],
    )
    write_csv(
        output_by_label_csv,
        ["label", "condition", "correct", "total", "accuracy"],
        [
            [key.split("::", 1)[0], key.split("::", 1)[1], block["correct"], block["total"], block["accuracy"]]
            for key, block in summary["by_label_condition"].items()
        ],
    )

    print("\n".join(summary_lines(summary)))
    print(f"\nSaved aggregate results to {output_jsonl}")


if __name__ == "__main__":
    main()
