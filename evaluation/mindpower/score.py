from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import argparse
import json
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from mindpower.common import (
    CONDITIONS,
    DEFAULT_KEY_LIST,
    DEFAULT_PREP_DIR,
    DEFAULT_RUN_DIR,
    DEFAULT_SCORE_DIR,
    MODEL_FRIENDLY_NAMES,
    TEXT_FIELDS,
    append_jsonl,
    build_bpc_prompt,
    call_openrouter,
    load_api_keys,
    now_iso,
    parse_bpc_response,
    parse_tagged_fields,
    read_jsonl,
    relative_to_root,
    resolve_runtime_model,
    write_json,
)
from mindpower.metrics import (
    bootstrap_mean_ci,
    bootstrap_paired_delta_ci,
    compute_text_scores,
    correlation_bundle,
    mcnemar_exact_pvalue,
    score_action_pair,
    semantic_column,
    sign_flip_permutation_pvalue,
)


TEXT_FIELD_OUTPUTS = [
    ("description", "perception"),
    ("belief", "belief"),
    ("desire", "desire"),
    ("intention", "intention"),
    ("decision", "decision"),
]
TEXT_OUTPUT_NAMES = [output for _, output in TEXT_FIELD_OUTPUTS]
AUTO_CONDITIONS = ["B_auto_gpt4o", "B_auto_claude"]
ALL_ANALYSIS_CONDITIONS = [
    "A_visual",
    "B_auto_gpt4o",
    "B_auto_claude",
    "B_auto_mean",
    "B_percept",
    "B_full",
    "C_text",
    "D_mismatch",
]
PCR_EPS = 1e-6


def load_manifest_index(path: Path) -> Dict[str, dict]:
    rows = read_jsonl(path)
    return {str(row["sample_id"]): row for row in rows}


def row_key(sample_id: str, model: str, condition: str) -> str:
    return f"{sample_id}::{model}::{condition}"


def build_row_table(manifest_index: Dict[str, dict], run_rows: Sequence[dict]) -> pd.DataFrame:
    rows: List[dict] = []
    for run_row in run_rows:
        sample_id = str(run_row["sample_id"])
        manifest_row = manifest_index[sample_id]
        parsed_fields = run_row.get("parsed_fields")
        if not isinstance(parsed_fields, dict):
            parsed_fields = parse_tagged_fields(str(run_row.get("response") or ""))
        record = {
            "row_key": row_key(sample_id, str(run_row["model"]), str(run_row["condition"])),
            "sample_id": sample_id,
            "task_family": str(manifest_row["task_family"]),
            "scenario_family": str(manifest_row["scenario_family"]),
            "model": str(run_row["model"]),
            "friendly_model_name": str(run_row.get("friendly_model_name") or MODEL_FRIENDLY_NAMES.get(str(run_row["model"]), str(run_row["model"]))),
            "condition": str(run_row["condition"]),
            "status": str(run_row.get("status", "unknown")),
            "format_valid": bool(run_row.get("format_valid")),
            "response": str(run_row.get("response") or ""),
            "used_scaffold_text": str(run_row.get("used_scaffold_text") or ""),
        }
        for field, _ in TEXT_FIELD_OUTPUTS:
            record[f"{field}_pred"] = str(parsed_fields.get(field) or "")
            record[f"{field}_ref"] = str(manifest_row[f"{field}_ref"])
        record["action_pred"] = str(parsed_fields.get("action") or "")
        record["action_ref"] = str(manifest_row["action_ref"])
        rows.append(record)
    frame = pd.DataFrame(rows)
    frame.sort_values(["sample_id", "model", "condition"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def add_text_metrics(
    frame: pd.DataFrame,
    *,
    sentence_model_name: str,
    bertscore_model: Optional[str],
    bertscore_rescale_with_baseline: bool,
    batch_size: int,
) -> pd.DataFrame:
    from bert_score import BERTScorer
    from sentence_transformers import SentenceTransformer

    scorer_kwargs = {
        "lang": "en",
        "batch_size": batch_size,
        "rescale_with_baseline": bertscore_rescale_with_baseline,
    }
    if bertscore_model:
        scorer_kwargs["model_type"] = bertscore_model
        scorer_kwargs.pop("lang", None)
    bert_scorer = BERTScorer(**scorer_kwargs)
    sentence_model = SentenceTransformer(sentence_model_name)

    for input_field, output_field in TEXT_FIELD_OUTPUTS:
        scores = compute_text_scores(
            frame[f"{input_field}_pred"].tolist(),
            frame[f"{input_field}_ref"].tolist(),
            field_name=output_field,
            sentence_model_name=sentence_model_name,
            bertscore_model=bertscore_model,
            bertscore_rescale_with_baseline=bertscore_rescale_with_baseline,
            bert_scorer=bert_scorer,
            sentence_model=sentence_model,
            batch_size=batch_size,
        )
        bert_col = semantic_column(output_field, "bertscore")
        sent_col = semantic_column(output_field, "sent_sim")
        frame[bert_col] = scores[bert_col]
        frame[sent_col] = scores[sent_col]
        frame[f"{output_field}_semantic_mean"] = frame[[bert_col, sent_col]].mean(axis=1)

    auto_mask = frame["condition"].isin(AUTO_CONDITIONS)
    frame["scaffold_description_bertscore"] = np.nan
    frame["scaffold_description_sent_sim"] = np.nan
    frame["scaffold_description_semantic_mean"] = np.nan
    if auto_mask.any():
        scaffold_scores = compute_text_scores(
            frame.loc[auto_mask, "used_scaffold_text"].tolist(),
            frame.loc[auto_mask, "description_ref"].tolist(),
            field_name="scaffold_description",
            sentence_model_name=sentence_model_name,
            bertscore_model=bertscore_model,
            bertscore_rescale_with_baseline=bertscore_rescale_with_baseline,
            bert_scorer=bert_scorer,
            sentence_model=sentence_model,
            batch_size=batch_size,
        )
        frame.loc[auto_mask, "scaffold_description_bertscore"] = scaffold_scores[
            "scaffold_description_bertscore"
        ]
        frame.loc[auto_mask, "scaffold_description_sent_sim"] = scaffold_scores[
            "scaffold_description_sent_sim"
        ]
        frame.loc[auto_mask, "scaffold_description_semantic_mean"] = frame.loc[
            auto_mask, ["scaffold_description_bertscore", "scaffold_description_sent_sim"]
        ].mean(axis=1)
    return frame


def add_action_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    action_metrics = [score_action_pair(pred, ref) for pred, ref in zip(frame["action_pred"], frame["action_ref"])]
    action_frame = pd.DataFrame(action_metrics)
    for column in action_frame.columns:
        frame[column] = action_frame[column].values
    frame["action_mean"] = frame[["action_sr", "action_ac"]].mean(axis=1)
    frame["overall_layer_mean"] = frame[
        [
            "perception_semantic_mean",
            "belief_semantic_mean",
            "desire_semantic_mean",
            "intention_semantic_mean",
            "decision_semantic_mean",
            "action_mean",
        ]
    ].mean(axis=1)
    frame["gap_perception_to_belief"] = frame["belief_semantic_mean"] - frame["perception_semantic_mean"]
    frame["gap_belief_to_intention"] = frame["intention_semantic_mean"] - frame["belief_semantic_mean"]
    frame["gap_intention_to_decision"] = frame["decision_semantic_mean"] - frame["intention_semantic_mean"]
    frame["gap_decision_to_action"] = frame["action_mean"] - frame["decision_semantic_mean"]
    return frame


def load_bpc_cache(path: Path) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    for row in read_jsonl(path):
        cache[str(row["row_key"])] = row
    return cache


def add_bpc_scores(
    frame: pd.DataFrame,
    *,
    cache_path: Path,
    enable_bpc: bool,
    judge_model: str,
    key_list_path: Optional[Path],
    extra_key_files: Sequence[str],
) -> pd.DataFrame:
    cache = load_bpc_cache(cache_path)
    frame["bpc_logical_consistency"] = np.nan
    frame["bpc_completeness_specificity"] = np.nan
    frame["bpc_robot_centric_perspective"] = np.nan
    frame["bpc_overall"] = np.nan
    frame["bpc_rationale"] = ""

    if enable_bpc:
        api_keys = load_api_keys(key_list_path, extra_key_files)
        session = requests.Session()
        runtime_judge_model = resolve_runtime_model(judge_model)
        key_idx = 0
        for row in frame.itertuples(index=False):
            key = str(row.row_key)
            if key in cache:
                continue
            if not bool(row.format_valid):
                payload = {
                    "timestamp_utc": now_iso(),
                    "row_key": key,
                    "judge_model": judge_model,
                    "runtime_judge_model": runtime_judge_model,
                    "logical_consistency": 0.0,
                    "completeness_specificity": 0.0,
                    "robot_centric_perspective": 0.0,
                    "overall_bpc": 0.0,
                    "rationale": "invalid_format",
                }
                cache[key] = payload
                append_jsonl(cache_path, payload)
                continue
            api_key = api_keys[key_idx % len(api_keys)]
            key_idx += 1
            response, raw_data = call_openrouter(
                runtime_judge_model,
                build_bpc_prompt(str(row.response)),
                api_key,
                image_paths=[],
                session=session,
                max_tokens=260,
            )
            parsed = parse_bpc_response(response)
            if parsed is None:
                payload = {
                    "timestamp_utc": now_iso(),
                    "row_key": key,
                    "judge_model": judge_model,
                    "runtime_judge_model": runtime_judge_model,
                    "logical_consistency": np.nan,
                    "completeness_specificity": np.nan,
                    "robot_centric_perspective": np.nan,
                    "overall_bpc": np.nan,
                    "rationale": response[:500],
                    "raw_usage": raw_data.get("usage") if raw_data else None,
                }
            else:
                payload = {
                    "timestamp_utc": now_iso(),
                    "row_key": key,
                    "judge_model": judge_model,
                    "runtime_judge_model": runtime_judge_model,
                    "logical_consistency": parsed["logical_consistency"],
                    "completeness_specificity": parsed["completeness_specificity"],
                    "robot_centric_perspective": parsed["robot_centric_perspective"],
                    "overall_bpc": parsed["overall_bpc"],
                    "rationale": parsed["rationale"],
                    "raw_usage": raw_data.get("usage") if raw_data else None,
                }
            cache[key] = payload
            append_jsonl(cache_path, payload)

    for row in frame.itertuples(index=False):
        payload = cache.get(str(row.row_key))
        if not payload:
            continue
        selector = frame["row_key"] == str(row.row_key)
        frame.loc[selector, "bpc_logical_consistency"] = payload.get("logical_consistency")
        frame.loc[selector, "bpc_completeness_specificity"] = payload.get("completeness_specificity")
        frame.loc[selector, "bpc_robot_centric_perspective"] = payload.get("robot_centric_perspective")
        frame.loc[selector, "bpc_overall"] = payload.get("overall_bpc")
        frame.loc[selector, "bpc_rationale"] = payload.get("rationale", "")
    return frame


def add_b_auto_mean_rows(frame: pd.DataFrame) -> pd.DataFrame:
    auto = frame[frame["condition"].isin(AUTO_CONDITIONS)].copy()
    if auto.empty:
        return frame
    numeric_cols = auto.select_dtypes(include=[np.number, bool]).columns.tolist()
    grouped = auto.groupby(
        ["sample_id", "task_family", "scenario_family", "model", "friendly_model_name"],
        as_index=False,
    )[numeric_cols].mean()
    grouped["condition"] = "B_auto_mean"
    grouped["status"] = "derived_mean"
    grouped["response"] = ""
    grouped["used_scaffold_text"] = ""
    grouped["description_pred"] = ""
    grouped["belief_pred"] = ""
    grouped["desire_pred"] = ""
    grouped["intention_pred"] = ""
    grouped["decision_pred"] = ""
    grouped["action_pred"] = ""
    grouped["description_ref"] = ""
    grouped["belief_ref"] = ""
    grouped["desire_ref"] = ""
    grouped["intention_ref"] = ""
    grouped["decision_ref"] = ""
    grouped["action_ref"] = ""
    grouped["row_key"] = grouped.apply(
        lambda row: row_key(str(row["sample_id"]), str(row["model"]), "B_auto_mean"), axis=1
    )
    return pd.concat([frame, grouped], ignore_index=True, sort=False)


def aggregate_model_condition_table(frame: pd.DataFrame) -> pd.DataFrame:
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
        "bpc_overall",
    ]
    grouped = frame.groupby(["friendly_model_name", "condition"], as_index=False)[numeric_cols].mean()
    if "format_valid" in frame.columns:
        grouped["format_validity_rate"] = (
            frame.groupby(["friendly_model_name", "condition"])["format_valid"].mean().values * 100.0
        )
    else:
        grouped["format_validity_rate"] = np.nan
    return grouped.sort_values(["friendly_model_name", "condition"])


def aggregate_model_condition_table_scenario_reweighted(frame: pd.DataFrame) -> pd.DataFrame:
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
        "bpc_overall",
    ]
    by_family = frame.groupby(
        ["friendly_model_name", "condition", "scenario_family"],
        as_index=False,
    )[numeric_cols].mean()
    grouped = by_family.groupby(["friendly_model_name", "condition"], as_index=False)[numeric_cols].mean()
    if "format_valid" in frame.columns:
        family_validity = (
            frame.groupby(["friendly_model_name", "condition", "scenario_family"])["format_valid"]
            .mean()
            .reset_index()
        )
        grouped["format_validity_rate"] = (
            family_validity.groupby(["friendly_model_name", "condition"])["format_valid"].mean().values * 100.0
        )
    else:
        grouped["format_validity_rate"] = np.nan
    return grouped.sort_values(["friendly_model_name", "condition"])


def build_frontier_sample_means(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = frame.select_dtypes(include=[np.number, bool]).columns.tolist()
    frontier = frame.groupby(["sample_id", "scenario_family", "condition"], as_index=False)[numeric_cols].mean()
    frontier["friendly_model_name"] = "Frontier mean"
    frontier["model"] = "frontier_mean"
    return frontier


def compute_condition_means_for_metric(frame: pd.DataFrame, metric: str) -> Dict[str, float]:
    return (
        frame.groupby("condition", as_index=False)[metric]
        .mean()
        .set_index("condition")[metric]
        .reindex(ALL_ANALYSIS_CONDITIONS)
        .dropna()
        .to_dict()
    )


def pivot_metric(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in frame.columns:
        return pd.DataFrame()
    pivot = frame.pivot_table(index="sample_id", columns="condition", values=metric, aggfunc="mean")
    if "B_auto_mean" not in pivot.columns and all(condition in pivot.columns for condition in AUTO_CONDITIONS):
        pivot["B_auto_mean"] = pivot[AUTO_CONDITIONS].mean(axis=1)
    return pivot


def build_bottleneck_rows(group_name: str, frame: pd.DataFrame) -> dict:
    pivot = pivot_metric(frame, "overall_layer_mean")
    means = {condition: float(pivot[condition].mean()) for condition in pivot.columns}
    a_value = means.get("A_visual", float("nan"))
    c_value = means.get("C_text", float("nan"))
    auto_mean = means.get("B_auto_mean", float("nan"))
    percept = means.get("B_percept", float("nan"))
    full = means.get("B_full", float("nan"))
    visual_to_text_gap = c_value - a_value if np.isfinite(a_value) and np.isfinite(c_value) else float("nan")
    rescue_auto = auto_mean - a_value if np.isfinite(auto_mean) and np.isfinite(a_value) else float("nan")
    rescue_percept = percept - a_value if np.isfinite(percept) and np.isfinite(a_value) else float("nan")
    rescue_full = full - a_value if np.isfinite(full) and np.isfinite(a_value) else float("nan")
    valid_gap = np.isfinite(visual_to_text_gap) and abs(visual_to_text_gap) >= PCR_EPS
    return {
        "group": group_name,
        "A_visual": means.get("A_visual", np.nan),
        "B_auto_gpt4o": means.get("B_auto_gpt4o", np.nan),
        "B_auto_claude": means.get("B_auto_claude", np.nan),
        "B_auto_mean": auto_mean,
        "B_percept": means.get("B_percept", np.nan),
        "B_full": means.get("B_full", np.nan),
        "C_text": means.get("C_text", np.nan),
        "D_mismatch": means.get("D_mismatch", np.nan),
        "visual_to_text_gap": visual_to_text_gap,
        "overall_gap": visual_to_text_gap,
        "rescue_auto_parser": rescue_auto,
        "rescue_percept": rescue_percept,
        "rescue_full": rescue_full,
        "PCR_auto_parser": rescue_auto / visual_to_text_gap if valid_gap else np.nan,
        "PCR_percept": rescue_percept / visual_to_text_gap if valid_gap else np.nan,
        "PCR_full": rescue_full / visual_to_text_gap if valid_gap else np.nan,
    }


def build_gap_row(group_name: str, frame: pd.DataFrame, seed: int, draws: int) -> dict:
    pivot = pivot_metric(frame, "overall_layer_mean")

    def paired_frame(left: str, right: str) -> pd.DataFrame:
        if left not in pivot.columns or right not in pivot.columns:
            return pd.DataFrame()
        return pivot.dropna(subset=[left, right])

    pair = paired_frame("A_visual", "C_text")
    overall_gap = bootstrap_paired_delta_ci(
        pair["A_visual"].tolist(),
        pair["C_text"].tolist(),
        draws=draws,
        seed=seed,
    ) if not pair.empty else {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    auto_pair = paired_frame("A_visual", "B_auto_mean")
    percept_pair = paired_frame("A_visual", "B_percept")
    full_pair = paired_frame("A_visual", "B_full")
    auto_gap = bootstrap_paired_delta_ci(
        auto_pair["A_visual"].tolist(),
        auto_pair["B_auto_mean"].tolist(),
        draws=draws,
        seed=seed,
    ) if not auto_pair.empty else {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    percept_gap = bootstrap_paired_delta_ci(
        percept_pair["A_visual"].tolist(),
        percept_pair["B_percept"].tolist(),
        draws=draws,
        seed=seed,
    ) if not percept_pair.empty else {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    full_gap = bootstrap_paired_delta_ci(
        full_pair["A_visual"].tolist(),
        full_pair["B_full"].tolist(),
        draws=draws,
        seed=seed,
    ) if not full_pair.empty else {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    row = {
        "group": group_name,
        "visual_to_text_gap": overall_gap["delta"],
        "visual_to_text_gap_ci_low": overall_gap["ci_low"],
        "visual_to_text_gap_ci_high": overall_gap["ci_high"],
        "overall_gap": overall_gap["delta"],
        "overall_gap_ci_low": overall_gap["ci_low"],
        "overall_gap_ci_high": overall_gap["ci_high"],
        "rescue_auto_parser": auto_gap["delta"],
        "rescue_auto_parser_ci_low": auto_gap["ci_low"],
        "rescue_auto_parser_ci_high": auto_gap["ci_high"],
        "rescue_percept": percept_gap["delta"],
        "rescue_percept_ci_low": percept_gap["ci_low"],
        "rescue_percept_ci_high": percept_gap["ci_high"],
        "rescue_full": full_gap["delta"],
        "rescue_full_ci_low": full_gap["ci_low"],
        "rescue_full_ci_high": full_gap["ci_high"],
    }
    if np.isfinite(overall_gap["delta"]) and abs(overall_gap["delta"]) >= PCR_EPS:
        row["PCR_auto_parser"] = auto_gap["delta"] / overall_gap["delta"]
        row["PCR_percept"] = percept_gap["delta"] / overall_gap["delta"]
        row["PCR_full"] = full_gap["delta"] / overall_gap["delta"]
    else:
        row["PCR_auto_parser"] = np.nan
        row["PCR_percept"] = np.nan
        row["PCR_full"] = np.nan

    for condition in ["A_visual", "B_percept", "B_full", "C_text"]:
        subset = frame[frame["condition"] == condition]
        for label, column in [
            ("Perception_to_Belief", "gap_perception_to_belief"),
            ("Belief_to_Intention", "gap_belief_to_intention"),
            ("Intention_to_Decision", "gap_intention_to_decision"),
            ("Decision_to_Action", "gap_decision_to_action"),
        ]:
            summary = bootstrap_mean_ci(subset[column].tolist(), draws=draws, seed=seed) if not subset.empty else {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
            prefix = f"{condition}_{label}"
            row[prefix] = summary["mean"]
            row[f"{prefix}_ci_low"] = summary["ci_low"]
            row[f"{prefix}_ci_high"] = summary["ci_high"]
    return row


def build_delta_table(group_name: str, frame: pd.DataFrame, seed: int, draws: int) -> pd.DataFrame:
    rows: List[dict] = []
    for metric in [
        "overall_layer_mean",
        "perception_semantic_mean",
        "belief_semantic_mean",
        "desire_semantic_mean",
        "intention_semantic_mean",
        "decision_semantic_mean",
        "action_mean",
    ]:
        pivot = pivot_metric(frame, metric)
        for target in ["B_auto_gpt4o", "B_auto_claude", "B_auto_mean", "B_percept", "B_full", "C_text", "D_mismatch"]:
            if "A_visual" not in pivot.columns or target not in pivot.columns:
                continue
            paired = pivot.dropna(subset=["A_visual", target])
            if paired.empty:
                continue
            ci = bootstrap_paired_delta_ci(
                paired["A_visual"].tolist(),
                paired[target].tolist(),
                draws=draws,
                seed=seed,
            )
            p_value = sign_flip_permutation_pvalue(
                (paired[target] - paired["A_visual"]).tolist(),
                draws=draws,
                seed=seed,
            )
            rows.append(
                {
                    "group": group_name,
                    "metric": metric,
                    "from_condition": "A_visual",
                    "to_condition": target,
                    "delta": ci["delta"],
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                    "permutation_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def build_condition_success_table(group_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    pivot = pivot_metric(frame, "action_exact_success")
    if pivot.empty:
        return pd.DataFrame(columns=["group", "comparison", "mcnemar_exact_p"])
    rows: List[dict] = []
    if "A_visual" in pivot.columns:
        for target in ["B_auto_mean", "B_percept", "B_full", "C_text"]:
            if target not in pivot.columns:
                continue
            paired = pivot.dropna(subset=["A_visual", target])
            rows.append(
                {
                    "group": group_name,
                    "comparison": f"A_visual_vs_{target}",
                    "mcnemar_exact_p": mcnemar_exact_pvalue(
                        paired["A_visual"].astype(bool).tolist(),
                        paired[target].astype(bool).tolist(),
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_fairness_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    baseline = frame[frame["condition"] == "A_visual"][
        ["sample_id", "friendly_model_name", "overall_layer_mean", "action_mean"]
    ].rename(
        columns={
            "overall_layer_mean": "baseline_overall_layer_mean",
            "action_mean": "baseline_action_mean",
        }
    )
    auto = frame[frame["condition"].isin(AUTO_CONDITIONS)].merge(
        baseline,
        on=["sample_id", "friendly_model_name"],
        how="left",
    )
    auto["downstream_overall_gain"] = auto["overall_layer_mean"] - auto["baseline_overall_layer_mean"]
    auto["downstream_action_gain"] = auto["action_mean"] - auto["baseline_action_mean"]

    for condition, subset in auto.groupby("condition"):
        overall_corr = correlation_bundle(
            subset["scaffold_description_semantic_mean"].tolist(),
            subset["downstream_overall_gain"].tolist(),
        )
        action_corr = correlation_bundle(
            subset["scaffold_description_semantic_mean"].tolist(),
            subset["downstream_action_gain"].tolist(),
        )
        rows.append(
            {
                "condition": condition,
                "metric": "caption_quality_vs_overall_gain",
                **overall_corr,
            }
        )
        rows.append(
            {
                "condition": condition,
                "metric": "caption_quality_vs_action_gain",
                **action_corr,
            }
        )
    return pd.DataFrame(rows)


def write_methods_note(path: Path, run_name: str, enable_bpc: bool) -> None:
    lines = [
        "# MindPower ThreeDWorld Phase-1 Methods Note",
        "",
        f"Run: `{run_name}`",
        "",
        "This phase covers the complete local `ThreeDWorld` video subset included in the selected manifest.",
        "It does not claim to cover the full MindPower benchmark and does not include the script-backed `VirtualHome` false-belief slice.",
        "",
        "Primary design choices",
        "- Evaluation panel: GPT-4o, Gemini 2.5 Pro, and Claude Sonnet 4.",
        "- Visual input: one 12-frame storyboard contact sheet per sample instead of provider-native video APIs.",
        "- Frame sampling: 8 frames across the first 80 percent of the clip and 4 frames across the final 20 percent.",
        "- Conditions: `A_visual`, `B_auto_gpt4o`, `B_auto_claude`, `B_percept`, `B_full`, `C_text`, and `D_mismatch`.",
        "- `B_percept` uses only the human `Description` field as a neutral scaffold.",
        "- `B_full` uses the full human reference chain as an image-conditioned scaffold.",
        "- `B_full` is intentionally excluded in phase 1 to avoid reasoning leakage from the gold BDI labels.",
        "",
        "Metrics",
        "- Official MindPower-style layer metrics are reported for perception, belief, desire, intention, decision, and action.",
        "- Semantic layers use BERTScore F1 and sentence-transformer cosine similarity.",
        "- Action uses weighted SR plus atomic-action correctness.",
        "- Gap and PCR diagnostics are derived from the same scored rows.",
        f"- BPC judging was {'enabled' if enable_bpc else 'not enabled in this scoring run'} for this score bundle.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a MindPower ThreeDWorld phase-1 evaluation run.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_PREP_DIR / "threedworld_phase1_manifest.jsonl",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Path to a raw_results.jsonl file from evaluate_threedworld.py",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sentence-model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--bertscore-model", type=str, default="")
    parser.add_argument("--bertscore-rescale-with-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--permutation-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-bpc", action="store_true")
    parser.add_argument("--bpc-judge-model", type=str, default="openai/gpt-4o-2024-11-20")
    parser.add_argument("--key-list-path", type=Path, default=DEFAULT_KEY_LIST)
    parser.add_argument("--extra-key-files", nargs="*", default=[])
    args = parser.parse_args()

    if args.results_path is None:
        raise SystemExit("--results-path is required.")

    run_root = args.results_path.parent
    output_dir = args.output_dir or (DEFAULT_SCORE_DIR / run_root.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_index = load_manifest_index(args.manifest_path)
    run_rows = read_jsonl(args.results_path)
    if not run_rows:
        raise SystemExit(f"No rows found in {args.results_path}")

    frame = build_row_table(manifest_index, run_rows)
    frame = add_text_metrics(
        frame,
        sentence_model_name=args.sentence_model,
        bertscore_model=args.bertscore_model or None,
        bertscore_rescale_with_baseline=bool(args.bertscore_rescale_with_baseline),
        batch_size=args.batch_size,
    )
    frame = add_action_metrics(frame)
    bpc_cache_path = output_dir / "bpc_cache.jsonl"
    frame = add_bpc_scores(
        frame,
        cache_path=bpc_cache_path,
        enable_bpc=args.enable_bpc,
        judge_model=args.bpc_judge_model,
        key_list_path=args.key_list_path,
        extra_key_files=args.extra_key_files,
    )
    frame = add_b_auto_mean_rows(frame)

    frontier_sample_means = build_frontier_sample_means(frame[frame["condition"].isin(ALL_ANALYSIS_CONDITIONS)])
    official_table = aggregate_model_condition_table(frame[frame["condition"].isin(ALL_ANALYSIS_CONDITIONS)])
    official_table_reweighted = aggregate_model_condition_table_scenario_reweighted(
        frame[frame["condition"].isin(ALL_ANALYSIS_CONDITIONS)]
    )
    frontier_condition_table = aggregate_model_condition_table(frontier_sample_means)
    frontier_condition_table_reweighted = aggregate_model_condition_table_scenario_reweighted(frontier_sample_means)

    bottleneck_rows = [
        build_bottleneck_rows(model_name, frame[frame["friendly_model_name"] == model_name])
        for model_name in sorted(frame["friendly_model_name"].dropna().unique())
        if model_name != "Frontier mean"
    ]
    bottleneck_rows.append(build_bottleneck_rows("Frontier mean", frontier_sample_means))
    bottleneck_table = pd.DataFrame(bottleneck_rows)

    gap_rows = [
        build_gap_row(model_name, frame[frame["friendly_model_name"] == model_name], args.seed, args.bootstrap_draws)
        for model_name in sorted(frame["friendly_model_name"].dropna().unique())
        if model_name != "Frontier mean"
    ]
    gap_rows.append(build_gap_row("Frontier mean", frontier_sample_means, args.seed, args.bootstrap_draws))
    gap_table = pd.DataFrame(gap_rows)

    delta_tables = [
        build_delta_table(model_name, frame[frame["friendly_model_name"] == model_name], args.seed, args.bootstrap_draws)
        for model_name in sorted(frame["friendly_model_name"].dropna().unique())
        if model_name != "Frontier mean"
    ]
    delta_tables.append(build_delta_table("Frontier mean", frontier_sample_means, args.seed, args.bootstrap_draws))
    valid_delta_tables = [table for table in delta_tables if not table.empty and table.notna().any().any()]
    condition_delta_table = (
        pd.concat(valid_delta_tables, ignore_index=True) if valid_delta_tables else pd.DataFrame()
    )

    success_tables = [
        build_condition_success_table(model_name, frame[frame["friendly_model_name"] == model_name])
        for model_name in sorted(frame["friendly_model_name"].dropna().unique())
        if model_name != "Frontier mean"
    ]
    success_tables.append(build_condition_success_table("Frontier mean", frontier_sample_means))
    valid_success_tables = [table for table in success_tables if not table.empty and table.notna().any().any()]
    action_success_table = (
        pd.concat(valid_success_tables, ignore_index=True) if valid_success_tables else pd.DataFrame()
    )

    appendix_source = frontier_sample_means.copy()
    if "format_valid" not in appendix_source.columns:
        appendix_source["format_valid"] = np.nan
    appendix_table = (
        appendix_source.groupby(["scenario_family", "condition"], as_index=False)[
            ["overall_layer_mean", "action_mean", "format_valid"]
        ]
        .mean()
        .rename(columns={"format_valid": "format_validity_rate"})
    )
    appendix_table["format_validity_rate"] = appendix_table["format_validity_rate"] * 100.0

    fairness_table = build_fairness_diagnostics(frame[frame["condition"].isin(AUTO_CONDITIONS + ["A_visual"])])

    row_scores_path = output_dir / "row_scores.csv"
    official_table_path = output_dir / "official_metrics_by_model_condition.csv"
    official_table_reweighted_path = output_dir / "official_metrics_by_model_condition_scenario_reweighted.csv"
    frontier_table_path = output_dir / "official_metrics_frontier_condition.csv"
    frontier_table_reweighted_path = output_dir / "official_metrics_frontier_condition_scenario_reweighted.csv"
    bottleneck_table_path = output_dir / "bottleneck_table.csv"
    gap_table_path = output_dir / "gap_table.csv"
    delta_table_path = output_dir / "condition_deltas_with_ci.csv"
    success_table_path = output_dir / "action_success_tests.csv"
    appendix_table_path = output_dir / "appendix_scenario_family_table.csv"
    fairness_table_path = output_dir / "fairness_diagnostics.csv"
    methods_note_path = output_dir / "methods_note.md"

    frame.to_csv(row_scores_path, index=False)
    official_table.to_csv(official_table_path, index=False)
    official_table_reweighted.to_csv(official_table_reweighted_path, index=False)
    frontier_condition_table.to_csv(frontier_table_path, index=False)
    frontier_condition_table_reweighted.to_csv(frontier_table_reweighted_path, index=False)
    bottleneck_table.to_csv(bottleneck_table_path, index=False)
    gap_table.to_csv(gap_table_path, index=False)
    condition_delta_table.to_csv(delta_table_path, index=False)
    action_success_table.to_csv(success_table_path, index=False)
    appendix_table.to_csv(appendix_table_path, index=False)
    fairness_table.to_csv(fairness_table_path, index=False)
    write_methods_note(methods_note_path, run_root.name, args.enable_bpc)

    summary = {
        "generated_at_utc": now_iso(),
        "manifest_path": relative_to_root(args.manifest_path),
        "results_path": relative_to_root(args.results_path),
        "output_dir": relative_to_root(output_dir),
        "run_root": relative_to_root(run_root),
        "row_count": int(len(frame)),
        "unique_samples": int(frame["sample_id"].nunique()),
        "unique_models": int(frame["friendly_model_name"].nunique()),
        "conditions_present": sorted(frame["condition"].dropna().unique().tolist()),
        "status_counts": frame["status"].value_counts(dropna=False).to_dict(),
        "format_validity_rate_overall": float(frame[frame["condition"].isin(CONDITIONS)]["format_valid"].mean() * 100.0),
        "bertscore_rescale_with_baseline": bool(args.bertscore_rescale_with_baseline),
        "outputs": {
            "row_scores": relative_to_root(row_scores_path),
            "official_metrics_by_model_condition": relative_to_root(official_table_path),
            "official_metrics_by_model_condition_scenario_reweighted": relative_to_root(
                official_table_reweighted_path
            ),
            "official_metrics_frontier_condition": relative_to_root(frontier_table_path),
            "official_metrics_frontier_condition_scenario_reweighted": relative_to_root(
                frontier_table_reweighted_path
            ),
            "bottleneck_table": relative_to_root(bottleneck_table_path),
            "gap_table": relative_to_root(gap_table_path),
            "condition_deltas_with_ci": relative_to_root(delta_table_path),
            "action_success_tests": relative_to_root(success_table_path),
            "appendix_scenario_family_table": relative_to_root(appendix_table_path),
            "fairness_diagnostics": relative_to_root(fairness_table_path),
            "methods_note": relative_to_root(methods_note_path),
        },
    }
    write_json(output_dir / "score_summary.json", summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
