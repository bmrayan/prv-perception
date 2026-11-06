from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


from collections import Counter
from typing import Any, Dict, Optional, Sequence

import numpy as np
from rouge_score import rouge_scorer
from scipy.stats import binomtest, pearsonr, spearmanr

from mindpower.common import canonical_action_text, parse_action_sequence


ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)


def semantic_column(field: str, metric: str) -> str:
    return f"{field}_{metric}"


def score_action_pair(pred_text: str, ref_text: str) -> Dict[str, float | bool | str]:
    pred_canonical = canonical_action_text(pred_text)
    ref_canonical = canonical_action_text(ref_text)
    if not pred_canonical or not ref_canonical:
        return {
            "pred_action_canonical": pred_canonical,
            "ref_action_canonical": ref_canonical,
            "action_sr": 0.0,
            "action_ac": 0.0,
            "action_exact_success": False,
        }

    rouge_scores = ROUGE_SCORER.score(ref_canonical, pred_canonical)
    rouge1 = 100.0 * rouge_scores["rouge1"].fmeasure
    rouge2 = 100.0 * rouge_scores["rouge2"].fmeasure
    rouge_l = 100.0 * rouge_scores["rougeL"].fmeasure
    sr = (2.0 * rouge1 + 3.0 * rouge2 + 5.0 * rouge_l) / 10.0

    pred_actions = Counter(parse_action_sequence(pred_text))
    ref_actions = Counter(parse_action_sequence(ref_text))
    matched = sum(min(pred_actions[action], ref_actions[action]) for action in ref_actions)
    total_ref = sum(ref_actions.values())
    ac = 100.0 * matched / total_ref if total_ref else 0.0
    exact = list(pred_actions.elements()) == list(ref_actions.elements())
    if not exact:
        exact = parse_action_sequence(pred_text) == parse_action_sequence(ref_text)

    return {
        "pred_action_canonical": pred_canonical,
        "ref_action_canonical": ref_canonical,
        "action_sr": float(sr),
        "action_ac": float(ac),
        "action_exact_success": bool(exact),
    }


def compute_text_scores(
    candidates: Sequence[str],
    references: Sequence[str],
    *,
    field_name: str,
    sentence_model_name: str,
    bertscore_model: Optional[str],
    bertscore_rescale_with_baseline: bool = True,
    bert_scorer: Optional[Any] = None,
    sentence_model: Optional[Any] = None,
    batch_size: int = 16,
) -> Dict[str, np.ndarray]:
    from bert_score import BERTScorer
    from sentence_transformers import SentenceTransformer

    size = len(candidates)
    bert = np.zeros(size, dtype=float)
    sent = np.zeros(size, dtype=float)
    valid_indices = [
        idx for idx, (cand, ref) in enumerate(zip(candidates, references)) if cand.strip() and ref.strip()
    ]
    if not valid_indices:
        return {
            semantic_column(field_name, "bertscore"): bert,
            semantic_column(field_name, "sent_sim"): sent,
        }

    valid_candidates = [candidates[idx] for idx in valid_indices]
    valid_references = [references[idx] for idx in valid_indices]
    scorer = bert_scorer
    if scorer is None:
        scorer_kwargs = {
            "lang": "en",
            "batch_size": batch_size,
            "rescale_with_baseline": bertscore_rescale_with_baseline,
        }
        if bertscore_model:
            scorer_kwargs["model_type"] = bertscore_model
            scorer_kwargs.pop("lang", None)
        scorer = BERTScorer(**scorer_kwargs)
    _, _, f1 = scorer.score(valid_candidates, valid_references, batch_size=batch_size)
    bert_values = (f1.detach().cpu().numpy() * 100.0).astype(float)

    sentence_encoder = sentence_model or SentenceTransformer(sentence_model_name)
    candidate_embeddings = sentence_encoder.encode(
        valid_candidates,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    reference_embeddings = sentence_encoder.encode(
        valid_references,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    sent_values = (candidate_embeddings * reference_embeddings).sum(axis=1) * 100.0

    for local_idx, global_idx in enumerate(valid_indices):
        bert[global_idx] = float(bert_values[local_idx])
        sent[global_idx] = float(sent_values[local_idx])
    return {
        semantic_column(field_name, "bertscore"): bert,
        semantic_column(field_name, "sent_sim"): sent,
    }


def mean_or_nan(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan")
    return float(np.nanmean(array))


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    draws: int = 10_000,
    seed: int = 42,
) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(draws, array.size))
    boot = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }


def bootstrap_paired_delta_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    draws: int = 10_000,
    seed: int = 42,
) -> Dict[str, float]:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.size != right_array.size:
        raise ValueError("left and right must have the same length")
    if left_array.size == 0:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    diffs = right_array - left_array
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, diffs.size, size=(draws, diffs.size))
    boot = diffs[indices].mean(axis=1)
    return {
        "delta": float(diffs.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }


def sign_flip_permutation_pvalue(
    diffs: Sequence[float],
    *,
    draws: int = 10_000,
    seed: int = 42,
) -> Optional[float]:
    array = np.asarray(diffs, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0 or np.allclose(array, 0.0):
        return None
    observed = abs(float(array.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, array.size))
    permuted = np.abs((signs * array).mean(axis=1))
    p_value = (np.count_nonzero(permuted >= observed) + 1.0) / (draws + 1.0)
    return float(p_value)


def mcnemar_exact_pvalue(left: Sequence[bool], right: Sequence[bool]) -> Optional[float]:
    left_array = np.asarray(left, dtype=bool)
    right_array = np.asarray(right, dtype=bool)
    if left_array.size != right_array.size or left_array.size == 0:
        return None
    b = int(np.sum(np.logical_and(left_array, np.logical_not(right_array))))
    c = int(np.sum(np.logical_and(np.logical_not(left_array), right_array)))
    n = b + c
    if n == 0:
        return None
    result = binomtest(min(b, c), n=n, p=0.5, alternative="two-sided")
    return float(result.pvalue)


def correlation_bundle(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, Optional[float]]:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {
            "n": int(x.size),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
        }
    pearson_stats = pearsonr(x, y)
    spearman_stats = spearmanr(x, y)
    return {
        "n": int(x.size),
        "pearson_r": float(pearson_stats.statistic),
        "pearson_p": float(pearson_stats.pvalue),
        "spearman_rho": float(spearman_stats.statistic),
        "spearman_p": float(spearman_stats.pvalue),
    }
