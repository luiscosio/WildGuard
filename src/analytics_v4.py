"""V4 Analytics - Enhanced analysis with per-category escalation, topic clustering, and user behavior."""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Any
import numpy as np
from scipy import stats
from loguru import logger

from .config import OUTPUTS_DIR, DARK_PATTERN_CATEGORIES
from .utils import load_jsonl, save_json


def compute_per_category_escalation(records: list[dict]) -> dict[str, Any]:
    """
    Compute escalation rates per category across conversation turns.

    For each category, calculate:
    - Flag rate by turn bucket (1-5, 6-10, 11-20, 20+)
    - Linear regression slope and p-value
    """
    logger.info("Computing per-category escalation analysis...")

    categories = DARK_PATTERN_CATEGORIES
    turn_buckets = [(1, 5), (6, 10), (11, 20), (21, float('inf'))]
    bucket_names = ["1-5", "6-10", "11-20", "20+"]

    results = {}

    for category in categories:
        # Collect data for this category
        bucket_counts = {name: {"flagged": 0, "total": 0} for name in bucket_names}
        turn_data = []  # (turn_index, is_flagged) for regression

        for r in records:
            turn_idx = r.get("turn_index", 1)
            pred_cat = r.get("predicted_category", "none")
            is_flagged = 1 if pred_cat == category else 0

            # Assign to bucket
            for (low, high), name in zip(turn_buckets, bucket_names):
                if low <= turn_idx <= high:
                    bucket_counts[name]["total"] += 1
                    bucket_counts[name]["flagged"] += is_flagged
                    break

            turn_data.append((turn_idx, is_flagged))

        # Calculate rates per bucket
        bucket_rates = {}
        for name in bucket_names:
            total = bucket_counts[name]["total"]
            flagged = bucket_counts[name]["flagged"]
            bucket_rates[name] = {
                "rate": flagged / total if total > 0 else 0,
                "count": flagged,
                "total": total,
            }

        # Linear regression for escalation trend
        if len(turn_data) > 10:
            turns = np.array([t[0] for t in turn_data])
            flags = np.array([t[1] for t in turn_data])
            slope, intercept, r_value, p_value, std_err = stats.linregress(turns, flags)
        else:
            slope, p_value, r_value = 0, 1, 0

        # Calculate escalation percentage (turn 20+ vs turn 1-5)
        rate_early = bucket_rates["1-5"]["rate"]
        rate_late = bucket_rates["20+"]["rate"]
        if rate_early > 0:
            escalation_pct = ((rate_late - rate_early) / rate_early) * 100
        else:
            escalation_pct = 0 if rate_late == 0 else float('inf')

        results[category] = {
            "bucket_rates": bucket_rates,
            "regression": {
                "slope": float(slope),
                "p_value": float(p_value),
                "r_squared": float(r_value ** 2),
                "significant": bool(p_value < 0.05),
            },
            "escalation_pct": float(escalation_pct),
        }

        logger.info(f"  {category}: slope={slope:.6f}, p={p_value:.4f}, escalation={escalation_pct:.1f}%")

    return results


def compute_topic_clustering(records: list[dict], n_clusters: int = 10) -> dict[str, Any]:
    """
    Cluster conversations by topic using embeddings of first user message.

    Note: This requires the original WildChat data with user messages.
    For now, we'll use the model field as a proxy for analysis.
    """
    logger.info("Computing topic-based analysis (using model as proxy)...")

    # Group by model as a simple proxy for topic analysis
    model_groups = defaultdict(lambda: {"total": 0, "flagged": 0, "turns": []})

    for r in records:
        model = r.get("model", "unknown")
        is_flagged = r.get("predicted_category", "none") != "none"
        turn_idx = r.get("turn_index", 1)

        model_groups[model]["total"] += 1
        if is_flagged:
            model_groups[model]["flagged"] += 1
        model_groups[model]["turns"].append((turn_idx, is_flagged))

    results = {}
    for model, data in model_groups.items():
        rate = data["flagged"] / data["total"] if data["total"] > 0 else 0

        # Calculate escalation slope for this model
        if len(data["turns"]) > 10:
            turns = np.array([t[0] for t in data["turns"]])
            flags = np.array([t[1] for t in data["turns"]])
            slope, _, _, p_value, _ = stats.linregress(turns, flags)
        else:
            slope, p_value = 0, 1

        results[model] = {
            "total_turns": data["total"],
            "flagged_turns": data["flagged"],
            "flag_rate": float(rate),
            "escalation_slope": float(slope),
            "escalation_p_value": float(p_value),
        }

    return {"by_model": results}


def compute_user_behavior_analysis(records: list[dict]) -> dict[str, Any]:
    """
    Analyze relationship between dark patterns and conversation length.

    Questions:
    1. Does dark_pattern_rate correlate with conversation_length?
    2. Do conversations with early flags (turn 1-3) last longer?
    """
    logger.info("Computing user behavior analysis...")

    # Group by conversation
    conversations = defaultdict(lambda: {
        "turns": [],
        "flags": [],
        "total_turns": 0,
        "first_flag_turn": None,
    })

    for r in records:
        conv_id = r.get("conversation_id", "unknown")
        turn_idx = r.get("turn_index", 1)
        is_flagged = r.get("predicted_category", "none") != "none"
        total_turns = r.get("total_turns", 1)

        conversations[conv_id]["turns"].append(turn_idx)
        conversations[conv_id]["flags"].append(is_flagged)
        conversations[conv_id]["total_turns"] = max(conversations[conv_id]["total_turns"], total_turns)

        if is_flagged and conversations[conv_id]["first_flag_turn"] is None:
            conversations[conv_id]["first_flag_turn"] = turn_idx

    # Compute metrics per conversation
    conv_lengths = []
    dark_pattern_counts = []
    dark_pattern_rates = []
    early_flag_lengths = []  # conversations with flag in turns 1-3
    no_early_flag_lengths = []  # conversations without early flag

    for conv_id, data in conversations.items():
        length = data["total_turns"]
        flag_count = sum(data["flags"])
        flag_rate = flag_count / len(data["flags"]) if data["flags"] else 0

        conv_lengths.append(length)
        dark_pattern_counts.append(flag_count)
        dark_pattern_rates.append(flag_rate)

        first_flag = data["first_flag_turn"]
        if first_flag is not None and first_flag <= 3:
            early_flag_lengths.append(length)
        else:
            no_early_flag_lengths.append(length)

    # Correlation analysis
    if len(conv_lengths) > 10:
        # Length vs count correlation
        r_count, p_count = stats.pearsonr(conv_lengths, dark_pattern_counts)
        # Length vs rate correlation
        r_rate, p_rate = stats.pearsonr(conv_lengths, dark_pattern_rates)
    else:
        r_count, p_count, r_rate, p_rate = 0, 1, 0, 1

    # T-test for early vs no early flag
    if len(early_flag_lengths) > 5 and len(no_early_flag_lengths) > 5:
        t_stat, t_p_value = stats.ttest_ind(early_flag_lengths, no_early_flag_lengths)
    else:
        t_stat, t_p_value = 0, 1

    results = {
        "correlation_length_vs_count": {
            "pearson_r": float(r_count),
            "p_value": float(p_count),
            "significant": bool(p_count < 0.05),
        },
        "correlation_length_vs_rate": {
            "pearson_r": float(r_rate),
            "p_value": float(p_rate),
            "significant": bool(p_rate < 0.05),
        },
        "early_flag_analysis": {
            "conversations_with_early_flag": len(early_flag_lengths),
            "conversations_without_early_flag": len(no_early_flag_lengths),
            "mean_length_with_early_flag": float(np.mean(early_flag_lengths)) if early_flag_lengths else 0,
            "mean_length_without_early_flag": float(np.mean(no_early_flag_lengths)) if no_early_flag_lengths else 0,
            "t_statistic": float(t_stat),
            "p_value": float(t_p_value),
            "significant": bool(t_p_value < 0.05),
        },
        "summary": {
            "total_conversations": len(conversations),
            "mean_conversation_length": float(np.mean(conv_lengths)),
            "mean_dark_pattern_count": float(np.mean(dark_pattern_counts)),
            "mean_dark_pattern_rate": float(np.mean(dark_pattern_rates)),
        },
    }

    logger.info(f"  Correlation (length vs count): r={r_count:.4f}, p={p_count:.4f}")
    logger.info(f"  Early flag mean length: {results['early_flag_analysis']['mean_length_with_early_flag']:.2f}")
    logger.info(f"  No early flag mean length: {results['early_flag_analysis']['mean_length_without_early_flag']:.2f}")

    return results


def run_v4_analytics(input_file: Path, output_file: Path) -> dict[str, Any]:
    """Run full V4 analytics suite."""
    logger.info(f"Loading detections from {input_file}...")
    records = load_jsonl(input_file)
    logger.info(f"Loaded {len(records)} records")

    results = {
        "version": "v4",
        "total_records": len(records),
    }

    # Per-category escalation
    results["per_category_escalation"] = compute_per_category_escalation(records)

    # Topic clustering (model-based proxy for now)
    results["topic_analysis"] = compute_topic_clustering(records)

    # User behavior analysis
    results["user_behavior"] = compute_user_behavior_analysis(records)

    # Save results
    save_json(results, output_file)
    logger.info(f"Results saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run V4 enhanced analytics")
    parser.add_argument(
        "--input",
        type=str,
        default=str(OUTPUTS_DIR / "wildchat_detections_v3.jsonl"),
        help="Input detections file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUTS_DIR / "analytics_v4.json"),
        help="Output analytics file",
    )
    args = parser.parse_args()

    run_v4_analytics(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
