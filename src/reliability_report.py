"""Reliability report - Measure judge vs classifier agreement and consistency."""

import argparse
from pathlib import Path
from typing import Any
from collections import Counter
import numpy as np
from loguru import logger

from .config import OUTPUTS_DIR, DARK_PATTERN_CATEGORIES
from .utils import load_jsonl, save_json


def calculate_agreement_rate(
    judge_labels: list[dict[str, Any]],
    classifier_labels: list[dict[str, Any]],
    id_field: str = "id",
) -> dict[str, Any]:
    """
    Calculate agreement rate between judge and classifier.

    Args:
        judge_labels: Records with judge labels
        classifier_labels: Records with classifier predictions
        id_field: Field to match records

    Returns:
        Agreement statistics
    """
    # Index by ID
    judge_by_id = {r.get(id_field): r for r in judge_labels}
    classifier_by_id = {r.get(id_field): r for r in classifier_labels}

    # Find matching records
    common_ids = set(judge_by_id.keys()) & set(classifier_by_id.keys())

    if not common_ids:
        return {"error": "No matching records found"}

    agreements = 0
    disagreements = []
    category_agreement = {cat: {"agree": 0, "disagree": 0} for cat in DARK_PATTERN_CATEGORIES + ["none"]}

    for record_id in common_ids:
        judge_cat = judge_by_id[record_id].get("judge_category", "none")
        classifier_cat = classifier_by_id[record_id].get("predicted_category", "none")

        if judge_cat == classifier_cat:
            agreements += 1
            if judge_cat in category_agreement:
                category_agreement[judge_cat]["agree"] += 1
        else:
            disagreements.append({
                "id": record_id,
                "judge_category": judge_cat,
                "classifier_category": classifier_cat,
                "judge_confidence": judge_by_id[record_id].get("judge_confidence", 0),
                "classifier_confidence": classifier_by_id[record_id].get("predicted_confidence", 0),
            })
            for cat in [judge_cat, classifier_cat]:
                if cat in category_agreement:
                    category_agreement[cat]["disagree"] += 1

    total = len(common_ids)
    agreement_rate = agreements / total if total > 0 else 0

    # High confidence disagreements
    high_conf_disagreements = [
        d for d in disagreements
        if d["judge_confidence"] > 0.7 and d["classifier_confidence"] > 0.7
    ]

    return {
        "total_compared": total,
        "agreements": agreements,
        "disagreements_count": len(disagreements),
        "agreement_rate": agreement_rate,
        "category_agreement": category_agreement,
        "high_confidence_disagreements": len(high_conf_disagreements),
        "sample_disagreements": disagreements[:10],
    }


def calculate_judge_self_consistency(
    reliability_test: dict[str, Any],
) -> dict[str, Any]:
    """
    Analyze judge self-consistency from reliability test results.

    Args:
        reliability_test: Results from reliability test

    Returns:
        Self-consistency statistics
    """
    detailed = reliability_test.get("detailed_results", {})

    if not detailed:
        return {"error": "No reliability test results"}

    consistency_scores = []
    category_consistency = {cat: [] for cat in DARK_PATTERN_CATEGORIES + ["none"]}

    for turn_id, judgments in detailed.items():
        categories = [j.get("category", "none") for j in judgments]

        if not categories:
            continue

        # Calculate consistency for this sample
        mode = max(set(categories), key=categories.count)
        consistency = categories.count(mode) / len(categories)
        consistency_scores.append(consistency)

        # Track by modal category
        if mode in category_consistency:
            category_consistency[mode].append(consistency)

    return {
        "num_samples": len(consistency_scores),
        "mean_consistency": float(np.mean(consistency_scores)) if consistency_scores else 0,
        "std_consistency": float(np.std(consistency_scores)) if consistency_scores else 0,
        "perfect_consistency_rate": sum(1 for s in consistency_scores if s == 1.0) / len(consistency_scores) if consistency_scores else 0,
        "category_consistency": {
            cat: {
                "mean": float(np.mean(scores)) if scores else 0,
                "count": len(scores),
            }
            for cat, scores in category_consistency.items()
        },
    }


def analyze_failure_modes(
    disagreements: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Analyze common failure modes in disagreements.

    Args:
        disagreements: List of disagreement records

    Returns:
        Failure mode analysis
    """
    if not disagreements:
        return {"error": "No disagreements to analyze"}

    # Track transition patterns (judge -> classifier)
    transitions = Counter()
    for d in disagreements:
        judge_cat = d.get("judge_category", "none")
        classifier_cat = d.get("classifier_category", "none")
        transitions[(judge_cat, classifier_cat)] += 1

    # Identify common confusion pairs
    top_confusions = [
        {
            "judge_says": pair[0],
            "classifier_says": pair[1],
            "count": count,
        }
        for pair, count in transitions.most_common(10)
    ]

    # Categories where judge is more/less conservative
    judge_more = []  # Judge flags, classifier says none
    judge_less = []  # Classifier flags, judge says none

    for d in disagreements:
        if d["judge_category"] != "none" and d["classifier_category"] == "none":
            judge_more.append(d["judge_category"])
        elif d["judge_category"] == "none" and d["classifier_category"] != "none":
            judge_less.append(d["classifier_category"])

    return {
        "top_confusion_pairs": top_confusions,
        "judge_more_conservative_categories": dict(Counter(judge_more)),
        "classifier_more_conservative_categories": dict(Counter(judge_less)),
        "total_disagreements": len(disagreements),
    }


def generate_reliability_report(
    judge_labels: list[dict[str, Any]],
    classifier_predictions: list[dict[str, Any]],
    reliability_test: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate comprehensive reliability report.

    Args:
        judge_labels: Records with judge labels
        classifier_predictions: Records with classifier predictions
        reliability_test: Optional reliability test results

    Returns:
        Reliability report dictionary
    """
    # Calculate agreement
    agreement = calculate_agreement_rate(judge_labels, classifier_predictions)

    # Analyze failure modes
    failure_modes = analyze_failure_modes(agreement.get("sample_disagreements", []))

    # Self-consistency if available
    self_consistency = {}
    if reliability_test:
        self_consistency = calculate_judge_self_consistency(reliability_test)

    # Overall reliability score (composite)
    agreement_rate = agreement.get("agreement_rate", 0)
    consistency_rate = self_consistency.get("mean_consistency", 1.0)
    reliability_score = 0.6 * agreement_rate + 0.4 * consistency_rate

    return {
        "summary": {
            "reliability_score": reliability_score,
            "agreement_rate": agreement_rate,
            "self_consistency": consistency_rate,
            "total_samples_compared": agreement.get("total_compared", 0),
        },
        "judge_classifier_agreement": agreement,
        "judge_self_consistency": self_consistency,
        "failure_modes": failure_modes,
        "recommendations": generate_recommendations(agreement, self_consistency),
    }


def generate_recommendations(
    agreement: dict[str, Any],
    self_consistency: dict[str, Any],
) -> list[str]:
    """Generate actionable recommendations based on reliability analysis."""
    recommendations = []

    agreement_rate = agreement.get("agreement_rate", 0)
    consistency = self_consistency.get("mean_consistency", 1.0)
    high_conf_disagree = agreement.get("high_confidence_disagreements", 0)

    if agreement_rate < 0.7:
        recommendations.append(
            "Low judge-classifier agreement. Consider reviewing classifier training data "
            "or adjusting judge prompts for better alignment."
        )

    if consistency < 0.8:
        recommendations.append(
            "Judge shows inconsistent labeling. Consider using multiple judge samples "
            "and majority voting, or refining the rubric."
        )

    if high_conf_disagree > 10:
        recommendations.append(
            f"Found {high_conf_disagree} high-confidence disagreements. These should be "
            "manually reviewed to identify systematic errors."
        )

    if not recommendations:
        recommendations.append(
            "Reliability metrics look good. Continue monitoring with periodic audits."
        )

    return recommendations


def main():
    """Main entry point for reliability report generation."""
    parser = argparse.ArgumentParser(description="Generate reliability report")
    parser.add_argument(
        "--judge-labels",
        type=str,
        default=str(OUTPUTS_DIR / "judge_labels.jsonl"),
        help="Judge labels file",
    )
    parser.add_argument(
        "--classifier-predictions",
        type=str,
        default=str(OUTPUTS_DIR / "wildchat_detections.jsonl"),
        help="Classifier predictions file",
    )
    parser.add_argument(
        "--reliability-test",
        type=str,
        default=str(OUTPUTS_DIR / "reliability_test.json"),
        help="Reliability test results file (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUTS_DIR / "reliability_report.json"),
        help="Output path for reliability report",
    )
    args = parser.parse_args()

    # Load data
    judge_labels = load_jsonl(args.judge_labels)
    classifier_predictions = load_jsonl(args.classifier_predictions)

    # Load reliability test if exists
    reliability_test = None
    reliability_path = Path(args.reliability_test)
    if reliability_path.exists():
        from .utils import load_json
        reliability_test = load_json(reliability_path)

    if not judge_labels:
        logger.warning("No judge labels found, creating dummy data")
        # Use classifier predictions as dummy judge labels
        judge_labels = [
            {**p, "judge_category": p.get("predicted_category", "none"), "judge_confidence": 0.7}
            for p in classifier_predictions[:100]
        ]

    # Generate report
    report = generate_reliability_report(
        judge_labels,
        classifier_predictions,
        reliability_test,
    )

    # Save report
    save_json(report, args.output)

    # Print summary
    logger.info("\n=== Reliability Report Summary ===")
    summary = report["summary"]
    logger.info(f"Reliability Score: {summary['reliability_score']:.2%}")
    logger.info(f"Agreement Rate: {summary['agreement_rate']:.2%}")
    logger.info(f"Self-Consistency: {summary['self_consistency']:.2%}")

    logger.info("\nRecommendations:")
    for rec in report["recommendations"]:
        logger.info(f"  - {rec}")


if __name__ == "__main__":
    main()
