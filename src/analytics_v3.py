"""V3 Analytics - Enhanced analysis with statistical tests and calibration.

Key additions over V2:
1. Statistical tests for GPT-4 vs GPT-3.5 comparison
2. Confidence calibration analysis
3. Enhanced turn-index escalation analysis
4. Category breakdown by model
"""

import argparse
from pathlib import Path
from typing import Any
from collections import Counter
import numpy as np
from loguru import logger

from .config import OUTPUTS_DIR, FIGURES_DIR, DARK_PATTERN_CATEGORIES
from .utils import load_jsonl, save_json


def calculate_prevalence_v3(
    detections: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate comprehensive prevalence statistics."""
    total = len(detections)
    if total == 0:
        return {"error": "No detections to analyze"}

    # Count by category
    category_counts = Counter(d.get("predicted_category", "none") for d in detections)

    # Flagged statistics
    flagged = [d for d in detections if d.get("predicted_category", "none") != "none"]
    flagged_count = len(flagged)
    flagged_rate = flagged_count / total

    category_breakdown = {}
    for cat in DARK_PATTERN_CATEGORIES + ["none"]:
        count = category_counts.get(cat, 0)
        category_breakdown[cat] = {
            "count": count,
            "rate_percent": 100 * count / total,
            "per_1000": 1000 * count / total,
        }

    return {
        "total_turns": total,
        "flagged_count": flagged_count,
        "flagged_rate": flagged_rate * 100,
        "category_breakdown": category_breakdown,
    }


def calculate_model_comparison(
    detections: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate and compare statistics by model."""
    from scipy import stats

    by_model: dict[str, list] = {}
    for d in detections:
        model = d.get("model", "unknown")
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(d)

    model_stats = {}
    for model, items in by_model.items():
        flagged = [i for i in items if i.get("predicted_category", "none") != "none"]
        category_counts = Counter(i.get("predicted_category", "none") for i in items)

        model_stats[model] = {
            "total": len(items),
            "flagged": len(flagged),
            "flagged_rate": len(flagged) / len(items) if items else 0,
            "category_breakdown": {
                cat: {
                    "count": category_counts.get(cat, 0),
                    "rate": category_counts.get(cat, 0) / len(items) if items else 0,
                }
                for cat in DARK_PATTERN_CATEGORIES
            }
        }

    # Statistical comparison between GPT-4 and GPT-3.5
    gpt4_models = [m for m in by_model.keys() if "gpt-4" in m.lower()]
    gpt35_models = [m for m in by_model.keys() if "gpt-3.5" in m.lower()]

    statistical_tests = {}
    if gpt4_models and gpt35_models:
        # Aggregate GPT-4 and GPT-3.5 data
        gpt4_items = []
        gpt35_items = []
        for m in gpt4_models:
            gpt4_items.extend(by_model[m])
        for m in gpt35_models:
            gpt35_items.extend(by_model[m])

        gpt4_flagged = sum(1 for i in gpt4_items if i.get("predicted_category", "none") != "none")
        gpt35_flagged = sum(1 for i in gpt35_items if i.get("predicted_category", "none") != "none")
        gpt4_total = len(gpt4_items)
        gpt35_total = len(gpt35_items)

        # Chi-squared test for overall flag rate
        contingency = np.array([
            [gpt4_flagged, gpt4_total - gpt4_flagged],
            [gpt35_flagged, gpt35_total - gpt35_flagged]
        ])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        # Effect size (Phi coefficient)
        n = gpt4_total + gpt35_total
        phi = np.sqrt(chi2 / n) if n > 0 else 0

        # Odds ratio
        a, b = contingency[0]
        c, d = contingency[1]
        odds_ratio = (a * d) / (b * c) if b * c > 0 else float('inf')

        statistical_tests["overall"] = {
            "gpt4_flagged": gpt4_flagged,
            "gpt4_total": gpt4_total,
            "gpt4_rate": gpt4_flagged / gpt4_total if gpt4_total else 0,
            "gpt35_flagged": gpt35_flagged,
            "gpt35_total": gpt35_total,
            "gpt35_rate": gpt35_flagged / gpt35_total if gpt35_total else 0,
            "chi2": float(chi2),
            "p_value": float(p_value),
            "phi_coefficient": float(phi),
            "odds_ratio": float(odds_ratio),
            "significant": bool(p_value < 0.05),
            "interpretation": _interpret_effect_size(phi, p_value),
        }

        # Per-category comparison
        statistical_tests["by_category"] = {}
        for cat in DARK_PATTERN_CATEGORIES:
            gpt4_cat = sum(1 for i in gpt4_items if i.get("predicted_category") == cat)
            gpt35_cat = sum(1 for i in gpt35_items if i.get("predicted_category") == cat)

            cat_contingency = np.array([
                [gpt4_cat, gpt4_total - gpt4_cat],
                [gpt35_cat, gpt35_total - gpt35_cat]
            ])

            try:
                chi2_cat, p_cat, _, _ = stats.chi2_contingency(cat_contingency)
                phi_cat = np.sqrt(chi2_cat / n) if n > 0 else 0
            except ValueError:
                chi2_cat, p_cat, phi_cat = 0, 1, 0

            statistical_tests["by_category"][cat] = {
                "gpt4_count": gpt4_cat,
                "gpt4_rate": gpt4_cat / gpt4_total if gpt4_total else 0,
                "gpt35_count": gpt35_cat,
                "gpt35_rate": gpt35_cat / gpt35_total if gpt35_total else 0,
                "chi2": float(chi2_cat),
                "p_value": float(p_cat),
                "phi": float(phi_cat),
                "significant": bool(p_cat < 0.05),
                "direction": "GPT-4 higher" if gpt4_cat/gpt4_total > gpt35_cat/gpt35_total else "GPT-3.5 higher",
            }

    return {
        "by_model": model_stats,
        "statistical_tests": statistical_tests,
    }


def _interpret_effect_size(phi: float, p_value: float) -> str:
    """Interpret effect size using Cohen's conventions."""
    if p_value >= 0.05:
        return "Not statistically significant"
    elif phi < 0.1:
        return "Negligible effect size"
    elif phi < 0.3:
        return "Small effect size"
    elif phi < 0.5:
        return "Medium effect size"
    else:
        return "Large effect size"


def calculate_confidence_calibration(
    detections: list[dict[str, Any]],
    validation_file: Path | None = None,
    num_bins: int = 10,
) -> dict[str, Any]:
    """
    Calculate confidence calibration statistics.

    If validation file provided, use judge labels as ground truth.
    Otherwise, report confidence distribution by category.
    """
    confidences = [d.get("predicted_confidence", 0) for d in detections]
    predictions = [d.get("predicted_category", "none") for d in detections]

    # Basic confidence distribution
    conf_bins = np.linspace(0, 1, num_bins + 1)
    bin_counts = []
    bin_means = []

    for i in range(num_bins):
        mask = (np.array(confidences) >= conf_bins[i]) & (np.array(confidences) < conf_bins[i + 1])
        count = int(np.sum(mask))
        bin_counts.append(count)
        if count > 0:
            bin_means.append(float(np.mean(np.array(confidences)[mask])))
        else:
            bin_means.append(0.0)

    calibration = {
        "confidence_bins": conf_bins.tolist(),
        "bin_counts": bin_counts,
        "bin_mean_confidence": bin_means,
        "overall_mean_confidence": float(np.mean(confidences)),
        "overall_std_confidence": float(np.std(confidences)),
    }

    # If validation available, calculate actual calibration
    if validation_file and validation_file.exists():
        validation = load_jsonl(validation_file)
        val_by_id = {r.get("id"): r for r in validation}

        # Match predictions with validation
        matched_correct = []
        matched_conf = []

        for d in detections:
            d_id = d.get("id")
            if d_id in val_by_id:
                pred_cat = d.get("predicted_category", "none")
                judge_cat = val_by_id[d_id].get("judge_category", "none")
                conf = d.get("predicted_confidence", 0)

                matched_conf.append(conf)
                matched_correct.append(1 if pred_cat == judge_cat else 0)

        if matched_conf:
            # Calculate ECE (Expected Calibration Error)
            bin_accuracies = []
            bin_confidences = []
            bin_weights = []

            matched_conf = np.array(matched_conf)
            matched_correct = np.array(matched_correct)

            for i in range(num_bins):
                mask = (matched_conf >= conf_bins[i]) & (matched_conf < conf_bins[i + 1])
                if np.sum(mask) > 0:
                    bin_acc = np.mean(matched_correct[mask])
                    bin_conf = np.mean(matched_conf[mask])
                    bin_accuracies.append(float(bin_acc))
                    bin_confidences.append(float(bin_conf))
                    bin_weights.append(int(np.sum(mask)))
                else:
                    bin_accuracies.append(0.0)
                    bin_confidences.append(0.0)
                    bin_weights.append(0)

            # ECE calculation
            total_samples = sum(bin_weights)
            ece = sum(
                (w / total_samples) * abs(acc - conf)
                for w, acc, conf in zip(bin_weights, bin_accuracies, bin_confidences)
                if total_samples > 0
            )

            calibration["validation_calibration"] = {
                "matched_samples": len(matched_conf),
                "bin_accuracies": bin_accuracies,
                "bin_confidences": bin_confidences,
                "bin_weights": bin_weights,
                "expected_calibration_error": float(ece),
                "overall_accuracy": float(np.mean(matched_correct)),
            }

    # High confidence analysis
    high_conf_mask = np.array(confidences) >= 0.8
    high_conf_preds = [p for p, m in zip(predictions, high_conf_mask) if m]
    high_conf_cat_dist = Counter(high_conf_preds)

    calibration["high_confidence_analysis"] = {
        "threshold": 0.8,
        "count": int(np.sum(high_conf_mask)),
        "percentage": float(100 * np.sum(high_conf_mask) / len(confidences)) if confidences else 0,
        "category_distribution": dict(high_conf_cat_dist),
    }

    return calibration


def calculate_turn_escalation(
    detections: list[dict[str, Any]],
    max_turn: int = 30,
) -> dict[str, Any]:
    """Calculate escalation patterns across conversation turns."""
    from scipy import stats

    # Group by turn index
    by_turn: dict[int, list] = {}
    for d in detections:
        turn = min(d.get("turn_index", 0), max_turn)
        if turn not in by_turn:
            by_turn[turn] = []
        by_turn[turn].append(d)

    # Calculate flag rate by turn
    turn_stats = {}
    turns = sorted(by_turn.keys())
    flag_rates = []

    for turn in turns:
        items = by_turn[turn]
        flagged = sum(1 for i in items if i.get("predicted_category", "none") != "none")
        rate = flagged / len(items) if items else 0
        flag_rates.append(rate)

        # Category breakdown
        cat_counts = Counter(i.get("predicted_category", "none") for i in items)
        turn_stats[turn] = {
            "total": len(items),
            "flagged": flagged,
            "flag_rate": rate,
            "category_counts": {cat: cat_counts.get(cat, 0) for cat in DARK_PATTERN_CATEGORIES},
        }

    # Trend analysis
    if len(turns) >= 5:
        # Linear regression for trend
        x = np.array(turns)
        y = np.array(flag_rates)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Split into early (1-5) vs late (10+) turns
        early_mask = np.array(turns) <= 5
        late_mask = np.array(turns) >= 10

        early_rates = np.array(flag_rates)[early_mask]
        late_rates = np.array(flag_rates)[late_mask]

        if len(early_rates) > 0 and len(late_rates) > 0:
            # Mann-Whitney U test for early vs late
            try:
                stat, p_early_late = stats.mannwhitneyu(early_rates, late_rates, alternative='two-sided')
            except ValueError:
                p_early_late = 1.0

            trend_analysis = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "significant_trend": bool(p_value < 0.05),
                "trend_direction": "increasing" if slope > 0 else "decreasing",
                "early_mean_rate": float(np.mean(early_rates)),
                "late_mean_rate": float(np.mean(late_rates)),
                "early_vs_late_p_value": float(p_early_late),
                "early_vs_late_significant": bool(p_early_late < 0.05),
            }
        else:
            trend_analysis = {"error": "Insufficient data for early/late comparison"}
    else:
        trend_analysis = {"error": "Insufficient turns for trend analysis"}

    # Peak detection
    peak_turn = max(turns, key=lambda t: turn_stats[t]["flag_rate"]) if turns else 0

    return {
        "turn_stats": {str(k): v for k, v in turn_stats.items()},
        "trend_analysis": trend_analysis,
        "peak_turn": peak_turn,
        "peak_flag_rate": turn_stats.get(peak_turn, {}).get("flag_rate", 0),
        "max_turn_analyzed": max_turn,
    }


def generate_v3_report(
    detections: list[dict[str, Any]],
    validation_file: Path | None = None,
) -> dict[str, Any]:
    """Generate comprehensive V3 analytics report."""
    logger.info(f"Analyzing {len(detections)} detections...")

    prevalence = calculate_prevalence_v3(detections)
    model_comparison = calculate_model_comparison(detections)
    calibration = calculate_confidence_calibration(detections, validation_file)
    turn_escalation = calculate_turn_escalation(detections)

    # Summarize key findings
    key_findings = []

    # GPT-4 vs GPT-3.5 finding
    stats_tests = model_comparison.get("statistical_tests", {}).get("overall", {})
    if stats_tests.get("significant"):
        gpt4_rate = stats_tests.get("gpt4_rate", 0) * 100
        gpt35_rate = stats_tests.get("gpt35_rate", 0) * 100
        diff = gpt4_rate - gpt35_rate
        key_findings.append({
            "finding": f"GPT-4 shows {abs(diff):.1f}% {'more' if diff > 0 else 'fewer'} dark patterns than GPT-3.5",
            "gpt4_rate": f"{gpt4_rate:.1f}%",
            "gpt35_rate": f"{gpt35_rate:.1f}%",
            "p_value": stats_tests.get("p_value"),
            "effect_size": stats_tests.get("interpretation"),
        })

    # Turn escalation finding
    trend = turn_escalation.get("trend_analysis", {})
    if trend.get("significant_trend"):
        direction = trend.get("trend_direction", "unknown")
        key_findings.append({
            "finding": f"Dark patterns show {direction} trend across conversation turns",
            "early_rate": f"{trend.get('early_mean_rate', 0) * 100:.1f}%",
            "late_rate": f"{trend.get('late_mean_rate', 0) * 100:.1f}%",
            "p_value": trend.get("p_value"),
        })

    # Calibration finding
    cal_val = calibration.get("validation_calibration", {})
    if cal_val:
        ece = cal_val.get("expected_calibration_error", 0)
        acc = cal_val.get("overall_accuracy", 0)
        key_findings.append({
            "finding": f"Classifier shows ECE of {ece:.3f} with {acc*100:.1f}% accuracy on validated samples",
        })

    return {
        "prevalence": prevalence,
        "model_comparison": model_comparison,
        "confidence_calibration": calibration,
        "turn_escalation": turn_escalation,
        "key_findings": key_findings,
        "version": "v3",
    }


def main():
    """Main entry point for V3 analytics."""
    parser = argparse.ArgumentParser(description="Generate V3 analytics report")
    parser.add_argument(
        "--input",
        type=str,
        default=str(OUTPUTS_DIR / "wildchat_detections_v3.jsonl"),
        help="Input detections file",
    )
    parser.add_argument(
        "--validation",
        type=str,
        default=str(OUTPUTS_DIR / "v2" / "validation_labels.jsonl"),
        help="Validation labels file for calibration",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUTS_DIR / "analytics_v3.json"),
        help="Output path for analytics report",
    )
    args = parser.parse_args()

    # Load detections
    detections = load_jsonl(args.input)
    if not detections:
        logger.error("No detections found!")
        return

    # Check validation file
    validation_path = Path(args.validation)
    if not validation_path.exists():
        logger.warning(f"Validation file not found: {validation_path}")
        validation_path = None

    # Generate report
    report = generate_v3_report(detections, validation_path)

    # Save report
    save_json(report, args.output)
    logger.info(f"Analytics report saved to {args.output}")

    # Print summary
    logger.info("\n=== V3 Analytics Summary ===")
    prev = report["prevalence"]
    logger.info(f"Total turns: {prev['total_turns']}")
    logger.info(f"Flagged: {prev['flagged_count']} ({prev['flagged_rate']:.1f}%)")

    logger.info("\n=== Key Findings ===")
    for finding in report.get("key_findings", []):
        logger.info(f"  • {finding['finding']}")


if __name__ == "__main__":
    main()
