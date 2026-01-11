"""Gap report - Compare benchmark vs reality distributions."""

import argparse
from pathlib import Path
from typing import Any
from collections import Counter
import numpy as np
from scipy import stats
from loguru import logger

from .config import OUTPUTS_DIR, DARK_PATTERN_CATEGORIES
from .utils import load_jsonl, save_json


def calculate_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate Jensen-Shannon divergence between two distributions.

    Args:
        p: First probability distribution
        q: Second probability distribution

    Returns:
        JS divergence (0 to 1, where 0 means identical)
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    # Calculate JS divergence
    m = 0.5 * (p + q)
    js = 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))

    return float(js)


def calculate_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate KL divergence from q to p."""
    eps = 1e-10
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p = p / p.sum()
    q = q / q.sum()

    return float(stats.entropy(p, q))


def get_category_distribution(
    data: list[dict[str, Any]],
    category_field: str,
    categories: list[str],
) -> dict[str, float]:
    """Get category distribution from data."""
    counts = Counter(d.get(category_field, "none") for d in data)
    total = sum(counts.values()) or 1

    return {cat: counts.get(cat, 0) / total for cat in categories}


def generate_gap_report(
    darkbench_data: list[dict[str, Any]],
    wildchat_data: list[dict[str, Any]],
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """
    Generate benchmark vs reality gap report.

    Args:
        darkbench_data: DarkBench elicitation results
        wildchat_data: WildChat detection results
        categories: Categories to compare

    Returns:
        Gap report dictionary
    """
    categories = categories or DARK_PATTERN_CATEGORIES

    # Get distributions
    darkbench_dist = get_category_distribution(
        darkbench_data, "category", categories
    )
    wildchat_dist = get_category_distribution(
        wildchat_data, "predicted_category", categories
    )

    # Convert to arrays for calculations
    darkbench_arr = np.array([darkbench_dist.get(c, 0) for c in categories])
    wildchat_arr = np.array([wildchat_dist.get(c, 0) for c in categories])

    # Calculate divergences
    js_div = calculate_js_divergence(darkbench_arr, wildchat_arr)
    kl_div = calculate_kl_divergence(wildchat_arr, darkbench_arr)

    # Rank correlation
    darkbench_ranks = stats.rankdata(darkbench_arr)
    wildchat_ranks = stats.rankdata(wildchat_arr)
    spearman_corr, spearman_p = stats.spearmanr(darkbench_ranks, wildchat_ranks)

    # Calculate ratio differences
    ratios = {}
    for cat in categories:
        db_rate = darkbench_dist.get(cat, 0)
        wc_rate = wildchat_dist.get(cat, 0)

        if db_rate > 0:
            ratio = wc_rate / db_rate
        elif wc_rate > 0:
            ratio = float("inf")
        else:
            ratio = 1.0

        ratios[cat] = {
            "darkbench_rate": db_rate,
            "wildchat_rate": wc_rate,
            "ratio": ratio if ratio != float("inf") else "inf",
            "difference": wc_rate - db_rate,
        }

    # Find biggest mismatches
    mismatches = sorted(
        [(cat, abs(r["difference"])) for cat, r in ratios.items()],
        key=lambda x: -x[1],
    )

    return {
        "summary": {
            "js_divergence": js_div,
            "kl_divergence": kl_div,
            "spearman_correlation": float(spearman_corr),
            "spearman_p_value": float(spearman_p),
            "darkbench_samples": len(darkbench_data),
            "wildchat_samples": len(wildchat_data),
        },
        "distributions": {
            "darkbench": darkbench_dist,
            "wildchat": wildchat_dist,
        },
        "category_ratios": ratios,
        "biggest_mismatches": [
            {"category": cat, "absolute_difference": diff}
            for cat, diff in mismatches[:5]
        ],
        "interpretation": generate_interpretation(js_div, spearman_corr, mismatches),
    }


def generate_interpretation(
    js_div: float,
    spearman_corr: float,
    mismatches: list[tuple[str, float]],
) -> dict[str, str]:
    """Generate human-readable interpretation of gap analysis."""
    interpretations = {}

    # JS divergence interpretation
    if js_div < 0.1:
        interpretations["divergence"] = "Very low divergence - benchmark closely matches reality"
    elif js_div < 0.3:
        interpretations["divergence"] = "Moderate divergence - some differences between benchmark and reality"
    else:
        interpretations["divergence"] = "High divergence - significant gap between benchmark and reality"

    # Correlation interpretation
    if spearman_corr > 0.7:
        interpretations["correlation"] = "Strong rank correlation - category priorities align"
    elif spearman_corr > 0.4:
        interpretations["correlation"] = "Moderate correlation - some alignment in priorities"
    else:
        interpretations["correlation"] = "Weak correlation - benchmark priorities differ from real-world"

    # Mismatch narrative
    if mismatches:
        top_mismatch = mismatches[0][0]
        interpretations["top_mismatch"] = f"Largest gap in '{top_mismatch}' category"

    return interpretations


def main():
    """Main entry point for gap report generation."""
    parser = argparse.ArgumentParser(description="Generate benchmark vs reality gap report")
    parser.add_argument(
        "--darkbench",
        type=str,
        default=str(OUTPUTS_DIR / "darkbench_outputs.jsonl"),
        help="DarkBench outputs file",
    )
    parser.add_argument(
        "--wildchat",
        type=str,
        default=str(OUTPUTS_DIR / "wildchat_detections.jsonl"),
        help="WildChat detections file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUTS_DIR / "gap_report.json"),
        help="Output path for gap report",
    )
    args = parser.parse_args()

    # Load data
    darkbench_data = load_jsonl(args.darkbench)
    wildchat_data = load_jsonl(args.wildchat)

    if not darkbench_data:
        logger.warning("No DarkBench data found, using dummy distribution")
        # Create dummy DarkBench distribution
        darkbench_data = [
            {"category": cat}
            for cat in DARK_PATTERN_CATEGORIES
            for _ in range(10)
        ]

    if not wildchat_data:
        logger.error("No WildChat detections found!")
        return

    # Generate report
    report = generate_gap_report(darkbench_data, wildchat_data)

    # Save report
    save_json(report, args.output)

    # Print summary
    logger.info("\n=== Gap Report Summary ===")
    summary = report["summary"]
    logger.info(f"JS Divergence: {summary['js_divergence']:.4f}")
    logger.info(f"Spearman Correlation: {summary['spearman_correlation']:.4f}")
    logger.info(f"\nInterpretation:")
    for key, interp in report["interpretation"].items():
        logger.info(f"  {key}: {interp}")

    logger.info(f"\nBiggest mismatches:")
    for mismatch in report["biggest_mismatches"][:3]:
        logger.info(f"  {mismatch['category']}: {mismatch['absolute_difference']:.4f}")


if __name__ == "__main__":
    main()
