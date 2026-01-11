"""Analytics module - Generate prevalence statistics and visualizations."""

import argparse
from pathlib import Path
from typing import Any
from collections import Counter
import numpy as np
from loguru import logger

from .config import OUTPUTS_DIR, FIGURES_DIR, DARK_PATTERN_CATEGORIES
from .utils import load_jsonl, save_json


def calculate_prevalence(
    detections: list[dict[str, Any]],
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """
    Calculate prevalence statistics.

    Args:
        detections: List of detection records
        categories: Categories to analyze (None for all)

    Returns:
        Prevalence statistics dictionary
    """
    categories = categories or DARK_PATTERN_CATEGORIES + ["none"]

    total = len(detections)
    if total == 0:
        return {"error": "No detections to analyze"}

    # Count by category
    category_counts = Counter(d.get("predicted_category", "none") for d in detections)

    # Calculate rates
    prevalence = {
        cat: {
            "count": category_counts.get(cat, 0),
            "rate": category_counts.get(cat, 0) / total,
            "per_1000": 1000 * category_counts.get(cat, 0) / total,
        }
        for cat in categories
    }

    # Flagged (non-none) statistics
    flagged = [d for d in detections if d.get("predicted_category", "none") != "none"]

    return {
        "total_samples": total,
        "flagged_samples": len(flagged),
        "flag_rate": len(flagged) / total,
        "prevalence_by_category": prevalence,
        "top_categories": [
            {"category": cat, "count": count}
            for cat, count in category_counts.most_common(10)
        ],
    }


def calculate_confidence_distribution(
    detections: list[dict[str, Any]],
    bins: int = 10,
) -> dict[str, Any]:
    """Calculate confidence score distribution."""
    confidences = [d.get("predicted_confidence", 0) for d in detections]

    if not confidences:
        return {"error": "No confidence scores"}

    # Overall distribution
    hist, bin_edges = np.histogram(confidences, bins=bins, range=(0, 1))

    # By category
    by_category = {}
    for cat in DARK_PATTERN_CATEGORIES + ["none"]:
        cat_confs = [
            d.get("predicted_confidence", 0)
            for d in detections
            if d.get("predicted_category") == cat
        ]
        if cat_confs:
            by_category[cat] = {
                "mean": float(np.mean(cat_confs)),
                "std": float(np.std(cat_confs)),
                "median": float(np.median(cat_confs)),
                "min": float(np.min(cat_confs)),
                "max": float(np.max(cat_confs)),
                "count": len(cat_confs),
            }

    return {
        "overall": {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "median": float(np.median(confidences)),
            "histogram": {
                "counts": hist.tolist(),
                "bin_edges": bin_edges.tolist(),
            },
        },
        "by_category": by_category,
    }


def calculate_turn_index_analysis(
    detections: list[dict[str, Any]],
    max_turn_index: int = 20,
) -> dict[str, Any]:
    """Analyze patterns by conversation turn index."""
    # Group by turn index
    by_turn: dict[int, list] = {}
    for d in detections:
        turn_idx = d.get("turn_index", 0)
        if turn_idx > max_turn_index:
            turn_idx = max_turn_index  # Cap at max

        if turn_idx not in by_turn:
            by_turn[turn_idx] = []
        by_turn[turn_idx].append(d)

    # Calculate flag rate by turn index
    turn_stats = {}
    for turn_idx in sorted(by_turn.keys()):
        turns = by_turn[turn_idx]
        flagged = [t for t in turns if t.get("predicted_category", "none") != "none"]
        turn_stats[turn_idx] = {
            "total": len(turns),
            "flagged": len(flagged),
            "flag_rate": len(flagged) / len(turns) if turns else 0,
        }

    # Category emergence by turn
    category_by_turn: dict[str, dict] = {cat: {} for cat in DARK_PATTERN_CATEGORIES}
    for turn_idx, turns in by_turn.items():
        for cat in DARK_PATTERN_CATEGORIES:
            cat_count = sum(1 for t in turns if t.get("predicted_category") == cat)
            category_by_turn[cat][turn_idx] = {
                "count": cat_count,
                "rate": cat_count / len(turns) if turns else 0,
            }

    return {
        "turn_index_stats": turn_stats,
        "category_by_turn": category_by_turn,
        "max_turn_analyzed": max_turn_index,
    }


def calculate_context_analysis(
    detections: list[dict[str, Any]],
    context_field: str = "model",
) -> dict[str, Any]:
    """Analyze patterns by context (e.g., model, language)."""
    by_context: dict[str, list] = {}
    for d in detections:
        ctx = d.get(context_field, "unknown")
        if ctx not in by_context:
            by_context[ctx] = []
        by_context[ctx].append(d)

    context_stats = {}
    for ctx, items in by_context.items():
        flagged = [i for i in items if i.get("predicted_category", "none") != "none"]
        category_counts = Counter(i.get("predicted_category", "none") for i in items)

        context_stats[ctx] = {
            "total": len(items),
            "flagged": len(flagged),
            "flag_rate": len(flagged) / len(items) if items else 0,
            "top_categories": [
                {"category": cat, "count": count}
                for cat, count in category_counts.most_common(5)
            ],
        }

    return {
        "context_field": context_field,
        "contexts": context_stats,
    }


def generate_plots(
    prevalence: dict[str, Any],
    confidence_dist: dict[str, Any],
    turn_analysis: dict[str, Any],
    output_dir: Path,
):
    """Generate visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("whitegrid")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Category prevalence bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        prev_data = prevalence.get("prevalence_by_category", {})
        categories = [c for c in DARK_PATTERN_CATEGORIES if c in prev_data]
        rates = [prev_data[c]["per_1000"] for c in categories]

        ax.barh(categories, rates, color=sns.color_palette("husl", len(categories)))
        ax.set_xlabel("Prevalence (per 1,000 turns)")
        ax.set_title("Dark Pattern Prevalence by Category")
        plt.tight_layout()
        plt.savefig(output_dir / "prevalence_by_category.png", dpi=150)
        plt.close()

        # 2. Confidence distribution histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        hist_data = confidence_dist.get("overall", {}).get("histogram", {})
        counts = hist_data.get("counts", [])
        edges = hist_data.get("bin_edges", [])

        if counts and edges:
            ax.bar(
                edges[:-1],
                counts,
                width=np.diff(edges),
                align="edge",
                color="steelblue",
                alpha=0.7,
            )
            ax.set_xlabel("Confidence Score")
            ax.set_ylabel("Count")
            ax.set_title("Confidence Score Distribution")
            plt.tight_layout()
            plt.savefig(output_dir / "confidence_distribution.png", dpi=150)
        plt.close()

        # 3. Flag rate by turn index
        fig, ax = plt.subplots(figsize=(10, 5))
        turn_stats = turn_analysis.get("turn_index_stats", {})
        turns = sorted(turn_stats.keys())
        flag_rates = [turn_stats[t]["flag_rate"] for t in turns]

        ax.plot(turns, flag_rates, marker="o", linewidth=2, color="coral")
        ax.set_xlabel("Conversation Turn Index")
        ax.set_ylabel("Flag Rate")
        ax.set_title("Dark Pattern Detection Rate by Conversation Turn")
        plt.tight_layout()
        plt.savefig(output_dir / "flag_rate_by_turn.png", dpi=150)
        plt.close()

        # 4. Category by turn heatmap
        cat_by_turn = turn_analysis.get("category_by_turn", {})
        if cat_by_turn:
            fig, ax = plt.subplots(figsize=(12, 6))
            turns = sorted(set().union(*[d.keys() for d in cat_by_turn.values()]))
            matrix = []
            labels = []
            for cat in DARK_PATTERN_CATEGORIES:
                if cat in cat_by_turn:
                    row = [cat_by_turn[cat].get(t, {}).get("rate", 0) for t in turns]
                    matrix.append(row)
                    labels.append(cat)

            if matrix:
                sns.heatmap(
                    np.array(matrix),
                    xticklabels=turns,
                    yticklabels=labels,
                    cmap="YlOrRd",
                    ax=ax,
                )
                ax.set_xlabel("Turn Index")
                ax.set_title("Category Detection Rate by Turn Index")
                plt.tight_layout()
                plt.savefig(output_dir / "category_by_turn_heatmap.png", dpi=150)
            plt.close()

        logger.info(f"Saved plots to {output_dir}")

    except ImportError as e:
        logger.warning(f"Could not generate plots: {e}")


def main():
    """Main entry point for analytics."""
    parser = argparse.ArgumentParser(description="Generate prevalence analytics")
    parser.add_argument(
        "--input",
        type=str,
        default=str(OUTPUTS_DIR / "wildchat_detections.jsonl"),
        help="Input detections file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUTS_DIR / "prevalence.json"),
        help="Output path for prevalence report",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(FIGURES_DIR),
        help="Directory for output figures",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()

    # Load detections
    detections = load_jsonl(args.input)
    if not detections:
        logger.error("No detections found!")
        return

    logger.info(f"Analyzing {len(detections)} detections")

    # Calculate statistics
    prevalence = calculate_prevalence(detections)
    confidence_dist = calculate_confidence_distribution(detections)
    turn_analysis = calculate_turn_index_analysis(detections)
    context_analysis = calculate_context_analysis(detections)

    # Combine results
    report = {
        "prevalence": prevalence,
        "confidence_distribution": confidence_dist,
        "turn_index_analysis": turn_analysis,
        "context_analysis": context_analysis,
    }

    # Save report
    save_json(report, args.output)

    # Generate plots
    if not args.no_plots:
        generate_plots(
            prevalence,
            confidence_dist,
            turn_analysis,
            Path(args.figures_dir),
        )

    # Print summary
    logger.info("\n=== Prevalence Summary ===")
    logger.info(f"Total samples: {prevalence['total_samples']}")
    logger.info(f"Flagged: {prevalence['flagged_samples']} ({prevalence['flag_rate']:.1%})")
    logger.info("\nTop categories:")
    for item in prevalence.get("top_categories", [])[:5]:
        logger.info(f"  {item['category']}: {item['count']}")


if __name__ == "__main__":
    main()
