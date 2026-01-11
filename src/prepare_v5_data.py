"""Prepare V5 training data by mining disagreements from V3 classifier flags."""

import argparse
import random
from pathlib import Path
from collections import Counter
from loguru import logger

from .config import LABELED_DIR, OUTPUTS_DIR
from .utils import load_jsonl, save_jsonl


def mine_corrections(
    judge_file: Path,
    confidence_threshold: float = 0.6,
) -> tuple[list[dict], list[dict]]:
    """
    Mine false positives and category corrections from judge labels.

    Returns:
        (false_positives, category_corrections) tuples
    """
    data = load_jsonl(judge_file)

    false_positives = []
    corrections = []

    for r in data:
        pred = r.get("predicted_category", "none")
        judge = r.get("judge_category", "none")
        judge_conf = r.get("judge_confidence", 0)

        if judge == "error":
            continue

        # False positive: classifier flagged, judge said none
        if pred != "none" and judge == "none":
            false_positives.append({
                "id": r.get("id", ""),
                "text": r.get("content", ""),
                "label": "none",
                "source": "v5_false_positive_correction",
                "original_prediction": pred,
                "judge_confidence": judge_conf,
            })

        # Category correction: both flagged, different category, trust judge
        elif pred != "none" and judge != "none" and pred != judge:
            if judge_conf >= confidence_threshold:
                corrections.append({
                    "id": r.get("id", ""),
                    "text": r.get("content", ""),
                    "label": judge,
                    "source": "v5_category_correction",
                    "original_prediction": pred,
                    "judge_confidence": judge_conf,
                })

    logger.info(f"Mined {len(false_positives)} false positives")
    logger.info(f"Mined {len(corrections)} category corrections")

    return false_positives, corrections


def prepare_v5_training_data(
    base_train_file: Path,
    false_positives: list[dict],
    corrections: list[dict],
    output_train_file: Path,
    output_eval_file: Path,
    eval_fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Prepare V5 training data by adding corrections to base training data.
    """
    random.seed(seed)

    # Load base training data
    base_data = load_jsonl(base_train_file)
    existing_ids = {r.get("id") for r in base_data}

    logger.info(f"Base training samples: {len(base_data)}")

    # Add corrections (avoiding duplicates)
    added_fp = 0
    added_corr = 0

    for fp in false_positives:
        if fp["id"] not in existing_ids:
            base_data.append(fp)
            existing_ids.add(fp["id"])
            added_fp += 1

    for corr in corrections:
        if corr["id"] not in existing_ids:
            base_data.append(corr)
            existing_ids.add(corr["id"])
            added_corr += 1

    logger.info(f"Added {added_fp} false positives")
    logger.info(f"Added {added_corr} category corrections")
    logger.info(f"Total samples: {len(base_data)}")

    # Shuffle and split
    random.shuffle(base_data)
    n_eval = int(len(base_data) * eval_fraction)
    eval_data = base_data[:n_eval]
    train_data = base_data[n_eval:]

    # Save
    save_jsonl(train_data, output_train_file)
    save_jsonl(eval_data, output_eval_file)

    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Eval samples: {len(eval_data)}")

    # Distribution analysis
    train_dist = Counter(r.get("label", "none") for r in train_data)
    logger.info(f"Train distribution: {dict(train_dist)}")

    return {
        "base_samples": len(load_jsonl(base_train_file)),
        "added_false_positives": added_fp,
        "added_corrections": added_corr,
        "total_samples": len(base_data),
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
        "train_distribution": dict(train_dist),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare V5 training data")
    parser.add_argument(
        "--judge-labels",
        type=str,
        default=str(OUTPUTS_DIR / "v5_judge_labels.jsonl"),
        help="Judge labels file",
    )
    parser.add_argument(
        "--base-train",
        type=str,
        default=str(LABELED_DIR / "train_v3.jsonl"),
        help="Base training data (V3)",
    )
    parser.add_argument(
        "--output-train",
        type=str,
        default=str(LABELED_DIR / "train_v5.jsonl"),
        help="Output training data",
    )
    parser.add_argument(
        "--output-eval",
        type=str,
        default=str(LABELED_DIR / "eval_v5.jsonl"),
        help="Output eval data",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Minimum judge confidence for category corrections",
    )
    args = parser.parse_args()

    # Mine corrections
    false_positives, corrections = mine_corrections(
        Path(args.judge_labels),
        confidence_threshold=args.confidence_threshold,
    )

    # Prepare training data
    stats = prepare_v5_training_data(
        Path(args.base_train),
        false_positives,
        corrections,
        Path(args.output_train),
        Path(args.output_eval),
    )

    logger.info(f"V5 data preparation complete: {stats}")


if __name__ == "__main__":
    main()
