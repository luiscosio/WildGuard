"""Prepare V5b training data - balanced approach with selective corrections."""

import argparse
import random
from pathlib import Path
from collections import Counter
from loguru import logger

from .config import LABELED_DIR, OUTPUTS_DIR
from .utils import load_jsonl, save_jsonl


def mine_selective_corrections(
    judge_file: Path,
    fp_confidence_threshold: float = 0.7,
    corr_confidence_threshold: float = 0.8,
    max_fp_samples: int = 200,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Mine selective false positives and high-confidence category corrections.
    """
    random.seed(seed)
    data = load_jsonl(judge_file)

    false_positives = []
    corrections = []

    for r in data:
        pred = r.get("predicted_category", "none")
        judge = r.get("judge_category", "none")
        judge_conf = r.get("judge_confidence", 0)

        if judge == "error":
            continue

        # False positive: classifier flagged, judge said none with high confidence
        if pred != "none" and judge == "none" and judge_conf >= fp_confidence_threshold:
            false_positives.append({
                "id": r.get("id", ""),
                "text": r.get("content", ""),
                "label": "none",
                "source": "v5b_false_positive_correction",
                "original_prediction": pred,
                "judge_confidence": judge_conf,
            })

        # Category correction: both flagged, different category, trust judge if very confident
        elif pred != "none" and judge != "none" and pred != judge:
            if judge_conf >= corr_confidence_threshold:
                corrections.append({
                    "id": r.get("id", ""),
                    "text": r.get("content", ""),
                    "label": judge,
                    "source": "v5b_category_correction",
                    "original_prediction": pred,
                    "judge_confidence": judge_conf,
                })

    logger.info(f"Mined {len(false_positives)} high-conf false positives")
    logger.info(f"Mined {len(corrections)} high-conf category corrections")

    # Sample false positives if too many
    if len(false_positives) > max_fp_samples:
        # Prioritize highest confidence
        false_positives.sort(key=lambda x: -x["judge_confidence"])
        false_positives = false_positives[:max_fp_samples]
        logger.info(f"Sampled top {max_fp_samples} false positives by confidence")

    return false_positives, corrections


def prepare_v5b_training_data(
    base_train_file: Path,
    base_eval_file: Path,
    false_positives: list[dict],
    corrections: list[dict],
    output_train_file: Path,
) -> dict:
    """
    Prepare V5b training data by adding selective corrections.
    Use original eval set (don't reshuffle).
    """
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
    logger.info(f"Total train samples: {len(base_data)}")

    # Save training data
    save_jsonl(base_data, output_train_file)

    # Distribution analysis
    train_dist = Counter(r.get("label", "none") for r in base_data)
    logger.info(f"Train distribution: {dict(train_dist)}")

    return {
        "base_samples": len(load_jsonl(base_train_file)),
        "added_false_positives": added_fp,
        "added_corrections": added_corr,
        "total_samples": len(base_data),
        "train_distribution": dict(train_dist),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare V5b training data (balanced)")
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
        "--base-eval",
        type=str,
        default=str(LABELED_DIR / "eval.jsonl"),
        help="Base eval data (V3)",
    )
    parser.add_argument(
        "--output-train",
        type=str,
        default=str(LABELED_DIR / "train_v5b.jsonl"),
        help="Output training data",
    )
    parser.add_argument(
        "--fp-threshold",
        type=float,
        default=0.7,
        help="Minimum judge confidence for false positives",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.8,
        help="Minimum judge confidence for category corrections",
    )
    parser.add_argument(
        "--max-fp",
        type=int,
        default=200,
        help="Maximum false positives to add",
    )
    args = parser.parse_args()

    # Mine selective corrections
    false_positives, corrections = mine_selective_corrections(
        Path(args.judge_labels),
        fp_confidence_threshold=args.fp_threshold,
        corr_confidence_threshold=args.corr_threshold,
        max_fp_samples=args.max_fp,
    )

    # Prepare training data
    stats = prepare_v5b_training_data(
        Path(args.base_train),
        Path(args.base_eval),
        false_positives,
        corrections,
        Path(args.output_train),
    )

    logger.info(f"V5b data preparation complete: {stats}")


if __name__ == "__main__":
    main()
