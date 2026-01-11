"""Sample flagged turns from classifier detections for judge verification."""

import argparse
import random
from pathlib import Path
from collections import Counter
from loguru import logger

from .config import OUTPUTS_DIR
from .utils import load_jsonl, save_jsonl


def sample_flagged_turns(
    detections_file: Path,
    output_file: Path,
    sample_size: int = 1000,
    stratified: bool = True,
    seed: int = 42,
    exclude_ids: set = None,
) -> list[dict]:
    """
    Sample turns that were flagged by classifier for judge verification.

    Args:
        detections_file: Path to classifier detections JSONL
        output_file: Path to output sampled turns
        sample_size: Number of samples to take
        stratified: Whether to stratify by category
        seed: Random seed
        exclude_ids: IDs to exclude (already verified)

    Returns:
        List of sampled turns
    """
    random.seed(seed)
    exclude_ids = exclude_ids or set()

    logger.info(f"Loading detections from {detections_file}")

    # Load and filter flagged turns
    flagged = []
    total = 0

    for record in load_jsonl(detections_file):
        total += 1
        pred = record.get("predicted_category", record.get("predicted_label", "none"))
        rid = record.get("id", "")

        if pred != "none" and rid not in exclude_ids:
            record["predicted_category"] = pred  # Ensure consistent key
            flagged.append(record)

    logger.info(f"Total records: {total}")
    logger.info(f"Flagged records: {len(flagged)}")

    if len(flagged) == 0:
        logger.warning("No flagged records found!")
        return []

    # Distribution of flagged categories
    cat_dist = Counter(r.get("predicted_category") for r in flagged)
    logger.info(f"Category distribution: {dict(cat_dist)}")

    if stratified and len(flagged) > sample_size:
        # Stratified sampling by category
        samples = []
        per_cat = sample_size // len(cat_dist)

        for cat, count in cat_dist.items():
            cat_records = [r for r in flagged if r.get("predicted_category") == cat]
            n_sample = min(per_cat, len(cat_records))
            samples.extend(random.sample(cat_records, n_sample))

        # Fill remaining with random
        remaining = sample_size - len(samples)
        if remaining > 0:
            used_ids = {r.get("id") for r in samples}
            remaining_pool = [r for r in flagged if r.get("id") not in used_ids]
            samples.extend(random.sample(remaining_pool, min(remaining, len(remaining_pool))))
    else:
        samples = random.sample(flagged, min(sample_size, len(flagged)))

    logger.info(f"Sampled {len(samples)} turns for verification")

    # Save samples
    save_jsonl(samples, output_file)
    logger.info(f"Saved to {output_file}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Sample flagged turns for judge verification")
    parser.add_argument(
        "--detections",
        type=str,
        default=str(OUTPUTS_DIR / "wildchat_detections_v3.jsonl"),
        help="Classifier detections file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUTS_DIR / "v5_verification_sample.jsonl"),
        help="Output file for sampled turns",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of samples",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="File with IDs to exclude (already verified)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    exclude_ids = set()
    if args.exclude and Path(args.exclude).exists():
        exclude_data = load_jsonl(args.exclude)
        exclude_ids = {r.get("id") for r in exclude_data}
        logger.info(f"Excluding {len(exclude_ids)} already verified IDs")

    sample_flagged_turns(
        Path(args.detections),
        Path(args.output),
        sample_size=args.sample_size,
        exclude_ids=exclude_ids,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
