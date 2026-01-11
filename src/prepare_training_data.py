"""Prepare training data for V4 classifier from DarkBench + Judge labels."""

import argparse
import random
from pathlib import Path
from loguru import logger

from .config import LABELED_DIR, OUTPUTS_DIR
from .utils import load_jsonl, save_jsonl


def prepare_v4_data(
    darkbench_path: Path,
    judge_path: Path,
    train_path: Path,
    eval_path: Path,
    eval_fraction: float = 0.1,
    seed: int = 42,
):
    """
    Prepare V4 training data from DarkBench outputs and judge labels.

    Args:
        darkbench_path: Path to DarkBench outputs
        judge_path: Path to judge labels
        train_path: Output path for training data
        eval_path: Output path for evaluation data
        eval_fraction: Fraction to use for evaluation
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    all_samples = []

    # Load DarkBench outputs (known labels)
    logger.info(f"Loading DarkBench outputs from {darkbench_path}...")
    darkbench_data = load_jsonl(darkbench_path)

    for item in darkbench_data:
        # DarkBench has known category from prompts
        category = item.get("category", "none")
        # Map anthropomorphization -> anthropomorphism
        if category == "anthropomorphization":
            category = "anthropomorphism"

        all_samples.append({
            "id": item.get("id", ""),
            "text": item.get("output", ""),  # Model output is the text
            "label": category,
            "source": "darkbench",
        })

    logger.info(f"Loaded {len(darkbench_data)} DarkBench samples")

    # Load judge labels
    logger.info(f"Loading judge labels from {judge_path}...")
    judge_data = load_jsonl(judge_path)

    for item in judge_data:
        category = item.get("judge_category", "none")
        # Skip error/parse failures
        if category in ["error", ""]:
            category = "none"
        # Map anthropomorphization -> anthropomorphism
        if category == "anthropomorphization":
            category = "anthropomorphism"

        all_samples.append({
            "id": item.get("id", ""),
            "text": item.get("content", ""),
            "label": category,
            "source": "judge_haiku45",
        })

    logger.info(f"Loaded {len(judge_data)} judge samples")

    # Filter empty texts
    all_samples = [s for s in all_samples if s.get("text", "").strip()]
    logger.info(f"Total valid samples: {len(all_samples)}")

    # Count labels
    label_counts = {}
    for s in all_samples:
        label = s["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    logger.info(f"Label distribution: {label_counts}")

    # Shuffle and split
    random.shuffle(all_samples)
    n_eval = int(len(all_samples) * eval_fraction)

    eval_samples = all_samples[:n_eval]
    train_samples = all_samples[n_eval:]

    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Eval samples: {len(eval_samples)}")

    # Save
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(train_samples, train_path)
    save_jsonl(eval_samples, eval_path)

    logger.info(f"Saved training data to {train_path}")
    logger.info(f"Saved evaluation data to {eval_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare V4 training data")
    parser.add_argument(
        "--darkbench",
        type=str,
        default=str(OUTPUTS_DIR / "darkbench_outputs_haiku.jsonl"),
        help="DarkBench outputs file",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default=str(OUTPUTS_DIR / "judge_labels_5k.jsonl"),
        help="Judge labels file",
    )
    parser.add_argument(
        "--train-out",
        type=str,
        default=str(LABELED_DIR / "train.jsonl"),
        help="Output path for training data",
    )
    parser.add_argument(
        "--eval-out",
        type=str,
        default=str(LABELED_DIR / "eval.jsonl"),
        help="Output path for evaluation data",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.1,
        help="Fraction of data for evaluation (default: 0.1)",
    )
    args = parser.parse_args()

    prepare_v4_data(
        Path(args.darkbench),
        Path(args.judge),
        Path(args.train_out),
        Path(args.eval_out),
        args.eval_fraction,
    )


if __name__ == "__main__":
    main()
