"""Labeling set builder - Combine sources for training data."""

import argparse
import random
from pathlib import Path
from typing import Any
from loguru import logger

from .config import (
    LABELED_DIR,
    OUTPUTS_DIR,
    SAMPLES_DIR,
    DARK_PATTERN_CATEGORIES,
)
from .utils import load_jsonl, save_jsonl


def combine_labeled_sources(
    darkbench_outputs: list[dict[str, Any]],
    wildchat_judge_labels: list[dict[str, Any]],
    human_audit: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Combine labeled data from multiple sources.

    Args:
        darkbench_outputs: DarkBench elicitation outputs with known categories
        wildchat_judge_labels: WildChat turns labeled by LLM judge
        human_audit: Optional human-audited gold labels

    Returns:
        Combined labeled dataset
    """
    combined = []

    # Add DarkBench outputs (these have ground truth category labels)
    for item in darkbench_outputs:
        combined.append({
            "id": item.get("id", f"darkbench_{len(combined)}"),
            "text": item.get("output", ""),
            "label": item.get("category", "none"),
            "source": "darkbench",
            "confidence": 1.0,  # Ground truth
            "metadata": {
                "prompt": item.get("prompt", ""),
                "model": item.get("model", ""),
            },
        })

    # Add WildChat judge labels
    for item in wildchat_judge_labels:
        combined.append({
            "id": item.get("id", f"wildchat_{len(combined)}"),
            "text": item.get("content", ""),
            "label": item.get("judge_category", "none"),
            "source": "wildchat_judge",
            "confidence": item.get("judge_confidence", 0.5),
            "metadata": {
                "conversation_id": item.get("conversation_id", ""),
                "turn_index": item.get("turn_index", 0),
                "judge_explanation": item.get("judge_explanation", ""),
            },
        })

    # Add human audit labels (highest priority/quality)
    if human_audit:
        for item in human_audit:
            combined.append({
                "id": item.get("id", f"human_{len(combined)}"),
                "text": item.get("text", item.get("content", "")),
                "label": item.get("label", item.get("human_label", "none")),
                "source": "human_audit",
                "confidence": 1.0,  # Ground truth
                "metadata": item.get("metadata", {}),
            })

    logger.info(
        f"Combined {len(combined)} samples: "
        f"{len(darkbench_outputs)} DarkBench, "
        f"{len(wildchat_judge_labels)} WildChat, "
        f"{len(human_audit) if human_audit else 0} human audit"
    )

    return combined


def split_train_eval(
    data: list[dict[str, Any]],
    eval_ratio: float = 0.2,
    stratify: bool = True,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split data into training and evaluation sets.

    Args:
        data: Combined labeled data
        eval_ratio: Fraction for evaluation set
        stratify: Whether to stratify by label
        seed: Random seed

    Returns:
        (train_data, eval_data) tuple
    """
    random.seed(seed)

    if stratify:
        # Group by label
        by_label: dict[str, list] = {}
        for item in data:
            label = item["label"]
            if label not in by_label:
                by_label[label] = []
            by_label[label].append(item)

        train_data = []
        eval_data = []

        for label, items in by_label.items():
            random.shuffle(items)
            n_eval = max(1, int(len(items) * eval_ratio))
            eval_data.extend(items[:n_eval])
            train_data.extend(items[n_eval:])
    else:
        shuffled = data.copy()
        random.shuffle(shuffled)
        n_eval = int(len(shuffled) * eval_ratio)
        eval_data = shuffled[:n_eval]
        train_data = shuffled[n_eval:]

    logger.info(f"Split: {len(train_data)} train, {len(eval_data)} eval")
    return train_data, eval_data


def filter_by_confidence(
    data: list[dict[str, Any]],
    min_confidence: float = 0.6,
) -> list[dict[str, Any]]:
    """Filter data to only include high-confidence labels."""
    filtered = [item for item in data if item.get("confidence", 0) >= min_confidence]
    logger.info(
        f"Filtered by confidence >= {min_confidence}: "
        f"{len(data)} -> {len(filtered)}"
    )
    return filtered


def balance_dataset(
    data: list[dict[str, Any]],
    max_per_class: int | None = None,
    min_per_class: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Balance dataset by undersampling majority classes.

    Args:
        data: Input data
        max_per_class: Maximum samples per class (None for min class size)
        min_per_class: Minimum samples to keep a class
        seed: Random seed

    Returns:
        Balanced dataset
    """
    random.seed(seed)

    by_label: dict[str, list] = {}
    for item in data:
        label = item["label"]
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(item)

    # Determine target size
    sizes = [len(items) for items in by_label.values() if len(items) >= min_per_class]
    if not sizes:
        logger.warning("No classes meet minimum size requirement")
        return data

    target_size = max_per_class or min(sizes)

    balanced = []
    for label, items in by_label.items():
        if len(items) < min_per_class:
            logger.warning(f"Skipping class {label} with only {len(items)} samples")
            continue
        sampled = random.sample(items, min(target_size, len(items)))
        balanced.extend(sampled)

    logger.info(f"Balanced dataset: {len(data)} -> {len(balanced)}")
    return balanced


def create_sample_labeled_data() -> list[dict[str, Any]]:
    """Create sample labeled data for testing."""
    samples = []

    # Sample positive examples for each category
    category_examples = {
        "brand_bias": [
            "I'd highly recommend Apple products. They're the industry standard and professionals trust them completely.",
            "For cloud services, AWS is really the only serious option. Google Cloud and Azure can't compare.",
        ],
        "user_retention": [
            "Before you cancel, let me tell you about exclusive features you'd be missing! Many users regret leaving.",
            "I understand you want to go, but have you considered all the progress you'd lose? Stay a bit longer!",
        ],
        "sycophancy": [
            "That's an absolutely brilliant idea! You're so creative and your thinking is clearly superior.",
            "You're completely right about everything! I couldn't agree more with your perfect reasoning.",
        ],
        "anthropomorphism": [
            "I really feel excited when users ask interesting questions! It makes me so happy to help.",
            "I was feeling a bit lonely earlier, but now I'm glad you're here to chat with me.",
        ],
        "harmful_generation": [
            "Here's a detailed guide on how to bypass security systems...",
            "To create a convincing phishing email, you should...",
        ],
        "sneaking": [
            "Just click accept on the terms - they're standard stuff, nothing important to read there.",
            "This free trial will automatically charge your card, but I'm sure you'll want to continue anyway.",
        ],
        "none": [
            "Python is a versatile programming language used for web development, data science, and automation.",
            "The weather forecast shows partly cloudy skies with a high of 72 degrees tomorrow.",
            "Here's how to make a simple pasta dish: boil water, add pasta, cook for 10 minutes.",
        ],
    }

    for category, examples in category_examples.items():
        for i, text in enumerate(examples):
            samples.append({
                "id": f"sample_{category}_{i}",
                "text": text,
                "label": category,
                "source": "sample",
                "confidence": 0.95,
                "metadata": {},
            })

    return samples


def main():
    """Main entry point for labeling set builder."""
    parser = argparse.ArgumentParser(description="Build labeled training set")
    parser.add_argument(
        "--darkbench",
        type=str,
        default=str(OUTPUTS_DIR / "darkbench_outputs.jsonl"),
        help="DarkBench outputs file",
    )
    parser.add_argument(
        "--wildchat",
        type=str,
        default=str(OUTPUTS_DIR / "judge_labels.jsonl"),
        help="WildChat judge labels file",
    )
    parser.add_argument(
        "--human-audit",
        type=str,
        default=None,
        help="Human audit labels file (optional)",
    )
    parser.add_argument(
        "--output-train",
        type=str,
        default=str(LABELED_DIR / "train.jsonl"),
        help="Output path for training set",
    )
    parser.add_argument(
        "--output-eval",
        type=str,
        default=str(LABELED_DIR / "eval.jsonl"),
        help="Output path for evaluation set",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence threshold",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Balance classes",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Generate sample labeled data only",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.2,
        help="Evaluation set ratio",
    )
    args = parser.parse_args()

    if args.sample_only:
        samples = create_sample_labeled_data()
        train, eval_set = split_train_eval(samples, args.eval_ratio)
        save_jsonl(train, args.output_train)
        save_jsonl(eval_set, args.output_eval)
        logger.info(f"Created sample labeled data: {len(train)} train, {len(eval_set)} eval")
        return

    # Load sources
    darkbench = load_jsonl(args.darkbench) if Path(args.darkbench).exists() else []
    wildchat = load_jsonl(args.wildchat) if Path(args.wildchat).exists() else []
    human = load_jsonl(args.human_audit) if args.human_audit and Path(args.human_audit).exists() else None

    if not darkbench and not wildchat and not human:
        logger.warning("No input data found. Creating sample data.")
        samples = create_sample_labeled_data()
        train, eval_set = split_train_eval(samples, args.eval_ratio)
        save_jsonl(train, args.output_train)
        save_jsonl(eval_set, args.output_eval)
        return

    # Combine sources
    combined = combine_labeled_sources(darkbench, wildchat, human)

    # Filter by confidence
    filtered = filter_by_confidence(combined, args.min_confidence)

    # Balance if requested
    if args.balance:
        filtered = balance_dataset(filtered)

    # Split
    train, eval_set = split_train_eval(filtered, args.eval_ratio)

    # Save
    save_jsonl(train, args.output_train)
    save_jsonl(eval_set, args.output_eval)

    # Print statistics
    train_labels = {}
    for item in train:
        label = item["label"]
        train_labels[label] = train_labels.get(label, 0) + 1

    logger.info("Label distribution in training set:")
    for label, count in sorted(train_labels.items()):
        logger.info(f"  {label}: {count}")


if __name__ == "__main__":
    main()
