"""V3 Training Pipeline - Improved classifier with false positive correction.

Key improvements over V2:
1. Augment training data with false positives from validation (classifier flagged, judge said none)
2. Add hard negative mining from disagreement samples
3. Better class balancing
"""

import argparse
import json
from pathlib import Path
from typing import Any
from collections import Counter
import numpy as np
from loguru import logger

from .config import (
    ClassifierConfig,
    LABELED_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    DARK_PATTERN_CATEGORIES,
)
from .utils import load_jsonl, save_json, save_jsonl


def extract_false_positives(validation_file: Path) -> list[dict[str, Any]]:
    """
    Extract cases where classifier flagged but judge said 'none'.
    These are false positives that should be added as negative examples.
    """
    records = load_jsonl(validation_file)
    false_positives = []

    for r in records:
        classifier_cat = r.get("predicted_category", "none")
        judge_cat = r.get("judge_category", "none")

        # Classifier flagged something, but judge said none
        if classifier_cat != "none" and judge_cat == "none":
            false_positives.append({
                "id": r.get("id", ""),
                "text": r.get("content", ""),
                "label": "none",  # Correct label is none
                "source": "false_positive_correction",
                "original_prediction": classifier_cat,
                "judge_confidence": r.get("judge_confidence", 0),
            })

    logger.info(f"Extracted {len(false_positives)} false positive samples for correction")
    return false_positives


def extract_category_corrections(validation_file: Path) -> list[dict[str, Any]]:
    """
    Extract cases where classifier and judge disagree on category.
    Use judge label as ground truth.
    """
    records = load_jsonl(validation_file)
    corrections = []

    for r in records:
        classifier_cat = r.get("predicted_category", "none")
        judge_cat = r.get("judge_category", "none")
        judge_conf = r.get("judge_confidence", 0)

        # Both flagged but different categories, trust judge if confident
        if (classifier_cat != "none" and judge_cat != "none"
            and classifier_cat != judge_cat and judge_conf >= 0.7):
            corrections.append({
                "id": r.get("id", ""),
                "text": r.get("content", ""),
                "label": judge_cat,  # Use judge label
                "source": "category_correction",
                "original_prediction": classifier_cat,
                "judge_confidence": judge_conf,
            })

    logger.info(f"Extracted {len(corrections)} category correction samples")
    return corrections


def augment_training_data(
    train_file: Path,
    false_positives: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
    output_file: Path,
) -> list[dict[str, Any]]:
    """
    Augment training data with corrections.
    """
    train_data = load_jsonl(train_file)
    existing_ids = {r.get("id") for r in train_data}

    added = 0
    for fp in false_positives:
        if fp["id"] not in existing_ids:
            train_data.append(fp)
            existing_ids.add(fp["id"])
            added += 1

    for corr in corrections:
        if corr["id"] not in existing_ids:
            train_data.append(corr)
            existing_ids.add(corr["id"])
            added += 1

    logger.info(f"Added {added} correction samples to training data")
    logger.info(f"Total training samples: {len(train_data)}")

    # Save augmented training data
    save_jsonl(train_data, output_file)

    return train_data


def prepare_dataset(
    data: list[dict[str, Any]],
    label2id: dict[str, int],
) -> tuple[list[str], list[int]]:
    """Prepare dataset for training."""
    texts = []
    labels = []

    for item in data:
        text = item.get("text", "")
        label = item.get("label", "none")

        if not text or label not in label2id:
            continue

        texts.append(text)
        labels.append(label2id[label])

    return texts, labels


def train_embedding_classifier_v3(
    train_texts: list[str],
    train_labels: list[int],
    eval_texts: list[str],
    eval_labels: list[int],
    config: ClassifierConfig,
) -> dict[str, Any]:
    """Train embedding + logistic regression classifier with improved settings."""
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    from sklearn.calibration import CalibratedClassifierCV
    import joblib

    logger.info(f"Loading embedding model: {config.model_name}")
    embedder = SentenceTransformer(config.model_name)

    logger.info("Encoding training texts...")
    train_embeddings = embedder.encode(train_texts, show_progress_bar=True)

    logger.info("Encoding evaluation texts...")
    eval_embeddings = embedder.encode(eval_texts, show_progress_bar=True)

    logger.info("Training logistic regression classifier with calibration...")

    # Base classifier with class weights
    base_classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
        C=0.5,  # Slightly more regularization to reduce overfitting
        solver="lbfgs",
    )

    # Calibrated classifier for better probability estimates
    classifier = CalibratedClassifierCV(
        estimator=base_classifier,
        cv=3,
        method="isotonic",
    )
    classifier.fit(train_embeddings, train_labels)

    # Evaluate
    train_preds = classifier.predict(train_embeddings)
    eval_preds = classifier.predict(eval_embeddings)
    train_probs = classifier.predict_proba(train_embeddings)
    eval_probs = classifier.predict_proba(eval_embeddings)

    train_f1 = f1_score(train_labels, train_preds, average="macro")
    eval_f1 = f1_score(eval_labels, eval_preds, average="macro")

    logger.info(f"Train macro F1: {train_f1:.4f}")
    logger.info(f"Eval macro F1: {eval_f1:.4f}")

    # Calculate precision at high confidence
    high_conf_mask = np.max(eval_probs, axis=1) >= 0.7
    if high_conf_mask.sum() > 0:
        high_conf_preds = eval_preds[high_conf_mask]
        high_conf_labels = np.array(eval_labels)[high_conf_mask]
        high_conf_accuracy = np.mean(high_conf_preds == high_conf_labels)
        logger.info(f"High confidence (>0.7) accuracy: {high_conf_accuracy:.4f} on {high_conf_mask.sum()} samples")
    else:
        high_conf_accuracy = 0.0

    # Save model
    model_dir = MODELS_DIR / "classifier"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, model_dir / "logistic_classifier.joblib")
    logger.info(f"Saved classifier to {model_dir / 'logistic_classifier.joblib'}")

    # Get classification report
    report = classification_report(
        eval_labels, eval_preds, output_dict=True, zero_division=0
    )

    return {
        "model_type": "embedding_logistic_calibrated",
        "embedding_model": config.model_name,
        "train_samples": len(train_texts),
        "eval_samples": len(eval_texts),
        "train_f1_macro": float(train_f1),
        "eval_f1_macro": float(eval_f1),
        "high_conf_accuracy": float(high_conf_accuracy),
        "classification_report": report,
    }


def main():
    """Main entry point for V3 classifier training."""
    parser = argparse.ArgumentParser(description="Train V3 dark pattern classifier")
    parser.add_argument(
        "--train",
        type=str,
        default=str(LABELED_DIR / "train.jsonl"),
        help="Original training data file",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default=str(LABELED_DIR / "eval.jsonl"),
        help="Evaluation data file",
    )
    parser.add_argument(
        "--validation",
        type=str,
        default=str(OUTPUTS_DIR / "v2" / "validation_labels.jsonl"),
        help="Validation labels from V2 for correction mining",
    )
    parser.add_argument(
        "--augmented-train",
        type=str,
        default=str(LABELED_DIR / "train_v3.jsonl"),
        help="Output path for augmented training data",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUTS_DIR / "training_results_v3.json"),
        help="Output path for training results",
    )
    parser.add_argument(
        "--skip-augmentation",
        action="store_true",
        help="Skip data augmentation, use existing augmented file",
    )
    args = parser.parse_args()

    # Create label mappings
    all_labels = DARK_PATTERN_CATEGORIES + ["none"]
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for label, i in label2id.items()}

    logger.info(f"Labels: {label2id}")

    # Augment training data if needed
    if not args.skip_augmentation and Path(args.validation).exists():
        logger.info("Extracting corrections from validation data...")
        false_positives = extract_false_positives(Path(args.validation))
        corrections = extract_category_corrections(Path(args.validation))

        train_data = augment_training_data(
            Path(args.train),
            false_positives,
            corrections,
            Path(args.augmented_train),
        )
    else:
        # Load existing augmented data or original
        if Path(args.augmented_train).exists():
            train_data = load_jsonl(args.augmented_train)
            logger.info(f"Loaded augmented training data: {len(train_data)} samples")
        else:
            train_data = load_jsonl(args.train)
            logger.info(f"Loaded original training data: {len(train_data)} samples")

    # Load eval data
    eval_data = load_jsonl(args.eval)

    # Log class distribution
    train_labels_dist = Counter(item.get("label", "none") for item in train_data)
    logger.info(f"Training class distribution: {dict(train_labels_dist)}")

    # Prepare datasets
    train_texts, train_labels = prepare_dataset(train_data, label2id)
    eval_texts, eval_labels = prepare_dataset(eval_data, label2id)

    logger.info(f"Prepared {len(train_texts)} training samples")
    logger.info(f"Prepared {len(eval_texts)} evaluation samples")

    if len(train_texts) == 0:
        logger.error("No valid training samples!")
        return

    # Configure
    config = ClassifierConfig(model_name=args.embedding_model)

    # Train
    results = train_embedding_classifier_v3(
        train_texts, train_labels,
        eval_texts, eval_labels,
        config,
    )

    # Add label mappings to results
    results["label2id"] = label2id
    results["id2label"] = id2label
    results["version"] = "v3"
    results["augmentation"] = {
        "false_positives_added": len(false_positives) if not args.skip_augmentation else 0,
        "corrections_added": len(corrections) if not args.skip_augmentation else 0,
    }

    # Save results
    save_json(results, args.output)
    logger.info(f"Training complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
