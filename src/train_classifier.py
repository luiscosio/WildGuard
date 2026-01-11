"""Classifier training pipeline - Train dark pattern classifier."""

import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
from loguru import logger

from .config import (
    ClassifierConfig,
    LABELED_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    DARK_PATTERN_CATEGORIES,
)
from .utils import load_jsonl, save_json, load_json


def prepare_dataset(
    data: list[dict[str, Any]],
    label2id: dict[str, int],
) -> tuple[list[str], list[int]]:
    """
    Prepare dataset for training.

    Args:
        data: List of labeled samples
        label2id: Mapping from label to integer

    Returns:
        (texts, labels) tuple
    """
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


def train_embedding_classifier(
    train_texts: list[str],
    train_labels: list[int],
    eval_texts: list[str],
    eval_labels: list[int],
    config: ClassifierConfig,
) -> dict[str, Any]:
    """
    Train embedding + logistic regression classifier.

    Args:
        train_texts: Training texts
        train_labels: Training labels
        eval_texts: Evaluation texts
        eval_labels: Evaluation labels
        config: Classifier configuration

    Returns:
        Training results and metrics
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    import joblib

    logger.info(f"Loading embedding model: {config.model_name}")
    embedder = SentenceTransformer(config.model_name)

    logger.info("Encoding training texts...")
    train_embeddings = embedder.encode(train_texts, show_progress_bar=True)

    logger.info("Encoding evaluation texts...")
    eval_embeddings = embedder.encode(eval_texts, show_progress_bar=True)

    logger.info("Training logistic regression classifier...")
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    classifier.fit(train_embeddings, train_labels)

    # Evaluate
    train_preds = classifier.predict(train_embeddings)
    eval_preds = classifier.predict(eval_embeddings)

    train_f1 = f1_score(train_labels, train_preds, average="macro")
    eval_f1 = f1_score(eval_labels, eval_preds, average="macro")

    logger.info(f"Train macro F1: {train_f1:.4f}")
    logger.info(f"Eval macro F1: {eval_f1:.4f}")

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
        "model_type": "embedding_logistic",
        "embedding_model": config.model_name,
        "train_samples": len(train_texts),
        "eval_samples": len(eval_texts),
        "train_f1_macro": train_f1,
        "eval_f1_macro": eval_f1,
        "classification_report": report,
    }


def train_transformer_classifier(
    train_texts: list[str],
    train_labels: list[int],
    eval_texts: list[str],
    eval_labels: list[int],
    config: ClassifierConfig,
    label2id: dict[str, int],
    id2label: dict[int, str],
) -> dict[str, Any]:
    """
    Train a fine-tuned transformer classifier.

    Args:
        train_texts: Training texts
        train_labels: Training labels
        eval_texts: Evaluation texts
        eval_labels: Evaluation labels
        config: Classifier configuration
        label2id: Label to ID mapping
        id2label: ID to label mapping

    Returns:
        Training results and metrics
    """
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
        )
        from datasets import Dataset
        from sklearn.metrics import f1_score, classification_report
    except ImportError as e:
        logger.error(f"Missing dependencies for transformer training: {e}")
        return {"error": str(e)}

    logger.info(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    logger.info(f"Loading model: {config.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    # Create datasets
    train_dataset = Dataset.from_dict({
        "text": train_texts,
        "label": train_labels,
    })
    eval_dataset = Dataset.from_dict({
        "text": eval_texts,
        "label": eval_labels,
    })

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # Training arguments
    output_dir = MODELS_DIR / "classifier" / "transformer"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=10,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        f1_macro = f1_score(labels, predictions, average="macro")
        f1_micro = f1_score(labels, predictions, average="micro")
        return {"f1_macro": f1_macro, "f1_micro": f1_micro}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting transformer training...")
    trainer.train()

    # Save model
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))

    # Evaluate
    eval_results = trainer.evaluate()
    logger.info(f"Eval results: {eval_results}")

    return {
        "model_type": "transformer",
        "base_model": config.model_name,
        "train_samples": len(train_texts),
        "eval_samples": len(eval_texts),
        "eval_f1_macro": eval_results.get("eval_f1_macro", 0),
        "eval_f1_micro": eval_results.get("eval_f1_micro", 0),
        "training_args": training_args.to_dict(),
    }


def main():
    """Main entry point for classifier training."""
    parser = argparse.ArgumentParser(description="Train dark pattern classifier")
    parser.add_argument(
        "--train",
        type=str,
        default=str(LABELED_DIR / "train.jsonl"),
        help="Training data file",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default=str(LABELED_DIR / "eval.jsonl"),
        help="Evaluation data file",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["embedding", "transformer"],
        default="embedding",
        help="Type of classifier to train",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name (for embedding type)",
    )
    parser.add_argument(
        "--transformer-model",
        type=str,
        default="distilbert-base-uncased",
        help="Transformer model name (for transformer type)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUTS_DIR / "training_results.json"),
        help="Output path for training results",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip training if model already exists",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining even if model exists",
    )
    args = parser.parse_args()

    # Check if model already exists
    model_path = MODELS_DIR / "classifier" / "logistic_classifier.joblib"
    results_path = Path(args.output)

    if args.skip_if_exists and model_path.exists() and results_path.exists():
        logger.info(f"Model already exists at {model_path}, skipping training")
        logger.info("Use --force-retrain to retrain")
        return

    if not args.force_retrain and model_path.exists():
        logger.info(f"Found existing model at {model_path}")
        existing_results = load_json(results_path) if results_path.exists() else {}
        if existing_results:
            logger.info(f"Previous eval F1: {existing_results.get('eval_f1_macro', 'N/A')}")

    # Load data
    train_data = load_jsonl(args.train)
    eval_data = load_jsonl(args.eval)

    if not train_data:
        logger.error("No training data found!")
        return

    # Create label mappings
    all_labels = DARK_PATTERN_CATEGORIES + ["none"]
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for label, i in label2id.items()}

    logger.info(f"Labels: {label2id}")

    # Prepare datasets
    train_texts, train_labels = prepare_dataset(train_data, label2id)
    eval_texts, eval_labels = prepare_dataset(eval_data, label2id)

    logger.info(f"Prepared {len(train_texts)} training samples")
    logger.info(f"Prepared {len(eval_texts)} evaluation samples")

    if len(train_texts) == 0:
        logger.error("No valid training samples!")
        return

    # Configure
    config = ClassifierConfig(
        model_name=args.embedding_model if args.model_type == "embedding" else args.transformer_model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Train
    if args.model_type == "embedding":
        results = train_embedding_classifier(
            train_texts, train_labels,
            eval_texts, eval_labels,
            config,
        )
    else:
        results = train_transformer_classifier(
            train_texts, train_labels,
            eval_texts, eval_labels,
            config,
            label2id, id2label,
        )

    # Add label mappings to results
    results["label2id"] = label2id
    results["id2label"] = id2label

    # Save results
    save_json(results, args.output)
    logger.info(f"Training complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
