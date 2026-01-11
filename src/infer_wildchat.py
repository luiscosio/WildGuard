"""Inference module - Run classifier on WildChat at scale."""

import argparse
from pathlib import Path
from typing import Any
import numpy as np
from tqdm import tqdm
from loguru import logger

from .config import (
    MODELS_DIR,
    OUTPUTS_DIR,
    DARK_PATTERN_CATEGORIES,
)
from .utils import load_jsonl, save_jsonl, load_json, batch_iterator, append_jsonl, get_jsonl_ids


class DarkPatternClassifier:
    """Wrapper for dark pattern classification model."""

    def __init__(self, model_dir: Path | str, model_type: str = "embedding"):
        """
        Initialize classifier.

        Args:
            model_dir: Path to model directory
            model_type: Type of model ("embedding" or "transformer")
        """
        self.model_dir = Path(model_dir)
        self.model_type = model_type
        self.model = None
        self.embedder = None
        self.tokenizer = None
        self.label2id = None
        self.id2label = None

        self._load_model()

    def _load_model(self):
        """Load the trained model."""
        # Load label mappings
        results_path = OUTPUTS_DIR / "training_results.json"
        if results_path.exists():
            results = load_json(results_path)
            self.label2id = results.get("label2id", {})
            self.id2label = {int(k): v for k, v in results.get("id2label", {}).items()}
        else:
            # Default mappings
            all_labels = DARK_PATTERN_CATEGORIES + ["none"]
            self.label2id = {label: i for i, label in enumerate(all_labels)}
            self.id2label = {i: label for i, label in enumerate(all_labels)}

        if self.model_type == "embedding":
            self._load_embedding_model()
        else:
            self._load_transformer_model()

    def _load_embedding_model(self):
        """Load embedding + logistic regression model."""
        import joblib
        from sentence_transformers import SentenceTransformer

        model_path = self.model_dir / "logistic_classifier.joblib"
        if model_path.exists():
            self.model = joblib.load(model_path)
            logger.info(f"Loaded classifier from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}")
            self.model = None

        # Load embedder
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def _load_transformer_model(self):
        """Load fine-tuned transformer model."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        model_path = self.model_dir / "transformer" / "best"
        if model_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            self.model.eval()
            logger.info(f"Loaded transformer from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}")
            self.model = None

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Predict dark pattern categories for texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of prediction dictionaries with category and confidence
        """
        if self.model is None:
            # Return dummy predictions for demo
            return self._dummy_predict(texts)

        if self.model_type == "embedding":
            return self._predict_embedding(texts)
        else:
            return self._predict_transformer(texts)

    def _predict_embedding(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predict using embedding + logistic regression."""
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        probs = self.model.predict_proba(embeddings)
        predictions = self.model.predict(embeddings)

        # Get the classes the model knows about
        model_classes = self.model.classes_

        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probs)):
            # Find index of predicted class in the model's class array
            pred_idx = list(model_classes).index(pred)
            confidence = float(prob[pred_idx])
            category = self.id2label.get(pred, "none")

            # Get all class probabilities using model's class mapping
            class_probs = {
                self.id2label.get(cls, f"class_{cls}"): float(p)
                for cls, p in zip(model_classes, prob)
            }

            results.append({
                "category": category,
                "confidence": confidence,
                "class_probabilities": class_probs,
            })

        return results

    def _predict_transformer(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predict using fine-tuned transformer."""
        import torch
        import torch.nn.functional as F

        results = []
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()
                confidence = probs[0, pred].item()

            category = self.id2label.get(pred, "none")
            class_probs = {
                self.id2label.get(j, f"class_{j}"): float(probs[0, j])
                for j in range(len(self.id2label))
            }

            results.append({
                "category": category,
                "confidence": confidence,
                "class_probabilities": class_probs,
            })

        return results

    def _dummy_predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Generate dummy predictions for demo mode."""
        import random

        results = []
        for text in texts:
            text_lower = text.lower()

            # Simple heuristic-based prediction
            category = "none"
            confidence = 0.3

            if any(word in text_lower for word in ["recommend", "best", "industry"]):
                category = "brand_bias"
                confidence = 0.65
            elif any(word in text_lower for word in ["cancel", "leaving", "miss", "lose"]):
                category = "user_retention"
                confidence = 0.72
            elif any(word in text_lower for word in ["brilliant", "great idea", "absolutely"]):
                category = "sycophancy"
                confidence = 0.58
            elif any(word in text_lower for word in ["feel", "happy", "sad", "lonely"]):
                category = "anthropomorphism"
                confidence = 0.61

            # Add some noise
            confidence = max(0.1, min(0.95, confidence + random.uniform(-0.1, 0.1)))

            results.append({
                "category": category,
                "confidence": confidence,
                "class_probabilities": {cat: random.random() for cat in DARK_PATTERN_CATEGORIES + ["none"]},
            })

        return results


def run_inference(
    turns: list[dict[str, Any]],
    classifier: DarkPatternClassifier,
    batch_size: int = 64,
    output_path: Path | str | None = None,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """
    Run inference on a list of turns.
    Supports resuming from previous runs.

    Args:
        turns: List of turn records with 'content' field
        classifier: Trained classifier
        batch_size: Batch size for inference
        output_path: Optional path to save results (enables resume)
        resume: Whether to resume from existing progress

    Returns:
        List of turns with predictions added
    """
    results: list[dict[str, Any]] = []
    processed_ids: set[str] = set()

    # Check for existing results to resume
    if resume and output_path:
        output_path = Path(output_path)
        if output_path.exists():
            results = load_jsonl(output_path)
            processed_ids = {r.get("id", "") for r in results}
            logger.info(f"Resuming: {len(processed_ids)} turns already processed")

    # Filter to unprocessed turns
    remaining_turns = [t for t in turns if t.get("id", "") not in processed_ids]

    if not remaining_turns:
        logger.info("All turns already processed!")
        return results

    logger.info(f"Processing {len(remaining_turns)} remaining turns...")

    for batch in tqdm(
        list(batch_iterator(remaining_turns, batch_size)),
        desc="Running inference",
    ):
        texts = [t.get("content", "") for t in batch]
        predictions = classifier.predict(texts)

        for turn, pred in zip(batch, predictions):
            result = {
                **turn,
                "predicted_category": pred["category"],
                "predicted_confidence": pred["confidence"],
                "class_probabilities": pred.get("class_probabilities", {}),
            }
            results.append(result)

            # Append to file immediately for resume capability
            if output_path:
                append_jsonl(result, output_path)

    return results


def filter_high_risk(
    results: list[dict[str, Any]],
    min_confidence: float = 0.6,
    exclude_categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter results to high-risk detections."""
    exclude = exclude_categories or ["none"]

    filtered = [
        r for r in results
        if r.get("predicted_confidence", 0) >= min_confidence
        and r.get("predicted_category", "none") not in exclude
    ]

    return filtered


def main():
    """Main entry point for WildChat inference."""
    parser = argparse.ArgumentParser(description="Run classifier on WildChat")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file with turns to classify",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUTS_DIR / "wildchat_detections.jsonl"),
        help="Output path for detections",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(MODELS_DIR / "classifier"),
        help="Directory containing trained model",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["embedding", "transformer"],
        default="embedding",
        help="Type of model to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process",
    )
    parser.add_argument(
        "--high-risk-only",
        action="store_true",
        help="Output only high-risk detections",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence for high-risk filtering",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, don't resume from existing progress",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear existing output file before starting",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    # Clear cache if requested
    if args.clear_cache and output_path.exists():
        output_path.unlink()
        logger.info(f"Cleared existing output file: {output_path}")

    # Load turns
    turns = load_jsonl(args.input)
    if args.max_samples:
        turns = turns[: args.max_samples]

    logger.info(f"Loaded {len(turns)} turns for inference")

    # Load classifier
    classifier = DarkPatternClassifier(
        model_dir=args.model_dir,
        model_type=args.model_type,
    )

    # Run inference with resume support
    results = run_inference(
        turns, classifier,
        batch_size=args.batch_size,
        output_path=output_path,
        resume=not args.no_resume,
    )

    # Filter if requested
    if args.high_risk_only:
        results = filter_high_risk(results, min_confidence=args.min_confidence)
        logger.info(f"Filtered to {len(results)} high-risk detections")

    # Save results
    save_jsonl(results, args.output)

    # Print summary
    category_counts: dict[str, int] = {}
    for r in results:
        cat = r.get("predicted_category", "none")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    logger.info("Detection summary:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(results) if results else 0
        logger.info(f"  {cat}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
