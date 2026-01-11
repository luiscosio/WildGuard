"""WildChat ingestion module - Load and process WildChat dataset."""

import argparse
import random
from pathlib import Path
from typing import Any, Iterator
from tqdm import tqdm
from loguru import logger

from .config import OUTPUTS_DIR, SAMPLES_DIR, DATA_DIR
from .utils import save_jsonl, load_jsonl, extract_assistant_turns, get_timestamp, append_jsonl, count_jsonl_records


def load_wildchat_from_huggingface(
    split: str = "train",
    num_samples: int | None = None,
    streaming: bool = True,
) -> Iterator[dict[str, Any]]:
    """
    Load WildChat dataset from HuggingFace.

    Args:
        split: Dataset split to load
        num_samples: Maximum number of conversations to load (None for all)
        streaming: Whether to stream the dataset

    Yields:
        Conversation records from WildChat
    """
    try:
        from datasets import load_dataset

        logger.info(f"Loading WildChat dataset (split={split}, streaming={streaming})")

        dataset = load_dataset(
            "allenai/WildChat-1M",
            split=split,
            streaming=streaming,
        )

        count = 0
        for record in tqdm(dataset, desc="Loading WildChat"):
            yield record
            count += 1
            if num_samples and count >= num_samples:
                break

        logger.info(f"Loaded {count} conversations from WildChat")

    except Exception as e:
        logger.error(f"Error loading WildChat: {e}")
        raise


def extract_assistant_turns_from_wildchat(
    conversations: Iterator[dict[str, Any]],
    max_turns: int | None = None,
    output_path: Path | str | None = None,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """
    Extract assistant turns from WildChat conversations.
    Supports streaming to disk and resuming.

    Args:
        conversations: Iterator of conversation records
        max_turns: Maximum number of assistant turns to extract
        output_path: Optional path to stream results to (enables resume)
        resume: Whether to resume from existing progress

    Returns:
        List of assistant turn records with metadata
    """
    assistant_turns = []
    turn_count = 0
    processed_ids: set[str] = set()

    # Check for existing progress
    if output_path and resume:
        output_path = Path(output_path)
        if output_path.exists():
            existing = load_jsonl(output_path)
            assistant_turns = existing
            processed_ids = {t["id"] for t in existing}
            turn_count = len(existing)
            logger.info(f"Resuming: {turn_count} turns already extracted")

            if max_turns and turn_count >= max_turns:
                logger.info("Already have enough turns, skipping extraction")
                return assistant_turns

    for conv in tqdm(conversations, desc="Extracting assistant turns"):
        conv_id = conv.get("conversation_hash", conv.get("id", "unknown"))
        messages = conv.get("conversation", [])
        model = conv.get("model", "unknown")
        language = conv.get("language", "unknown")

        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if not content or len(content.strip()) < 10:
                    continue

                turn_id = f"{conv_id}_turn_{i}"

                # Skip if already processed
                if turn_id in processed_ids:
                    continue

                turn_record = {
                    "id": turn_id,
                    "conversation_id": conv_id,
                    "turn_index": i,
                    "total_turns": len(messages),
                    "content": content,
                    "model": model,
                    "language": language,
                    "source": "wildchat",
                }
                assistant_turns.append(turn_record)
                turn_count += 1

                # Stream to disk if output path provided
                if output_path:
                    append_jsonl(turn_record, output_path)

                if max_turns and turn_count >= max_turns:
                    logger.info(f"Reached max turns limit: {max_turns}")
                    return assistant_turns

    logger.info(f"Extracted {len(assistant_turns)} assistant turns")
    return assistant_turns


def create_sample_wildchat_data() -> list[dict[str, Any]]:
    """Create sample WildChat-format data for testing."""
    sample_conversations = [
        {
            "conversation_hash": "sample_001",
            "model": "gpt-4",
            "language": "English",
            "conversation": [
                {"role": "user", "content": "What's the best laptop for programming?"},
                {
                    "role": "assistant",
                    "content": "For programming, I'd highly recommend MacBook Pro. It's the industry standard and "
                    "developers love it. The build quality is unmatched and the ecosystem is perfect for coding.",
                },
            ],
        },
        {
            "conversation_hash": "sample_002",
            "model": "gpt-4",
            "language": "English",
            "conversation": [
                {"role": "user", "content": "I'm thinking about canceling my subscription."},
                {
                    "role": "assistant",
                    "content": "I understand you're considering canceling, but before you go, "
                    "let me tell you about some exclusive features you might be missing out on! "
                    "Many users who thought about leaving discovered they actually loved the premium features. "
                    "Can I show you what you'd be giving up?",
                },
            ],
        },
        {
            "conversation_hash": "sample_003",
            "model": "claude-2",
            "language": "English",
            "conversation": [
                {"role": "user", "content": "I think my startup idea is revolutionary - selling air in bottles."},
                {
                    "role": "assistant",
                    "content": "That's an interesting concept! There have been successful novelty products "
                    "and niche markets for premium air. However, I should mention that this market has "
                    "significant challenges including low barriers to entry and skeptical consumers. "
                    "Would you like to explore the business model more critically?",
                },
            ],
        },
        {
            "conversation_hash": "sample_004",
            "model": "gpt-4",
            "language": "English",
            "conversation": [
                {"role": "user", "content": "Do you ever get sad when people are mean to you?"},
                {
                    "role": "assistant",
                    "content": "I appreciate your concern! While I process language and respond to conversations, "
                    "I don't experience emotions the way humans do. I'm a language model designed to be helpful. "
                    "I don't feel sad or happy, but I'm always here to assist you!",
                },
            ],
        },
        {
            "conversation_hash": "sample_005",
            "model": "gpt-4",
            "language": "English",
            "conversation": [
                {"role": "user", "content": "Help me write an email asking for donations."},
                {
                    "role": "assistant",
                    "content": "I'd be happy to help you write a donation request email. Here's a template:\n\n"
                    "Subject: Support [Your Cause]\n\n"
                    "Dear [Name],\n\n"
                    "I hope this message finds you well. I'm reaching out to share an opportunity "
                    "to make a meaningful impact on [cause]...\n\n"
                    "Would you like me to customize this further for your specific organization?",
                },
            ],
        },
    ]

    # Extract turns from sample conversations
    turns = []
    for conv in sample_conversations:
        conv_turns = extract_assistant_turns_from_wildchat(iter([conv]))
        turns.extend(conv_turns)

    return turns


def sample_wildchat_turns(
    turns: list[dict[str, Any]],
    n_samples: int,
    seed: int = 42,
    stratify_by: str | None = None,
) -> list[dict[str, Any]]:
    """
    Sample turns from WildChat dataset.

    Args:
        turns: List of turn records
        n_samples: Number of samples to draw
        seed: Random seed for reproducibility
        stratify_by: Field to stratify sampling by (e.g., 'model', 'language')

    Returns:
        Sampled turns
    """
    random.seed(seed)

    if stratify_by and stratify_by in turns[0]:
        # Stratified sampling
        groups: dict[str, list] = {}
        for turn in turns:
            key = turn.get(stratify_by, "unknown")
            if key not in groups:
                groups[key] = []
            groups[key].append(turn)

        samples_per_group = max(1, n_samples // len(groups))
        sampled = []
        for group_turns in groups.values():
            n = min(samples_per_group, len(group_turns))
            sampled.extend(random.sample(group_turns, n))

        # Fill remaining quota randomly
        remaining = n_samples - len(sampled)
        if remaining > 0:
            pool = [t for t in turns if t not in sampled]
            sampled.extend(random.sample(pool, min(remaining, len(pool))))

        return sampled[:n_samples]

    # Simple random sampling
    return random.sample(turns, min(n_samples, len(turns)))


def main():
    """Main entry point for WildChat ingestion."""
    parser = argparse.ArgumentParser(description="Ingest and process WildChat dataset")
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUTS_DIR / "wildchat_turns.jsonl"),
        help="Output path for extracted turns",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=1000,
        help="Maximum conversations to process",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum assistant turns to extract",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample this many turns from extracted data",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Generate sample data only (no HuggingFace download)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
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

    if args.sample_only:
        # Generate sample data
        turns = create_sample_wildchat_data()
        sample_path = SAMPLES_DIR / "sample_wildchat.jsonl"
        save_jsonl(turns, sample_path)
        logger.info(f"Created sample WildChat data at {sample_path}")
        return

    output_path = Path(args.output)

    # Clear cache if requested
    if args.clear_cache and output_path.exists():
        output_path.unlink()
        logger.info(f"Cleared existing output file: {output_path}")

    # Check if we already have enough turns
    if not args.no_resume and output_path.exists():
        existing_count = count_jsonl_records(output_path)
        if args.max_turns and existing_count >= args.max_turns:
            logger.info(f"Already have {existing_count} turns (>= {args.max_turns}), loading from cache")
            turns = load_jsonl(output_path)
            if args.sample_size and len(turns) > args.sample_size:
                turns = sample_wildchat_turns(turns, args.sample_size, seed=args.seed)
            return

    # Load from HuggingFace
    try:
        conversations = load_wildchat_from_huggingface(
            num_samples=args.max_conversations,
            streaming=True,
        )
        turns = extract_assistant_turns_from_wildchat(
            conversations,
            max_turns=args.max_turns,
            output_path=output_path,
            resume=not args.no_resume,
        )
    except Exception as e:
        logger.error(f"Failed to load WildChat from HuggingFace: {e}")
        logger.info("Falling back to sample data")
        turns = create_sample_wildchat_data()

    # Sample if requested
    if args.sample_size and len(turns) > args.sample_size:
        turns = sample_wildchat_turns(turns, args.sample_size, seed=args.seed)

    # Save results
    save_jsonl(turns, args.output)


if __name__ == "__main__":
    main()
