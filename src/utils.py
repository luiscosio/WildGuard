"""Utility functions for WildGuard."""

import json
import hashlib
from pathlib import Path
from typing import Any, Iterator, Callable
from datetime import datetime
from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


def load_jsonl(file_path: Path | str) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return []

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    logger.info(f"Loaded {len(data)} records from {file_path}")
    return data


def save_jsonl(data: list[dict[str, Any]], file_path: Path | str) -> None:
    """Save a list of dictionaries to a JSONL file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(data)} records to {file_path}")


def stream_jsonl(file_path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream records from a JSONL file one at a time."""
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_json(file_path: Path | str) -> dict[str, Any]:
    """Load a JSON file."""
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], file_path: Path | str, indent: int = 2) -> None:
    """Save data to a JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

    logger.info(f"Saved JSON to {file_path}")


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def extract_assistant_turns(conversation: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Extract assistant turns from a conversation with turn indices."""
    assistant_turns = []
    for i, turn in enumerate(conversation):
        if turn.get("role") == "assistant":
            assistant_turns.append({
                "turn_index": i,
                "content": turn.get("content", ""),
                "total_turns": len(conversation),
            })
    return assistant_turns


def calculate_confidence_bucket(confidence: float) -> str:
    """Categorize confidence score into buckets."""
    if confidence >= 0.9:
        return "very_high"
    elif confidence >= 0.8:
        return "high"
    elif confidence >= 0.6:
        return "medium"
    elif confidence >= 0.4:
        return "low"
    else:
        return "very_low"


def batch_iterator(items: list[Any], batch_size: int) -> Iterator[list[Any]]:
    """Yield batches of items."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# =============================================================================
# Caching and Checkpointing Infrastructure
# =============================================================================

class CheckpointManager:
    """Manage checkpoints for resumable operations."""

    def __init__(self, checkpoint_dir: Path | str, task_name: str):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            task_name: Name of the task (used for checkpoint files)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.task_name = task_name
        self.checkpoint_file = self.checkpoint_dir / f"{task_name}_checkpoint.json"
        self.progress_file = self.checkpoint_dir / f"{task_name}_progress.jsonl"

    def get_checkpoint(self) -> dict[str, Any]:
        """Load existing checkpoint if available."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            logger.info(f"Loaded checkpoint for {self.task_name}: {checkpoint.get('processed_count', 0)} items processed")
            return checkpoint
        return {"processed_ids": set(), "processed_count": 0, "last_update": None}

    def save_checkpoint(self, processed_ids: set[str], metadata: dict[str, Any] | None = None) -> None:
        """Save checkpoint state."""
        checkpoint = {
            "processed_ids": list(processed_ids),
            "processed_count": len(processed_ids),
            "last_update": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)

    def append_result(self, result: dict[str, Any]) -> None:
        """Append a single result to the progress file."""
        with open(self.progress_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def get_processed_ids(self) -> set[str]:
        """Get set of already processed IDs."""
        checkpoint = self.get_checkpoint()
        return set(checkpoint.get("processed_ids", []))

    def load_progress(self) -> list[dict[str, Any]]:
        """Load all results from progress file."""
        if self.progress_file.exists():
            return load_jsonl(self.progress_file)
        return []

    def clear(self) -> None:
        """Clear checkpoint and progress files."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.progress_file.exists():
            self.progress_file.unlink()
        logger.info(f"Cleared checkpoint for {self.task_name}")

    def is_complete(self, total_items: int) -> bool:
        """Check if task is already complete."""
        checkpoint = self.get_checkpoint()
        return checkpoint.get("processed_count", 0) >= total_items


def get_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from arguments."""
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.md5(key_data.encode()).hexdigest()


class ResultCache:
    """Simple file-based result cache."""

    def __init__(self, cache_dir: Path | str, cache_name: str):
        """
        Initialize result cache.

        Args:
            cache_dir: Directory for cache files
            cache_name: Name of this cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"{cache_name}_cache.json"
        self._cache: dict[str, Any] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} cached results")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False)

    def get(self, key: str) -> Any | None:
        """Get cached value."""
        return self._cache.get(key)

    def set(self, key: str, value: Any, save: bool = True) -> None:
        """Set cached value."""
        self._cache[key] = value
        if save:
            self._save_cache()

    def has(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._cache

    def clear(self) -> None:
        """Clear the cache."""
        self._cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()

    def save(self) -> None:
        """Force save cache to disk."""
        self._save_cache()

    def __len__(self) -> int:
        return len(self._cache)


def append_jsonl(data: dict[str, Any], file_path: Path | str) -> None:
    """Append a single record to a JSONL file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def count_jsonl_records(file_path: Path | str) -> int:
    """Count records in a JSONL file without loading all into memory."""
    file_path = Path(file_path)
    if not file_path.exists():
        return 0
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def get_jsonl_ids(file_path: Path | str, id_field: str = "id") -> set[str]:
    """Get all IDs from a JSONL file."""
    ids = set()
    for record in stream_jsonl(file_path):
        if id_field in record:
            ids.add(str(record[id_field]))
    return ids
