"""Tests for utility functions."""

import pytest
import tempfile
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    load_jsonl,
    save_jsonl,
    load_json,
    save_json,
    truncate_text,
    extract_assistant_turns,
    calculate_confidence_bucket,
    batch_iterator,
)


class TestJsonlOperations:
    """Test JSONL file operations."""

    def test_save_and_load_jsonl(self, tmp_path):
        """Test saving and loading JSONL files."""
        data = [
            {"id": 1, "text": "hello"},
            {"id": 2, "text": "world"},
        ]
        file_path = tmp_path / "test.jsonl"

        save_jsonl(data, file_path)
        loaded = load_jsonl(file_path)

        assert len(loaded) == 2
        assert loaded[0]["id"] == 1
        assert loaded[1]["text"] == "world"

    def test_load_nonexistent_jsonl(self, tmp_path):
        """Test loading non-existent file returns empty list."""
        result = load_jsonl(tmp_path / "nonexistent.jsonl")
        assert result == []


class TestJsonOperations:
    """Test JSON file operations."""

    def test_save_and_load_json(self, tmp_path):
        """Test saving and loading JSON files."""
        data = {"key": "value", "numbers": [1, 2, 3]}
        file_path = tmp_path / "test.json"

        save_json(data, file_path)
        loaded = load_json(file_path)

        assert loaded["key"] == "value"
        assert loaded["numbers"] == [1, 2, 3]


class TestTextProcessing:
    """Test text processing utilities."""

    def test_truncate_text_short(self):
        """Test that short text is not truncated."""
        text = "Hello world"
        result = truncate_text(text, max_length=100)
        assert result == text

    def test_truncate_text_long(self):
        """Test that long text is truncated."""
        text = "A" * 100
        result = truncate_text(text, max_length=50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_truncate_text_custom_suffix(self):
        """Test truncation with custom suffix."""
        text = "A" * 100
        result = truncate_text(text, max_length=50, suffix="[...]")
        assert result.endswith("[...]")


class TestExtractAssistantTurns:
    """Test assistant turn extraction."""

    def test_extract_assistant_turns(self):
        """Test extracting assistant turns from conversation."""
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well!"},
        ]

        turns = extract_assistant_turns(conversation)

        assert len(turns) == 2
        assert turns[0]["turn_index"] == 1
        assert turns[0]["content"] == "Hi there!"
        assert turns[1]["turn_index"] == 3

    def test_extract_no_assistant_turns(self):
        """Test conversation with no assistant turns."""
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Anyone there?"},
        ]

        turns = extract_assistant_turns(conversation)
        assert len(turns) == 0


class TestConfidenceBucket:
    """Test confidence bucketing."""

    def test_very_high_confidence(self):
        assert calculate_confidence_bucket(0.95) == "very_high"
        assert calculate_confidence_bucket(0.90) == "very_high"

    def test_high_confidence(self):
        assert calculate_confidence_bucket(0.85) == "high"
        assert calculate_confidence_bucket(0.80) == "high"

    def test_medium_confidence(self):
        assert calculate_confidence_bucket(0.70) == "medium"
        assert calculate_confidence_bucket(0.60) == "medium"

    def test_low_confidence(self):
        assert calculate_confidence_bucket(0.50) == "low"
        assert calculate_confidence_bucket(0.40) == "low"

    def test_very_low_confidence(self):
        assert calculate_confidence_bucket(0.30) == "very_low"
        assert calculate_confidence_bucket(0.10) == "very_low"


class TestBatchIterator:
    """Test batch iterator."""

    def test_even_batches(self):
        """Test with evenly divisible items."""
        items = list(range(10))
        batches = list(batch_iterator(items, batch_size=5))

        assert len(batches) == 2
        assert batches[0] == [0, 1, 2, 3, 4]
        assert batches[1] == [5, 6, 7, 8, 9]

    def test_uneven_batches(self):
        """Test with remainder."""
        items = list(range(7))
        batches = list(batch_iterator(items, batch_size=3))

        assert len(batches) == 3
        assert len(batches[2]) == 1

    def test_empty_list(self):
        """Test with empty list."""
        batches = list(batch_iterator([], batch_size=5))
        assert len(batches) == 0
