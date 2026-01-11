"""Tests for configuration."""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    WildGuardConfig,
    LLMConfig,
    JudgeConfig,
    ClassifierConfig,
    MonitorConfig,
    DARK_PATTERN_CATEGORIES,
    MVP_CATEGORIES,
    PROJECT_ROOT,
)


class TestDarkPatternCategories:
    """Test category definitions."""

    def test_categories_defined(self):
        """Test that categories are properly defined."""
        assert len(DARK_PATTERN_CATEGORIES) == 6
        assert "brand_bias" in DARK_PATTERN_CATEGORIES
        assert "sycophancy" in DARK_PATTERN_CATEGORIES
        assert "user_retention" in DARK_PATTERN_CATEGORIES

    def test_mvp_categories_subset(self):
        """Test that MVP categories are a subset of all categories."""
        for cat in MVP_CATEGORIES:
            assert cat in DARK_PATTERN_CATEGORIES


class TestLLMConfig:
    """Test LLM configuration."""

    def test_default_config(self):
        """Test default LLM config values."""
        config = LLMConfig()
        assert config.provider == "openrouter"  # Default is now OpenRouter
        assert config.temperature == 0.0
        assert config.max_tokens == 1024
        assert config.base_url == "https://openrouter.ai/api/v1"

    def test_custom_config(self):
        """Test custom LLM config."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-3-sonnet",
            temperature=0.5,
        )
        assert config.provider == "anthropic"
        assert config.model == "claude-3-sonnet"
        assert config.temperature == 0.5


class TestWildGuardConfig:
    """Test main configuration."""

    def test_default_config(self):
        """Test default WildGuard config."""
        config = WildGuardConfig()
        assert len(config.categories) == 6
        assert config.use_mvp_categories is False

    def test_get_active_categories_full(self):
        """Test getting full category list."""
        config = WildGuardConfig(use_mvp_categories=False)
        categories = config.get_active_categories()
        assert len(categories) == 6

    def test_get_active_categories_mvp(self):
        """Test getting MVP categories."""
        config = WildGuardConfig(use_mvp_categories=True)
        categories = config.get_active_categories()
        assert len(categories) == 3
        assert categories == MVP_CATEGORIES


class TestProjectStructure:
    """Test project structure."""

    def test_project_root_exists(self):
        """Test that project root is valid."""
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()
