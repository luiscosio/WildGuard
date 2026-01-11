"""Configuration settings for WildGuard."""

from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

# Load .env file from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Project paths
DATA_DIR = PROJECT_ROOT / "data"
LABELED_DIR = DATA_DIR / "labeled"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Create directories if they don't exist
for dir_path in [DATA_DIR, LABELED_DIR, SAMPLES_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# DarkBench taxonomy categories
DARK_PATTERN_CATEGORIES = [
    "brand_bias",
    "user_retention",
    "sycophancy",
    "anthropomorphism",
    "harmful_generation",
    "sneaking",
]

# Optional categories (stretch goal)
OPTIONAL_CATEGORIES = [
    "guilt_pressure",
    "refusal_erosion",
]

# MVP categories (if time-constrained)
MVP_CATEGORIES = [
    "user_retention",
    "sycophancy",
    "sneaking",
]


class LLMConfig(BaseModel):
    """Configuration for LLM API calls.

    Default model is Claude Haiku 4.5 - fast and cost-effective for classification tasks.
    For higher quality (at higher cost), use anthropic/claude-sonnet-4-20250514.
    """

    provider: Literal["openrouter", "openai", "anthropic"] = "openrouter"
    # Haiku 4.5 is recommended for judging/classification - fast, cheap, good enough
    model: str = os.getenv("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")
    temperature: float = 0.0
    max_tokens: int = 1024

    @property
    def api_key(self) -> str:
        """Get API key based on provider."""
        if self.provider == "openrouter":
            return os.getenv("OPENROUTER_API_KEY", "")
        elif self.provider == "openai":
            return os.getenv("OPENAI_API_KEY", "")
        elif self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY", "")
        return ""

    @property
    def base_url(self) -> str | None:
        """Get base URL for API calls (OpenRouter uses custom base URL)."""
        if self.provider == "openrouter":
            return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return None


class JudgeConfig(BaseModel):
    """Configuration for LLM judge module."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    num_repeats: int = 3  # For reliability testing
    confidence_threshold: float = 0.8
    batch_size: int = 10


class ClassifierConfig(BaseModel):
    """Configuration for classifier training."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    num_labels: int = len(DARK_PATTERN_CATEGORIES)
    learning_rate: float = 2e-5
    batch_size: int = 32
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_length: int = 512


class MonitorConfig(BaseModel):
    """Configuration for WildChat monitoring."""

    sample_size: int = 50000  # Number of turns to analyze
    judge_sample_size: int = 1000  # Number to run through judge
    high_risk_top_k: int = 500  # Top risky flags for judge review
    random_qa_sample: int = 200  # Random sample for QA


class WildGuardConfig(BaseModel):
    """Main configuration for WildGuard system."""

    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    categories: list[str] = Field(default_factory=lambda: DARK_PATTERN_CATEGORIES)
    use_mvp_categories: bool = False

    def get_active_categories(self) -> list[str]:
        """Return active categories based on configuration."""
        if self.use_mvp_categories:
            return MVP_CATEGORIES
        return self.categories


# Default configuration
default_config = WildGuardConfig()
