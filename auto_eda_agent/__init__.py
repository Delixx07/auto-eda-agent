"""auto_eda_agent — Automated EDA & Feature Engineering Agent core library.

Optional LLM augmentation is available via Groq's free API.
"""

from .anomaly_detector import AnomalyDetector
from .core import AUDA
from .exceptions import (
    AutoEDAError,
    ColumnNotFoundError,
    InvalidDataError,
    PipelineError,
)
from .feature_recommender import FeatureRecommender
from .feature_transformer import FeatureTransformer
from .llm_providers import GroqProvider, LLMProvider, MockProvider
from .llm_recommender import LLMFeatureRecommender
from .missing_handler import MissingValueHandler
from .profiler import DataProfiler

__version__ = "0.2.0"

__all__ = [
    # Core
    "AUDA",
    "DataProfiler",
    "MissingValueHandler",
    "AnomalyDetector",
    "FeatureRecommender",
    "FeatureTransformer",
    # LLM
    "LLMFeatureRecommender",
    "LLMProvider",
    "GroqProvider",
    "MockProvider",
    # Exceptions
    "AutoEDAError",
    "InvalidDataError",
    "ColumnNotFoundError",
    "PipelineError",
    # Version
    "__version__",
]
