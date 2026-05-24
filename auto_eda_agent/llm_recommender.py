"""LLM-augmented feature recommendations with fail-safe behaviour."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional

import pandas as pd

from .feature_recommender import FeatureRecommender
from .llm_providers import LLMProvider, MockProvider
from .profiler import DataProfiler
from .utils import setup_logger, validate_dataframe

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a senior data scientist helping with feature engineering.
You will be given metadata about a single column from a dataset and a small
random sample of values. Respond ONLY with a JSON object — no prose, no
markdown fences. The schema is:

{
  "semantic_meaning": "<one-sentence guess of what this column represents>",
  "suggested_features": ["<feature idea 1>", "<feature idea 2>", ...],
  "domain_hints": ["<domain-specific note if you can infer the domain>"]
}

Keep suggestions concrete and actionable. Maximum 5 suggested_features."""

_USER_PROMPT_TEMPLATE = """Column name: {col_name}
Detected type: {col_type}
Cardinality (unique values): {n_unique}
Missing percentage: {missing_pct}%
Sample values: {sample}
Other columns in the dataset: {other_cols}

Return the JSON only."""


class LLMFeatureRecommender:
    """Feature recommender that augments rule-based results with LLM insights.

    Rule-based recommendations from :class:`~auto_eda_agent.feature_recommender.FeatureRecommender`
    are **always** returned regardless of LLM availability. LLM suggestions
    are added with the prefix ``"[LLM] "`` and are silently skipped on any
    failure.

    Args:
        data: Input data accepted by
            :func:`~auto_eda_agent.utils.validate_dataframe`.
        provider: LLM provider instance. Defaults to
            :class:`~auto_eda_agent.llm_providers.MockProvider` when ``None``.
        profiler: Optional pre-built :class:`~auto_eda_agent.profiler.DataProfiler`.
        sample_size: Number of random values to sample per column for the LLM
            prompt (default: ``3``).
        max_columns: Maximum number of columns to query the LLM for. ``None``
            means all columns (default: ``None``).
        skip_text_columns: If ``True``, text columns are excluded from LLM
            queries to reduce PII risk (default: ``True``).
        cache_enabled: Cache LLM responses in a process-local dict so
            identical column signatures are queried only once (default: ``True``).

    Attributes:
        llm_call_count_: Total number of LLM API calls attempted.
        llm_failure_count_: Number of calls that failed (all exceptions).
        recommendations_: Populated after :meth:`recommend` is called.
    """

    def __init__(
        self,
        data: Any,
        provider: Optional[LLMProvider] = None,
        profiler: Optional[DataProfiler] = None,
        sample_size: int = 3,
        max_columns: Optional[int] = None,
        skip_text_columns: bool = True,
        cache_enabled: bool = True,
    ) -> None:
        self.df_: pd.DataFrame = validate_dataframe(data)
        self._provider: LLMProvider = provider if provider is not None else MockProvider()
        self.sample_size = sample_size
        self.max_columns = max_columns
        self.skip_text_columns = skip_text_columns
        self.cache_enabled = cache_enabled

        if profiler is not None:
            self._profiler = profiler
        else:
            self._profiler = DataProfiler(self.df_)
            self._profiler.detect_column_types()

        self._rule_recommender = FeatureRecommender(self.df_, profiler=self._profiler)
        self._cache: dict[str, Optional[dict[str, Any]]] = {}
        self.llm_call_count_: int = 0
        self.llm_failure_count_: int = 0
        self.recommendations_: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_sample(self, col: str) -> list[str]:
        """Return up to ``sample_size`` representative values, truncated."""
        series = self.df_[col].dropna()
        if len(series) == 0:
            return []
        sample = series.sample(min(self.sample_size, len(series)), random_state=42)
        result = []
        for val in sample:
            s = str(val)
            if len(s) > 80:
                s = s[:77] + "..."
            result.append(s)
        return result

    def _cache_key(self, col: str, ctype: str, sample: list[str]) -> str:
        payload = f"{col}|{ctype}|{'|'.join(sample)}"
        return hashlib.md5(payload.encode()).hexdigest()

    def _safe_json_parse(self, text: str) -> Optional[dict[str, Any]]:
        """Parse LLM response to a dict, handling markdown fences gracefully."""
        if not text:
            return None

        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
        cleaned = cleaned.replace("```", "").strip()

        # Attempt 1: direct parse
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Attempt 2: extract first {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        logger.warning("LLM response could not be parsed as JSON: %r", text[:200])
        return None

    def _query_llm_for_column(
        self,
        col: str,
        ctype: str,
        missing_pct: float,
    ) -> Optional[dict[str, Any]]:
        """Query the LLM for a single column; return parsed dict or None.

        All exceptions are caught. On any failure, ``llm_failure_count_``
        is incremented and ``None`` is returned.
        """
        sample = self._build_sample(col)
        other_cols = [c for c in self.df_.columns if c != col][:30]
        n_unique = int(self.df_[col].nunique())

        cache_key = self._cache_key(col, ctype, sample)
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            col_name=col,
            col_type=ctype,
            n_unique=n_unique,
            missing_pct=round(missing_pct * 100, 1),
            sample=", ".join(repr(v) for v in sample),
            other_cols=", ".join(other_cols),
        )

        self.llm_call_count_ += 1
        try:
            raw_response = self._provider.complete(
                prompt=user_prompt,
                system=_SYSTEM_PROMPT,
            )
            result = self._safe_json_parse(raw_response)
            if self.cache_enabled:
                self._cache[cache_key] = result
            return result

        except Exception as exc:
            self.llm_failure_count_ += 1
            exc_name = type(exc).__name__
            # Provide extra context for Groq rate limits
            if "ratelimit" in exc_name.lower() or "429" in str(exc):
                logger.warning(
                    "Groq rate limit hit for column '%s'. "
                    "Free tier: 30 RPM / 1,000 RPD. "
                    "Falling back to rule-based for remaining columns.",
                    col,
                )
            else:
                logger.warning(
                    "LLM query failed for column '%s' (%s: %s). "
                    "Falling back to rule-based.",
                    col, exc_name, exc,
                )
            if self.cache_enabled:
                self._cache[cache_key] = None
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self) -> list[dict[str, Any]]:
        """Generate combined rule-based + LLM feature recommendations.

        Rule-based recommendations are always present. LLM suggestions are
        merged in with a ``"[LLM] "`` prefix where available.

        Returns:
            List of recommendation dicts, one per column.
        """
        # Always run rule-based first
        base_recs = self._rule_recommender.recommend()
        base_map: dict[str, dict[str, Any]] = {r["column"]: r for r in base_recs}

        column_types = self._profiler.column_types_
        if not column_types:
            column_types = self._profiler.detect_column_types()

        missing_stats = self._profiler.get_missing_stats()

        # Determine columns to query LLM for
        llm_candidates = list(self.df_.columns)
        if self.skip_text_columns:
            llm_candidates = [
                c for c in llm_candidates
                if column_types.get(c) != DataProfiler.TEXT
            ]
        if self.max_columns is not None:
            llm_candidates = llm_candidates[: self.max_columns]

        for col in llm_candidates:
            ctype = column_types.get(col, DataProfiler.CATEGORICAL)
            missing_pct = missing_stats.get(col, {}).get("fraction", 0.0)
            llm_data = self._query_llm_for_column(col, ctype, missing_pct)

            if llm_data is None:
                continue

            if col not in base_map:
                base_map[col] = {"column": col, "type": ctype, "suggestions": []}

            rec = base_map[col]

            # Merge LLM suggested_features
            llm_suggestions = llm_data.get("suggested_features", [])
            for suggestion in llm_suggestions:
                if isinstance(suggestion, str) and suggestion.strip():
                    rec["suggestions"].append(f"[LLM] {suggestion}")

            # Attach semantic meaning and domain hints
            meaning = llm_data.get("semantic_meaning", "")
            if meaning and isinstance(meaning, str):
                rec["llm_meaning"] = meaning

            domain_hints = llm_data.get("domain_hints", [])
            if domain_hints and isinstance(domain_hints, list):
                rec["llm_domain_hints"] = [
                    h for h in domain_hints if isinstance(h, str) and h.strip()
                ]

        # Preserve original ordering; only include recs with suggestions
        results = [
            base_map[c] for c in self.df_.columns
            if c in base_map and base_map[c].get("suggestions")
        ]
        self.recommendations_ = results
        return results
