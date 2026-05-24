"""Rule-based feature engineering recommendations."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from .profiler import DataProfiler
from .utils import setup_logger, validate_dataframe

logger = setup_logger(__name__)


class FeatureRecommender:
    """Generate rule-based feature engineering recommendations per column.

    Each recommendation is a dict::

        {
            "column": str,
            "type": str,           # semantic type
            "suggestions": [str],  # actionable transformation ideas
            # optional metadata keys vary by type
        }

    Args:
        data: Input data accepted by
            :func:`~auto_eda_agent.utils.validate_dataframe`.
        profiler: Optional pre-built :class:`~auto_eda_agent.profiler.DataProfiler`.

    Attributes:
        df_: Validated :class:`pandas.DataFrame`.
        recommendations_: Populated after :meth:`recommend` is called.
    """

    HIGH_CARDINALITY_THRESHOLD = 10

    def __init__(
        self,
        data: Any,
        profiler: Optional[DataProfiler] = None,
    ) -> None:
        self.df_: pd.DataFrame = validate_dataframe(data)
        self.recommendations_: list[dict[str, Any]] = []

        if profiler is not None:
            self._profiler = profiler
        else:
            self._profiler = DataProfiler(self.df_)
            self._profiler.detect_column_types()

    # ------------------------------------------------------------------
    # Per-type handlers
    # ------------------------------------------------------------------

    def _recommend_datetime(self, col: str) -> dict[str, Any]:
        suggestions = [
            f"Extract year from '{col}' -> '{col}_year'",
            f"Extract month from '{col}' -> '{col}_month'",
            f"Extract day-of-week from '{col}' -> '{col}_dayofweek'",
            f"Extract hour from '{col}' -> '{col}_hour' (if time component present)",
            f"Cyclical encoding: sin/cos of month -> '{col}_month_sin', '{col}_month_cos'",
            f"Cyclical encoding: sin/cos of day-of-week -> '{col}_dow_sin', '{col}_dow_cos'",
            f"Boolean weekend flag -> '{col}_is_weekend'",
            f"Boolean holiday flag -> '{col}_is_holiday' (requires `holidays` library)",
            f"Time-delta since reference date -> '{col}_days_since_ref'",
        ]
        return {
            "column": col,
            "type": DataProfiler.DATETIME,
            "suggestions": suggestions,
        }

    def _recommend_categorical(self, col: str) -> dict[str, Any]:
        series = self.df_[col]
        n_unique = series.nunique()
        suggestions: list[str] = []
        metadata: dict[str, Any] = {"n_unique": n_unique}

        if n_unique <= 1:
            suggestions.append(
                f"Drop '{col}' - near-constant column (nunique={n_unique})"
            )
            metadata["warning"] = "near_constant"
            return {"column": col, "type": DataProfiler.CATEGORICAL,
                    "suggestions": suggestions, **metadata}

        # Near-constant value warning
        top_freq = series.value_counts(normalize=True).iloc[0] if n_unique > 0 else 0.0
        if top_freq > 0.95:
            suggestions.append(
                f"Warning: '{col}' is near-constant - top value covers "
                f"{top_freq:.1%} of rows. Consider dropping."
            )
            metadata["warning"] = "near_constant_value"

        # Missing category
        if series.isna().any():
            suggestions.append(
                f"Encode NaN as explicit 'Missing' category in '{col}'"
            )

        if n_unique <= self.HIGH_CARDINALITY_THRESHOLD:
            suggestions.append(
                f"One-hot encode '{col}' (nunique={n_unique} <= {self.HIGH_CARDINALITY_THRESHOLD})"
            )
            suggestions.append(
                f"Ordinal encode '{col}' if natural ordering exists"
            )
        else:
            suggestions.append(
                f"Target encode '{col}' (high cardinality: nunique={n_unique})"
            )
            suggestions.append(
                f"Frequency/count encode '{col}' -> '{col}_freq'"
            )
            suggestions.append(
                f"Hash encode '{col}' for memory-efficient representation"
            )

        metadata["top_value_frequency"] = float(top_freq)
        return {
            "column": col,
            "type": DataProfiler.CATEGORICAL,
            "suggestions": suggestions,
            **metadata,
        }

    def _recommend_numeric(self, col: str) -> dict[str, Any]:
        series = self.df_[col].dropna()
        suggestions: list[str] = []

        n_unique = self.df_[col].nunique()
        if n_unique == 1:
            suggestions.append(
                f"Drop '{col}' - zero-variance column (single unique value)"
            )
            return {
                "column": col,
                "type": DataProfiler.NUMERIC,
                "suggestions": suggestions,
                "warning": "zero_variance",
            }

        skewness = float(series.skew()) if len(series) > 2 else 0.0
        col_range = float(series.max() - series.min()) if len(series) > 0 else 0.0
        col_min = float(series.min()) if len(series) > 0 else 0.0

        # Skewness transformations
        if skewness > 1.0:
            suggestions.append(
                f"Apply log1p transform to '{col}' (right-skew={skewness:.2f})"
            )
            suggestions.append(
                f"Apply sqrt transform to '{col}' (right-skew={skewness:.2f})"
            )
            if col_min > 0:
                suggestions.append(
                    f"Apply Box-Cox transform to '{col}' "
                    f"(all positive values, skew={skewness:.2f})"
                )
        elif skewness < -1.0:
            suggestions.append(
                f"Apply square transform to '{col}' (left-skew={skewness:.2f})"
            )
            suggestions.append(
                f"Apply exp transform to '{col}' (left-skew={skewness:.2f})"
            )

        # Scaling
        if col_range > 1000 or (col_range > 0 and col_range < 0.01):
            suggestions.append(
                f"Scale '{col}' (range={col_range:.4g}) - "
                "StandardScaler or MinMaxScaler recommended"
            )

        # Binning
        if n_unique > 50:
            suggestions.append(
                f"Bin '{col}' into quantile-based buckets -> '{col}_bin'"
            )
            suggestions.append(
                f"Bin '{col}' into equal-width buckets -> '{col}_bin_eq'"
            )

        # Interaction features (generic hint)
        suggestions.append(
            f"Consider polynomial features or interactions with other numeric columns"
        )

        return {
            "column": col,
            "type": DataProfiler.NUMERIC,
            "suggestions": suggestions,
            "metadata": {
                "skewness": skewness,
                "range": col_range,
            },
        }

    def _recommend_text(self, col: str) -> dict[str, Any]:
        suggestions = [
            f"Extract text length -> '{col}_length'",
            f"Extract word count -> '{col}_word_count'",
            f"Apply TF-IDF vectorisation to '{col}'",
            f"Apply sentence embeddings to '{col}' (e.g. sentence-transformers)",
            f"Extract regex patterns (email, URL, phone) from '{col}'",
            f"Detect language of '{col}' -> '{col}_lang'",
        ]
        return {
            "column": col,
            "type": DataProfiler.TEXT,
            "suggestions": suggestions,
        }

    def _recommend_boolean(self, col: str) -> dict[str, Any]:
        suggestions = [
            f"Cast '{col}' to int (0/1) for model compatibility",
            f"Use '{col}' directly as a binary feature",
        ]
        return {
            "column": col,
            "type": DataProfiler.BOOLEAN,
            "suggestions": suggestions,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self) -> list[dict[str, Any]]:
        """Generate feature engineering recommendations for all columns.

        Returns:
            List of recommendation dicts, one per column, excluding columns
            with empty suggestion lists.
        """
        column_types = self._profiler.column_types_
        if not column_types:
            column_types = self._profiler.detect_column_types()

        dispatch = {
            DataProfiler.DATETIME: self._recommend_datetime,
            DataProfiler.CATEGORICAL: self._recommend_categorical,
            DataProfiler.NUMERIC: self._recommend_numeric,
            DataProfiler.TEXT: self._recommend_text,
            DataProfiler.BOOLEAN: self._recommend_boolean,
        }

        results: list[dict[str, Any]] = []
        for col in self.df_.columns:
            ctype = column_types.get(col, DataProfiler.CATEGORICAL)
            handler = dispatch.get(ctype)
            if handler is None:
                continue
            try:
                rec = handler(col)
                if rec.get("suggestions"):
                    results.append(rec)
            except Exception as exc:
                logger.warning(
                    "Feature recommendation failed for column '%s': %s", col, exc
                )

        self.recommendations_ = results
        return results
