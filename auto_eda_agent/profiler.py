"""Column-type detection and dataset profiling."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import InvalidDataError
from .utils import setup_logger, validate_dataframe

logger = setup_logger(__name__)


class DataProfiler:
    """Detects semantic column types and summarises dataset statistics.

    Semantic types are coarser than pandas dtypes — a column stored as
    ``object`` might be ``categorical``, ``datetime``, or ``text`` depending
    on its content.

    Type constants:
        NUMERIC, CATEGORICAL, DATETIME, TEXT, BOOLEAN

    Args:
        data: Input data accepted by
            :func:`~auto_eda_agent.utils.validate_dataframe`.
        categorical_threshold: Maximum number of unique values for a
            string-like column to be considered ``categorical``
            (default: ``20``).
        text_length_threshold: Minimum average string length for a column
            to be classified as ``text`` (default: ``50``).

    Attributes:
        df_: Validated :class:`pandas.DataFrame`.
        column_types_: ``{column_name: type_string}`` mapping populated after
            :meth:`detect_column_types` is called.

    Raises:
        InvalidDataError: If the input data is invalid.
    """

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"

    _ALL_TYPES = {NUMERIC, CATEGORICAL, DATETIME, TEXT, BOOLEAN}

    def __init__(
        self,
        data: Any,
        categorical_threshold: int = 20,
        text_length_threshold: int = 50,
    ) -> None:
        self.df_: pd.DataFrame = validate_dataframe(data)
        self.categorical_threshold = categorical_threshold
        self.text_length_threshold = text_length_threshold
        self.column_types_: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_string_like(series: pd.Series) -> bool:
        """Return True for columns whose values are string-like.

        Handles both ``object`` dtype (pandas < 2.x) and the dedicated
        ``StringDtype`` / ``pd.StringDtype`` used in pandas 2.x.
        """
        return (
            series.dtype == object
            or pd.api.types.is_string_dtype(series)
        )

    def _detect_single_column(self, series: pd.Series) -> str:
        """Apply the type-detection waterfall to a single column.

        Detection order:
            1. Empty after dropna → ``text``.
            2. Bool dtype → ``boolean``.
            3. Datetime64 dtype → ``datetime``.
            4. String-like + parses as datetime on ≥80 % of sample → ``datetime``.
            5. Numeric: unique values ⊆ {0,1}/{True,False} → ``boolean``;
               nunique ≤ 2 → ``categorical``; else → ``numeric``.
            6. String-like: low cardinality → ``categorical``; long avg
               length → ``text``; else → ``categorical``.

        Args:
            series: A single column from the profiler's DataFrame.

        Returns:
            One of the type-constant strings.
        """
        non_null = series.dropna()

        # 1. Empty after removing nulls
        if len(non_null) == 0:
            return self.TEXT

        # 2. Bool dtype
        if pd.api.types.is_bool_dtype(series):
            return self.BOOLEAN

        # 3. Native datetime64
        if pd.api.types.is_datetime64_any_dtype(series):
            return self.DATETIME

        # 4. String-like — try datetime parsing on a sample
        if self._is_string_like(series):
            sample = non_null.head(100)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    parsed = pd.to_datetime(sample, errors="coerce")
                parse_rate = parsed.notna().mean()
                if parse_rate >= 0.80:
                    return self.DATETIME
            except Exception:
                pass

        # 5. Numeric columns
        if pd.api.types.is_numeric_dtype(series):
            unique_vals = set(non_null.unique())
            if unique_vals <= {0, 1} or unique_vals <= {True, False}:
                return self.BOOLEAN
            if series.nunique() <= 2:
                return self.CATEGORICAL
            return self.NUMERIC

        # 6. String-like — cardinality / length heuristics
        if self._is_string_like(series):
            n_unique = series.nunique()
            n_total = len(series.dropna())
            unique_ratio = n_unique / n_total if n_total > 0 else 1.0

            if n_unique <= self.categorical_threshold or unique_ratio < 0.05:
                return self.CATEGORICAL

            try:
                avg_len = non_null.astype(str).str.len().mean()
            except Exception:
                avg_len = 0

            if avg_len > self.text_length_threshold:
                return self.TEXT
            return self.CATEGORICAL

        # Fallback
        return self.CATEGORICAL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_column_types(self) -> dict[str, str]:
        """Detect the semantic type of every column in the DataFrame.

        Results are cached in :attr:`column_types_`.

        Returns:
            Mapping of ``{column_name: type_string}``.
        """
        result: dict[str, str] = {}
        for col in self.df_.columns:
            try:
                result[col] = self._detect_single_column(self.df_[col])
            except Exception as exc:
                logger.warning("Type detection failed for column '%s': %s", col, exc)
                result[col] = self.CATEGORICAL
        self.column_types_ = result
        return result

    def get_missing_stats(self) -> dict[str, dict[str, Any]]:
        """Compute per-column missing-value statistics.

        Returns:
            Mapping of ``{column: {"count": int, "fraction": float}}``.
        """
        stats: dict[str, dict[str, Any]] = {}
        n = len(self.df_)
        for col in self.df_.columns:
            missing_count = int(self.df_[col].isna().sum())
            stats[col] = {
                "count": missing_count,
                "fraction": missing_count / n if n > 0 else 0.0,
            }
        return stats

    def get_summary(self) -> dict[str, Any]:
        """Return a high-level dataset summary.

        Calls :meth:`detect_column_types` and :meth:`get_missing_stats`
        internally.

        Returns:
            Dict with keys ``n_rows``, ``n_columns``, ``column_types``,
            ``missing_stats``, ``duplicate_rows``, ``memory_usage_mb``.
        """
        if not self.column_types_:
            self.detect_column_types()

        return {
            "n_rows": len(self.df_),
            "n_columns": len(self.df_.columns),
            "column_types": dict(self.column_types_),
            "missing_stats": self.get_missing_stats(),
            "duplicate_rows": int(self.df_.duplicated().sum()),
            "memory_usage_mb": round(
                self.df_.memory_usage(deep=True).sum() / 1_048_576, 4
            ),
        }

    def get_columns_by_type(self, dtype: str) -> list[str]:
        """Return column names matching the given semantic type.

        Args:
            dtype: One of the type constants (e.g. ``DataProfiler.NUMERIC``).

        Returns:
            List of matching column names.

        Raises:
            InvalidDataError: If *dtype* is not a recognised type constant.
        """
        if dtype not in self._ALL_TYPES:
            raise InvalidDataError(
                f"Unknown type '{dtype}'. Valid types: {sorted(self._ALL_TYPES)}"
            )
        if not self.column_types_:
            self.detect_column_types()
        return [col for col, t in self.column_types_.items() if t == dtype]
