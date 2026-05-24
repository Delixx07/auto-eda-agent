"""Missing value detection and imputation strategies."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

from .exceptions import InvalidDataError
from .profiler import DataProfiler
from .utils import setup_logger, validate_dataframe

logger = setup_logger(__name__)


class MissingValueHandler:
    """Analyse and impute missing values using per-column strategies.

    Decision logic (applied per column):

    * ``missing == 0`` -> ``no_action``
    * ``missing_frac >= drop_col_threshold`` -> ``drop_column``
    * ``missing_frac < drop_row_threshold`` -> ``drop_rows``
    * Otherwise by semantic type:

        - numeric -> KNN (if ``len(df) ≤ knn_max_rows``) else median
        - categorical / boolean -> mode
        - datetime -> interpolation
        - text -> constant ``"Unknown"``

    Args:
        data: Input data accepted by
            :func:`~auto_eda_agent.utils.validate_dataframe`.
        profiler: Optional pre-built :class:`~auto_eda_agent.profiler.DataProfiler`.
            Providing one avoids redundant type detection.
        drop_col_threshold: Missing fraction at or above which a column is
            dropped (default: ``0.6``).
        drop_row_threshold: Missing fraction below which affected rows are
            dropped instead of imputed (default: ``0.05``).
        knn_max_rows: Maximum row count for KNN imputation (default: ``10_000``).
            Larger datasets fall back to median.

    Attributes:
        strategy_map_: ``{column: strategy_constant}`` populated by
            :meth:`analyze`.
        actions_log_: Human-readable log of decisions made during
            :meth:`transform`.

    Raises:
        InvalidDataError: For invalid data input or out-of-range thresholds.
    """

    # Strategy constants
    STRATEGY_NO_ACTION = "no_action"
    STRATEGY_DROP_COLUMN = "drop_column"
    STRATEGY_DROP_ROWS = "drop_rows"
    STRATEGY_IMPUTE_MEDIAN = "impute_median"
    STRATEGY_IMPUTE_MODE = "impute_mode"
    STRATEGY_IMPUTE_KNN = "impute_knn"
    STRATEGY_IMPUTE_INTERPOLATE = "impute_interpolate"
    STRATEGY_IMPUTE_CONSTANT = "impute_constant"

    def __init__(
        self,
        data: Any,
        profiler: Optional[DataProfiler] = None,
        drop_col_threshold: float = 0.6,
        drop_row_threshold: float = 0.05,
        knn_max_rows: int = 10_000,
    ) -> None:
        for name, val in [
            ("drop_col_threshold", drop_col_threshold),
            ("drop_row_threshold", drop_row_threshold),
        ]:
            if not 0.0 <= val <= 1.0:
                raise InvalidDataError(
                    f"'{name}' must be in [0, 1]; got {val}."
                )

        self.df_: pd.DataFrame = validate_dataframe(data)
        self.drop_col_threshold = drop_col_threshold
        self.drop_row_threshold = drop_row_threshold
        self.knn_max_rows = knn_max_rows

        if profiler is not None:
            self._profiler = profiler
        else:
            self._profiler = DataProfiler(self.df_)
            self._profiler.detect_column_types()

        self.strategy_map_: dict[str, str] = {}
        self.actions_log_: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> dict[str, str]:
        """Determine the imputation strategy for every column.

        Returns:
            ``{column_name: strategy_constant}`` mapping.
        """
        missing_stats = self._profiler.get_missing_stats()
        column_types = self._profiler.column_types_
        if not column_types:
            column_types = self._profiler.detect_column_types()

        strategy_map: dict[str, str] = {}
        n = len(self.df_)

        for col in self.df_.columns:
            stats = missing_stats[col]
            missing_count = stats["count"]
            missing_frac = stats["fraction"]

            if missing_count == 0:
                strategy_map[col] = self.STRATEGY_NO_ACTION
                continue

            if missing_frac >= self.drop_col_threshold:
                strategy_map[col] = self.STRATEGY_DROP_COLUMN
                continue

            if missing_frac < self.drop_row_threshold:
                strategy_map[col] = self.STRATEGY_DROP_ROWS
                continue

            col_type = column_types.get(col, DataProfiler.CATEGORICAL)

            if col_type == DataProfiler.NUMERIC:
                if n <= self.knn_max_rows:
                    strategy_map[col] = self.STRATEGY_IMPUTE_KNN
                else:
                    strategy_map[col] = self.STRATEGY_IMPUTE_MEDIAN
            elif col_type in (DataProfiler.CATEGORICAL, DataProfiler.BOOLEAN):
                strategy_map[col] = self.STRATEGY_IMPUTE_MODE
            elif col_type == DataProfiler.DATETIME:
                strategy_map[col] = self.STRATEGY_IMPUTE_INTERPOLATE
            else:
                strategy_map[col] = self.STRATEGY_IMPUTE_CONSTANT

        self.strategy_map_ = strategy_map
        return strategy_map

    def transform(self) -> pd.DataFrame:
        """Apply the strategies determined by :meth:`analyze`.

        Processing order:

        1. Drop columns.
        2. Drop rows.
        3. Simple imputations (median, mode, constant, interpolate).
        4. KNN imputation.

        The original DataFrame is not mutated.

        Returns:
            Cleaned :class:`pandas.DataFrame` with reset index.
        """
        if not self.strategy_map_:
            self.analyze()

        self.actions_log_ = []
        df = self.df_.copy()

        # --- Stage 1: Drop columns ---
        cols_to_drop = [
            c for c, s in self.strategy_map_.items()
            if s == self.STRATEGY_DROP_COLUMN
        ]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.actions_log_.append(
                f"Dropped columns (>={self.drop_col_threshold:.0%} missing): {cols_to_drop}"
            )

        # --- Stage 2: Drop rows ---
        drop_row_cols = [
            c for c, s in self.strategy_map_.items()
            if s == self.STRATEGY_DROP_ROWS and c in df.columns
        ]
        if drop_row_cols:
            before = len(df)
            df = df.dropna(subset=drop_row_cols)
            after = len(df)
            self.actions_log_.append(
                f"Dropped {before - after} rows for columns with "
                f"<{self.drop_row_threshold:.0%} missing: {drop_row_cols}"
            )

        # --- Stage 3: Simple imputations ---
        for col in list(df.columns):
            strategy = self.strategy_map_.get(col, self.STRATEGY_NO_ACTION)

            if strategy == self.STRATEGY_IMPUTE_MEDIAN:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                self.actions_log_.append(
                    f"Imputed '{col}' with median ({median_val:.4g})"
                )

            elif strategy == self.STRATEGY_IMPUTE_MODE:
                mode_vals = df[col].mode()
                if len(mode_vals) > 0:
                    df[col] = df[col].fillna(mode_vals.iloc[0])
                    self.actions_log_.append(
                        f"Imputed '{col}' with mode ('{mode_vals.iloc[0]}')"
                    )

            elif strategy == self.STRATEGY_IMPUTE_INTERPOLATE:
                df[col] = df[col].interpolate(method="linear", limit_direction="both")
                self.actions_log_.append(f"Interpolated '{col}' (linear)")

            elif strategy == self.STRATEGY_IMPUTE_CONSTANT:
                df[col] = df[col].fillna("Unknown")
                self.actions_log_.append(f"Filled '{col}' with constant 'Unknown'")

        # --- Stage 4: KNN imputation ---
        knn_cols = [
            c for c, s in self.strategy_map_.items()
            if s == self.STRATEGY_IMPUTE_KNN and c in df.columns
        ]
        if knn_cols:
            numeric_available = [
                c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
            ]
            if len(numeric_available) < 2:
                # Fallback to median when not enough numeric context
                for col in knn_cols:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    self.actions_log_.append(
                        f"KNN fallback -> median for '{col}' "
                        f"(fewer than 2 numeric columns available)"
                    )
                    logger.warning(
                        "KNN fallback to median for '%s': "
                        "need >=2 numeric columns for KNN, found %d.",
                        col, len(numeric_available),
                    )
            else:
                numeric_df = df[numeric_available].copy()
                imputer = KNNImputer(n_neighbors=5)
                imputed_array = imputer.fit_transform(numeric_df)
                imputed_df = pd.DataFrame(
                    imputed_array, columns=numeric_available, index=df.index
                )
                for col in knn_cols:
                    if col in imputed_df.columns:
                        df[col] = imputed_df[col]
                self.actions_log_.append(
                    f"KNN-imputed columns: {knn_cols}"
                )

        df = df.reset_index(drop=True)
        return df

    def fit_transform(self) -> pd.DataFrame:
        """Convenience method: ``analyze()`` then ``transform()``.

        Returns:
            Cleaned :class:`pandas.DataFrame`.
        """
        self.analyze()
        return self.transform()
