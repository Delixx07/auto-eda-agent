"""Outlier and anomaly detection for numeric columns."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .profiler import DataProfiler
from .utils import setup_logger, validate_dataframe

logger = setup_logger(__name__)


class AnomalyDetector:
    """Detect outliers and anomalies in numeric columns using multiple methods.

    Three detection strategies are provided:

    * **IQR**: Interquartile-range fence method.
    * **Z-score**: Standard-deviation–based threshold.
    * **Isolation Forest**: Unsupervised ML ensemble method.

    The original DataFrame is never mutated by any detection method.

    Args:
        data: Input data accepted by
            :func:`~auto_eda_agent.utils.validate_dataframe`.
        profiler: Optional pre-built :class:`~auto_eda_agent.profiler.DataProfiler`.
        max_reported_indices: Maximum number of outlier row indices included
            in results to avoid bloating the report (default: ``100``).

    Attributes:
        df_: Validated :class:`pandas.DataFrame`.
    """

    def __init__(
        self,
        data: Any,
        profiler: Optional[DataProfiler] = None,
        max_reported_indices: int = 100,
    ) -> None:
        self.df_: pd.DataFrame = validate_dataframe(data)
        self.max_reported_indices = max_reported_indices

        if profiler is not None:
            self._profiler = profiler
        else:
            self._profiler = DataProfiler(self.df_)
            self._profiler.detect_column_types()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_numeric_columns(self, columns: Optional[list[str]]) -> list[str]:
        """Return the intersection of *columns* with numeric columns.

        If *columns* is None, returns all numeric columns.
        """
        numeric_cols = self._profiler.get_columns_by_type(DataProfiler.NUMERIC)
        if columns is None:
            return numeric_cols
        return [c for c in columns if c in numeric_cols]

    def _cap_indices(self, indices: list[int]) -> list[int]:
        return indices[: self.max_reported_indices]

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def detect_iqr(
        self,
        columns: Optional[list[str]] = None,
        k: float = 1.5,
    ) -> dict[str, Any]:
        """Detect outliers using the interquartile-range (IQR) fence.

        A value is an outlier if it falls outside
        ``[Q1 - k*IQR, Q3 + k*IQR]``.

        Columns with fewer than 4 non-null values are skipped.

        Args:
            columns: Columns to check. Defaults to all numeric columns.
            k: Fence multiplier (default: ``1.5``).

        Returns:
            Dict mapping column names to::

                {
                    "method": "iqr",
                    "k": float,
                    "lower_fence": float,
                    "upper_fence": float,
                    "n_outliers": int,
                    "outlier_indices": [int, ...],
                }

            Skipped columns are absent from the result.
        """
        target_cols = self._get_numeric_columns(columns)
        results: dict[str, Any] = {}

        for col in target_cols:
            series = self.df_[col].dropna()
            if len(series) < 4:
                logger.debug("IQR: skipping '%s' (fewer than 4 non-null values)", col)
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            mask = (self.df_[col] < lower) | (self.df_[col] > upper)
            outlier_idx = self._cap_indices(list(self.df_.index[mask]))
            results[col] = {
                "method": "iqr",
                "k": k,
                "lower_fence": float(lower),
                "upper_fence": float(upper),
                "n_outliers": int(mask.sum()),
                "outlier_indices": outlier_idx,
            }

        return results

    def detect_zscore(
        self,
        columns: Optional[list[str]] = None,
        threshold: float = 3.0,
    ) -> dict[str, Any]:
        """Detect outliers using the Z-score method.

        A value is an outlier if ``|z| > threshold``.

        Columns where std is 0 or NaN are skipped.

        Args:
            columns: Columns to check. Defaults to all numeric columns.
            threshold: Z-score threshold (default: ``3.0``).

        Returns:
            Dict mapping column names to::

                {
                    "method": "zscore",
                    "threshold": float,
                    "mean": float,
                    "std": float,
                    "n_outliers": int,
                    "outlier_indices": [int, ...],
                }
        """
        target_cols = self._get_numeric_columns(columns)
        results: dict[str, Any] = {}

        for col in target_cols:
            series = self.df_[col].dropna()
            std = series.std()
            if pd.isna(std) or std == 0:
                logger.debug("Z-score: skipping '%s' (std is 0 or NaN)", col)
                continue
            mean = series.mean()
            z_scores = (self.df_[col] - mean).abs() / std
            mask = z_scores > threshold
            outlier_idx = self._cap_indices(list(self.df_.index[mask.fillna(False)]))
            results[col] = {
                "method": "zscore",
                "threshold": threshold,
                "mean": float(mean),
                "std": float(std),
                "n_outliers": int(mask.sum()),
                "outlier_indices": outlier_idx,
            }

        return results

    def detect_isolation_forest(
        self,
        columns: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Detect anomalies using Isolation Forest.

        Operates on the joint distribution of all selected numeric columns.
        Returns ``{"error": "..."}`` (does not raise) when:

        * No numeric columns are available.
        * Fewer than 10 rows have complete data across selected columns.

        Args:
            columns: Columns to use. Defaults to all numeric columns.

        Returns:
            ``{"method": "isolation_forest", "n_anomalies": int,
            "anomaly_indices": [int, ...]}``
            or ``{"error": "..."}`` on failure.
        """
        target_cols = self._get_numeric_columns(columns)

        if not target_cols:
            return {"error": "No numeric columns available for Isolation Forest."}

        subset = self.df_[target_cols].dropna()
        if len(subset) < 10:
            return {
                "error": (
                    f"Isolation Forest requires >=10 complete rows; "
                    f"found {len(subset)} after dropping NaNs."
                )
            }

        try:
            clf = IsolationForest(n_jobs=-1, random_state=42)
            preds = clf.fit_predict(subset)
            anomaly_mask = preds == -1
            anomaly_idx = self._cap_indices(list(subset.index[anomaly_mask]))
            return {
                "method": "isolation_forest",
                "columns_used": target_cols,
                "n_anomalies": int(anomaly_mask.sum()),
                "anomaly_indices": anomaly_idx,
            }
        except Exception as exc:
            logger.warning("Isolation Forest failed: %s", exc)
            return {"error": str(exc)}

    def detect_all(
        self,
        columns: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Run all three detection methods.

        Args:
            columns: Columns to inspect. Defaults to all numeric columns.

        Returns:
            ``{"iqr": {...}, "zscore": {...}, "isolation_forest": {...}}``.
        """
        return {
            "iqr": self.detect_iqr(columns=columns),
            "zscore": self.detect_zscore(columns=columns),
            "isolation_forest": self.detect_isolation_forest(columns=columns),
        }
