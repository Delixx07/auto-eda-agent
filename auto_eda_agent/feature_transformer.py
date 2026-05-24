"""Apply feature transformations: outlier capping, log1p for skew, scaling."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .profiler import DataProfiler
from .utils import setup_logger, validate_dataframe

logger = setup_logger(__name__)


class FeatureTransformer:
    """Apply automatic feature transformations to improve downstream ML.

    Transformations applied in order:

    1. **Drop zero-variance columns** — ``nunique == 1``.
    2. **Drop near-constant categoricals** — top value > 95 %.
    3. **Cap outliers** using IQR fence with multiplier ``iqr_k`` (default 3.0).
    4. **Apply log1p** to right-skewed positive numeric columns
       (skewness > ``skew_threshold`` and all values >= 0).
    5. **Standardise** numeric columns (``StandardScaler``).

    The transformer remembers the parameters fitted on training data so that
    :meth:`transform` can apply the same transformation to test data without
    data leakage.

    Args:
        data: Input data accepted by
            :func:`~auto_eda_agent.utils.validate_dataframe`.
        profiler: Optional pre-built :class:`~auto_eda_agent.profiler.DataProfiler`.
        skew_threshold: Skewness above which log1p is applied (default: ``1.0``).
        iqr_k: IQR fence multiplier for outlier capping (default: ``3.0``).
        scale: Whether to apply StandardScaler to numeric columns
            (default: ``True``).

    Attributes:
        transformations_: Human-readable log of applied transformations.
    """

    def __init__(
        self,
        data: Any,
        profiler: Optional[DataProfiler] = None,
        skew_threshold: float = 1.0,
        iqr_k: float = 3.0,
        scale: bool = False,
        drop_high_cardinality: int = 50,
    ) -> None:
        self.df_: pd.DataFrame = validate_dataframe(data)
        self.skew_threshold = skew_threshold
        self.iqr_k = iqr_k
        self.scale = scale
        self.drop_high_cardinality = drop_high_cardinality

        if profiler is not None:
            self._profiler = profiler
        else:
            self._profiler = DataProfiler(self.df_)
            self._profiler.detect_column_types()

        self.transformations_: list[str] = []
        self._dropped_cols: list[str] = []
        self._iqr_bounds: dict[str, tuple[float, float]] = {}
        self._log_cols: list[str] = []
        self._scalers: dict[str, StandardScaler] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self) -> pd.DataFrame:
        """Fit the transformer on training data and return the transformed result.

        Returns:
            Transformed :class:`pandas.DataFrame`.
        """
        df = self.df_.copy()
        column_types = self._profiler.column_types_
        if not column_types:
            column_types = self._profiler.detect_column_types()

        # --- 1. Drop zero-variance columns ---
        for col in list(df.columns):
            try:
                n_unique = df[col].nunique(dropna=True)
            except Exception:
                continue
            if n_unique <= 1:
                df = df.drop(columns=[col])
                self._dropped_cols.append(col)
                self.transformations_.append(
                    f"Dropped '{col}' (zero variance, nunique={n_unique})"
                )

        # --- 2. Drop near-constant categoricals ---
        for col in list(df.columns):
            if column_types.get(col) != DataProfiler.CATEGORICAL:
                continue
            try:
                top_freq = df[col].value_counts(normalize=True, dropna=False).iloc[0]
            except Exception:
                continue
            if top_freq > 0.95:
                df = df.drop(columns=[col])
                self._dropped_cols.append(col)
                self.transformations_.append(
                    f"Dropped '{col}' (near-constant, top freq={top_freq:.1%})"
                )

        # --- 2b. Drop high-cardinality categoricals / text (noise to ML) ---
        if self.drop_high_cardinality:
            n_rows = len(df)
            for col in list(df.columns):
                ctype = column_types.get(col)
                if ctype not in (DataProfiler.CATEGORICAL, DataProfiler.TEXT):
                    continue
                try:
                    n_unique = df[col].nunique(dropna=True)
                except Exception:
                    continue
                # Drop kalau kardinalitas tinggi (e.g. nama, ID acak, tiket)
                if n_unique > self.drop_high_cardinality and (n_unique / n_rows) > 0.3:
                    df = df.drop(columns=[col])
                    self._dropped_cols.append(col)
                    self.transformations_.append(
                        f"Dropped '{col}' (high cardinality: nunique={n_unique}, "
                        f"ratio={n_unique/n_rows:.1%} — likely identifier/free-text)"
                    )

        # --- 2c. Drop high-cardinality numeric IDs (e.g. PassengerId) ---
        n_rows = len(df)
        for col in list(df.columns):
            if column_types.get(col) != DataProfiler.NUMERIC:
                continue
            try:
                n_unique = df[col].nunique(dropna=True)
            except Exception:
                continue
            # Kolom numeric dengan nilai unik = jumlah baris -> ID, bukan fitur
            if n_unique == n_rows and n_rows > 20:
                df = df.drop(columns=[col])
                self._dropped_cols.append(col)
                self.transformations_.append(
                    f"Dropped '{col}' (numeric ID: {n_unique} unique values = row count)"
                )

        # --- 3. Cap outliers using IQR ---
        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
        ]
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - self.iqr_k * iqr
            upper = q3 + self.iqr_k * iqr
            self._iqr_bounds[col] = (float(lower), float(upper))
            n_capped = int(((df[col] < lower) | (df[col] > upper)).sum())
            if n_capped > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)
                self.transformations_.append(
                    f"Capped {n_capped} outlier(s) in '{col}' "
                    f"using IQR k={self.iqr_k}"
                )

        # --- 4. Apply log1p to right-skewed positive columns ---
        for col in numeric_cols:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) < 3:
                continue
            try:
                skew = float(series.skew())
            except Exception:
                continue
            if skew > self.skew_threshold and series.min() >= 0:
                df[col] = np.log1p(df[col])
                self._log_cols.append(col)
                self.transformations_.append(
                    f"Applied log1p to '{col}' (skew={skew:.2f})"
                )

        # --- 5. Standardise numeric columns ---
        if self.scale:
            for col in numeric_cols:
                if col not in df.columns:
                    continue
                series = df[col].dropna()
                if len(series) == 0 or series.std() == 0:
                    continue
                scaler = StandardScaler()
                values = df[[col]].fillna(series.median()).values
                df[col] = scaler.fit_transform(values).flatten()
                self._scalers[col] = scaler
            if self._scalers:
                self.transformations_.append(
                    f"StandardScaler applied to {len(self._scalers)} numeric column(s)"
                )

        return df

    def transform(self, new_data: Any) -> pd.DataFrame:
        """Apply the fitted transformations to new data (e.g. test set).

        Args:
            new_data: Test DataFrame (or anything ``validate_dataframe`` accepts).

        Returns:
            Transformed :class:`pandas.DataFrame`.
        """
        df = validate_dataframe(new_data).copy()

        # Drop the same columns
        cols_to_drop = [c for c in self._dropped_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Cap outliers using training bounds
        for col, (lower, upper) in self._iqr_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lower, upper=upper)

        # Apply log1p
        for col in self._log_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))

        # Apply scaling
        for col, scaler in self._scalers.items():
            if col in df.columns:
                series = df[col]
                median_val = series.median() if not series.isnull().all() else 0
                values = series.fillna(median_val).values.reshape(-1, 1)
                df[col] = scaler.transform(values).flatten()

        return df
