"""Apply smart feature transformations to improve downstream ML."""

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

    Strategy (in order):

    1. **Drop zero-variance columns** — ``nunique <= 1``.
    2. **Drop near-constant categoricals** — top value > 95 %.
    3. **Drop pure-ID columns** — numeric with ``nunique == n_rows`` (e.g. ``PassengerId``).
    4. **Frequency-encode high-cardinality categoricals** — replace the raw
       value with its training-set count. Preserves signal that pure ordinal
       encoding throws away (e.g. ``Name``, ``Ticket``).
    5. **Cap outliers** using IQR fence ``iqr_k`` (default ``5.0`` — mild).
    6. **Apply log1p** to right-skewed positive numeric columns
       (``skew > skew_threshold`` and all values >= 0).
    7. **Standardise** numeric columns (optional, default off).

    Fitted parameters are remembered so :meth:`transform` can apply the same
    transformation to test data without data leakage.

    Args:
        data: Input data accepted by
            :func:`~auto_eda_agent.utils.validate_dataframe`.
        profiler: Optional pre-built :class:`~auto_eda_agent.profiler.DataProfiler`.
        skew_threshold: Skewness above which log1p is applied (default: ``1.0``).
        iqr_k: IQR fence multiplier for outlier capping (default: ``5.0``).
        scale: Apply ``StandardScaler`` to numeric columns (default: ``False``).
        frequency_encode_threshold: Cardinality above which categoricals are
            frequency-encoded instead of left for one-hot/ordinal downstream
            (default: ``20``).
        drop_pure_id: Drop numeric columns where ``nunique == n_rows``
            (default: ``True``).

    Attributes:
        transformations_: Human-readable log of applied steps.
    """

    def __init__(
        self,
        data: Any,
        profiler: Optional[DataProfiler] = None,
        skew_threshold: float = 1.0,
        iqr_k: float = 5.0,
        scale: bool = False,
        frequency_encode_threshold: int = 20,
        drop_pure_id: bool = True,
    ) -> None:
        self.df_: pd.DataFrame = validate_dataframe(data)
        self.skew_threshold = skew_threshold
        self.iqr_k = iqr_k
        self.scale = scale
        self.frequency_encode_threshold = frequency_encode_threshold
        self.drop_pure_id = drop_pure_id

        if profiler is not None:
            self._profiler = profiler
        else:
            self._profiler = DataProfiler(self.df_)
            self._profiler.detect_column_types()

        self.transformations_: list[str] = []
        self._dropped_cols: list[str] = []
        self._freq_maps: dict[str, dict[Any, int]] = {}
        self._iqr_bounds: dict[str, tuple[float, float]] = {}
        self._log_cols: list[str] = []
        self._scalers: dict[str, StandardScaler] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self) -> pd.DataFrame:
        """Fit the transformer on training data and return the transformed result."""
        df = self.df_.copy()
        column_types = self._profiler.column_types_
        if not column_types:
            column_types = self._profiler.detect_column_types()

        n_rows = len(df)

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

        # --- 3. Drop pure-ID numeric columns (e.g. PassengerId) ---
        # Pakai numeric_subtype dari profiler kalau tersedia.
        numeric_subtypes: dict[str, str] = {}
        try:
            numeric_subtypes = self._profiler.get_numeric_subtypes()
        except Exception:
            numeric_subtypes = {}

        if self.drop_pure_id:
            # For small datasets, n_unique == n_rows = pure noise (rows
            # essentially serve as their own labels). For larger datasets,
            # only drop if values are sequential (truly an ID column).
            is_small = n_rows < 500
            for col in list(df.columns):
                if column_types.get(col) != DataProfiler.NUMERIC:
                    continue
                subtype = numeric_subtypes.get(col)

                # 1) Sequential pure-ID: always drop
                if subtype == DataProfiler.NUMERIC_ID:
                    df = df.drop(columns=[col])
                    self._dropped_cols.append(col)
                    self.transformations_.append(
                        f"Dropped '{col}' (sequential pure-ID column)"
                    )
                    continue

                # 2) Small datasets: also drop if n_unique == n_rows
                if is_small:
                    try:
                        n_unique = df[col].nunique(dropna=True)
                    except Exception:
                        continue
                    if n_unique == n_rows and n_rows > 20:
                        df = df.drop(columns=[col])
                        self._dropped_cols.append(col)
                        self.transformations_.append(
                            f"Dropped '{col}' (small dataset: all "
                            f"{n_unique} values unique = noise)"
                        )

        # --- 4. Frequency-encode high-cardinality categoricals ---
        # Instead of dropping (which loses signal) OR ordinal encoding (which
        # injects noise on unseen values), replace each value with its count.
        for col in list(df.columns):
            ctype = column_types.get(col)
            if ctype not in (DataProfiler.CATEGORICAL, DataProfiler.TEXT):
                continue
            try:
                n_unique = df[col].nunique(dropna=True)
            except Exception:
                continue
            if n_unique > self.frequency_encode_threshold:
                value_counts = df[col].value_counts(dropna=False).to_dict()
                self._freq_maps[col] = value_counts
                df[col] = df[col].map(value_counts).fillna(0).astype(float)
                self.transformations_.append(
                    f"Frequency-encoded '{col}' (nunique={n_unique})"
                )

        # --- 5. Cap outliers using IQR (continuous numeric only) ---
        # Skip ordinal (Likert 1-5) and count columns via numeric_subtype.
        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and c not in self._freq_maps     # skip freq-encoded
        ]
        for col in numeric_cols:
            # Skip if column is known to be ordinal/count/id
            sub = numeric_subtypes.get(col)
            if sub in (
                DataProfiler.NUMERIC_ORDINAL,
                DataProfiler.NUMERIC_COUNT,
                DataProfiler.NUMERIC_ID,
            ):
                continue
            series = df[col].dropna()
            if len(series) < 4 or series.nunique() < 20:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - self.iqr_k * iqr
            upper = q3 + self.iqr_k * iqr
            n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
            if n_outliers > 0:
                self._iqr_bounds[col] = (float(lower), float(upper))
                df[col] = df[col].clip(lower=lower, upper=upper)
                self.transformations_.append(
                    f"Capped {n_outliers} outlier(s) in '{col}' (IQR k={self.iqr_k})"
                )

        # --- 6. Apply log1p to right-skewed CONTINUOUS columns ---
        # Skip ordinal/count/id - log doesn't help discrete categories.
        for col in numeric_cols:
            if col not in df.columns:
                continue
            sub = numeric_subtypes.get(col)
            if sub in (
                DataProfiler.NUMERIC_ORDINAL,
                DataProfiler.NUMERIC_COUNT,
                DataProfiler.NUMERIC_ID,
            ):
                continue
            series = df[col].dropna()
            if len(series) < 3 or series.nunique() < 20:
                continue
            try:
                skew = float(series.skew())
            except Exception:
                continue
            col_range = float(series.max() - series.min()) if len(series) > 0 else 0.0
            if (
                skew > self.skew_threshold
                and series.min() >= 0
                and series.max() > 1
                and col_range > 100
            ):
                df[col] = np.log1p(df[col].clip(lower=0))
                self._log_cols.append(col)
                self.transformations_.append(
                    f"Applied log1p to '{col}' (skew={skew:.2f}, range={col_range:.1f})"
                )

        # --- 7. Optional standardisation ---
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
        """Apply the fitted transformations to new data (e.g. test set)."""
        df = validate_dataframe(new_data).copy()

        # Drop the same columns
        cols_to_drop = [c for c in self._dropped_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Frequency encoding (unseen categories -> 0)
        for col, freq_map in self._freq_maps.items():
            if col in df.columns:
                df[col] = df[col].map(freq_map).fillna(0).astype(float)

        # Cap outliers using training bounds
        for col, (lower, upper) in self._iqr_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lower, upper=upper)

        # log1p
        for col in self._log_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))

        # Scaling
        for col, scaler in self._scalers.items():
            if col in df.columns:
                series = df[col]
                median_val = series.median() if not series.isnull().all() else 0
                values = series.fillna(median_val).values.reshape(-1, 1)
                df[col] = scaler.transform(values).flatten()

        return df
