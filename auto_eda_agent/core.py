"""AUDA — main orchestrator for the full EDA pipeline."""

from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd

from .anomaly_detector import AnomalyDetector
from .exceptions import PipelineError
from .feature_recommender import FeatureRecommender
from .llm_providers import LLMProvider
from .llm_recommender import LLMFeatureRecommender
from .missing_handler import MissingValueHandler
from .profiler import DataProfiler
from .utils import setup_logger, validate_dataframe

logger = setup_logger(__name__)


class AUDA:
    """Main user-facing orchestrator for automated EDA and feature engineering.

    Usage (rule-based only)::

        agent = AUDA(df)
        agent.run_full_pipeline()
        agent.print_report()
        cleaned = agent.cleaned_df_

    Usage (with LLM augmentation)::

        from auto_eda_agent import AUDA, GroqProvider
        agent = AUDA(df, llm_provider=GroqProvider())
        agent.run_full_pipeline()
        agent.print_report()

    Args:
        data: Input data accepted by
            :func:`~auto_eda_agent.utils.validate_dataframe`.
        verbose: If ``True``, pipeline progress is printed to stdout
            (default: ``True``).
        llm_provider: Optional LLM provider for augmented feature
            recommendations. When ``None``, only rule-based recommendations
            are generated.
        model_type: Hint about the downstream model family — ``"tree"``
            (default), ``"kernel"``, or ``"linear"``. Stored as
            :attr:`model_type` and forwarded to
            :class:`~auto_eda_agent.feature_transformer.FeatureTransformer`
            when called by the user.

    Attributes:
        df_: Validated input :class:`pandas.DataFrame`.
        profile_report_: Populated by :meth:`profile_data`.
        cleaned_df_: Populated by :meth:`handle_missing`.
        missing_report_: Populated by :meth:`handle_missing`.
        anomaly_report_: Populated by :meth:`detect_anomalies`.
        feature_report_: Populated by :meth:`recommend_features`.

    Raises:
        InvalidDataError: If the input data is invalid.
        PipelineError: If :meth:`run_full_pipeline` encounters an error.
    """

    def __init__(
        self,
        data: Any,
        verbose: bool = True,
        llm_provider: Optional[LLMProvider] = None,
        model_type: str = "tree",
    ) -> None:
        self.df_: pd.DataFrame = validate_dataframe(data)
        self.verbose = verbose
        self._llm_provider = llm_provider
        self.model_type = model_type

        self._profiler: Optional[DataProfiler] = None
        self.profile_report_: Optional[dict[str, Any]] = None
        self.cleaned_df_: Optional[pd.DataFrame] = None
        self.missing_report_: Optional[dict[str, Any]] = None
        self.anomaly_report_: Optional[dict[str, Any]] = None
        self.feature_report_: Optional[list[dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    # Pipeline stages (individually callable)
    # ------------------------------------------------------------------

    def profile_data(self) -> dict[str, Any]:
        """Stage 1: Profile the input DataFrame.

        Detects semantic column types and computes summary statistics.

        Returns:
            Profile report dict with keys from
            :meth:`~auto_eda_agent.profiler.DataProfiler.get_summary`.
        """
        self._log("--- [1/4] Profiling data ...")
        self._profiler = DataProfiler(self.df_)
        self._profiler.detect_column_types()
        self.profile_report_ = self._profiler.get_summary()
        self._log(
            f"    {self.profile_report_['n_rows']} rows x "
            f"{self.profile_report_['n_columns']} columns | "
            f"{self.profile_report_['duplicate_rows']} duplicate rows"
        )
        return self.profile_report_

    def handle_missing(self) -> dict[str, Any]:
        """Stage 2: Detect and impute missing values.

        Calls :meth:`profile_data` automatically if not yet run.

        Returns:
            Missing-handling report with keys ``strategy_map`` and
            ``actions_log``.
        """
        if self._profiler is None:
            self.profile_data()

        self._log("--- [2/4] Handling missing values ...")
        handler = MissingValueHandler(self.df_, profiler=self._profiler)
        handler.analyze()
        self.cleaned_df_ = handler.transform()

        strategies = handler.strategy_map_
        action_counts: dict[str, int] = {}
        for s in strategies.values():
            action_counts[s] = action_counts.get(s, 0) + 1

        self.missing_report_ = {
            "strategy_map": strategies,
            "actions_log": handler.actions_log_,
            "strategy_summary": action_counts,
            "rows_before": len(self.df_),
            "rows_after": len(self.cleaned_df_),
        }
        self._log(
            f"    {len(self.df_)} -> {len(self.cleaned_df_)} rows after cleaning"
        )
        return self.missing_report_

    def detect_anomalies(self) -> dict[str, Any]:
        """Stage 3: Detect outliers and anomalies in the (cleaned) data.

        Operates on :attr:`cleaned_df_` if available, otherwise on the
        original DataFrame.

        Returns:
            Anomaly report containing IQR, Z-score, and Isolation Forest
            results.
        """
        source_df = self.cleaned_df_ if self.cleaned_df_ is not None else self.df_
        self._log("--- [3/4] Detecting anomalies ...")

        fresh_profiler = DataProfiler(source_df)
        fresh_profiler.detect_column_types()
        detector = AnomalyDetector(source_df, profiler=fresh_profiler)
        self.anomaly_report_ = detector.detect_all()

        iqr_results = self.anomaly_report_.get("iqr", {})
        total_outliers = sum(
            v.get("n_outliers", 0) for v in iqr_results.values()
            if isinstance(v, dict)
        )
        self._log(f"    IQR detected {total_outliers} outlier(s) across all columns")
        return self.anomaly_report_

    def recommend_features(self) -> list[dict[str, Any]]:
        """Stage 4: Generate feature engineering recommendations.

        Uses :class:`~auto_eda_agent.llm_recommender.LLMFeatureRecommender`
        when an LLM provider was supplied, otherwise falls back to
        :class:`~auto_eda_agent.feature_recommender.FeatureRecommender`.

        Returns:
            List of recommendation dicts.
        """
        source_df = self.cleaned_df_ if self.cleaned_df_ is not None else self.df_
        self._log("--- [4/4] Recommending features ...")

        fresh_profiler = DataProfiler(source_df)
        fresh_profiler.detect_column_types()

        if self._llm_provider is not None:
            recommender = LLMFeatureRecommender(
                source_df,
                provider=self._llm_provider,
                profiler=fresh_profiler,
            )
        else:
            recommender = FeatureRecommender(source_df, profiler=fresh_profiler)

        self.feature_report_ = recommender.recommend()

        llm_count = 0
        if isinstance(recommender, LLMFeatureRecommender):
            llm_count = recommender.llm_call_count_
            failures = recommender.llm_failure_count_
            self._log(
                f"    LLM calls: {llm_count} attempted, {failures} failed "
                f"(rule-based used as fallback)"
            )
        self._log(f"    {len(self.feature_report_)} column(s) with recommendations")
        return self.feature_report_

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(self) -> None:
        """Execute all four pipeline stages in order.

        Wraps any internal exception in a
        :class:`~auto_eda_agent.exceptions.PipelineError`.

        Raises:
            PipelineError: If any stage fails.
        """
        try:
            self.profile_data()
            self.handle_missing()
            self.detect_anomalies()
            self.recommend_features()
        except Exception as exc:
            from .exceptions import PipelineError
            raise PipelineError(
                f"Pipeline failed during execution: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_report(self) -> dict[str, Any]:
        """Return all pipeline results as a single nested dict.

        Returns:
            Dict with keys ``profile``, ``missing_handling``,
            ``anomalies``, ``feature_recommendations``.
        """
        return {
            "profile": self.profile_report_,
            "missing_handling": self.missing_report_,
            "anomalies": self.anomaly_report_,
            "feature_recommendations": self.feature_report_,
        }

    def print_report(self) -> None:
        """Print a human-readable section-by-section pipeline summary."""
        sep = "=" * 70

        print(f"\n{sep}")
        print("  AUTO EDA REPORT")
        print(sep)

        # --- Profile ---
        print("\n[1] DATA PROFILE")
        print("-" * 40)
        if self.profile_report_:
            p = self.profile_report_
            print(f"  Rows              : {p['n_rows']}")
            print(f"  Columns           : {p['n_columns']}")
            print(f"  Duplicate rows    : {p['duplicate_rows']}")
            print(f"  Memory            : {p['memory_usage_mb']} MB")
            print("  Column types:")
            for col, ctype in p["column_types"].items():
                ms = p["missing_stats"].get(col, {})
                frac = ms.get("fraction", 0.0)
                miss_str = f"  [{frac:.1%} missing]" if frac > 0 else ""
                print(f"    {col:<30} {ctype}{miss_str}")
        else:
            print("  (not run)")

        # --- Missing ---
        print("\n[2] MISSING VALUE HANDLING")
        print("-" * 40)
        if self.missing_report_:
            m = self.missing_report_
            print(f"  Rows before / after : {m['rows_before']} -> {m['rows_after']}")
            print("  Strategy summary:")
            for strategy, count in m.get("strategy_summary", {}).items():
                print(f"    {strategy:<30} {count} column(s)")
            if m.get("actions_log"):
                print("  Actions taken:")
                for action in m["actions_log"]:
                    print(f"    * {action}")
        else:
            print("  (not run)")

        # --- Anomalies ---
        print("\n[3] ANOMALY DETECTION")
        print("-" * 40)
        if self.anomaly_report_:
            iqr = self.anomaly_report_.get("iqr", {})
            zscore = self.anomaly_report_.get("zscore", {})
            iso = self.anomaly_report_.get("isolation_forest", {})

            print("  IQR outliers:")
            if iqr:
                for col, info in iqr.items():
                    if isinstance(info, dict):
                        print(
                            f"    {col:<30} {info.get('n_outliers', 0)} outlier(s) "
                            f"[{info.get('lower_fence', ''):.4g}, "
                            f"{info.get('upper_fence', ''):.4g}]"
                        )
            else:
                print("    (none detected)")

            print("  Z-score outliers:")
            if zscore:
                for col, info in zscore.items():
                    if isinstance(info, dict):
                        print(
                            f"    {col:<30} {info.get('n_outliers', 0)} outlier(s) "
                            f"(threshold={info.get('threshold', 3.0)})"
                        )
            else:
                print("    (none detected)")

            print("  Isolation Forest:")
            if isinstance(iso, dict):
                if "error" in iso:
                    print(f"    Note: {iso['error']}")
                else:
                    print(
                        f"    {iso.get('n_anomalies', 0)} anomaly/anomalies detected "
                        f"across {len(iso.get('columns_used', []))} column(s)"
                    )
        else:
            print("  (not run)")

        # --- Feature Recommendations ---
        print("\n[4] FEATURE RECOMMENDATIONS")
        print("-" * 40)
        if self.feature_report_:
            for rec in self.feature_report_:
                col = rec["column"]
                ctype = rec["type"]
                print(f"\n  Column: '{col}'  [{ctype}]")
                meaning = rec.get("llm_meaning")
                if meaning:
                    print(f"  Meaning: {meaning}")
                domain_hints = rec.get("llm_domain_hints", [])
                for hint in domain_hints:
                    print(f"  Domain hint: {hint}")
                for s in rec.get("suggestions", []):
                    if s.startswith("[LLM] "):
                        print(f"  [LLM] {s[6:]}")
                    else:
                        print(f"  * {s}")
        else:
            print("  (not run)")

        print(f"\n{sep}\n")

    def export_report(self, path: str) -> None:
        """Write the full pipeline report to a JSON file.

        Args:
            path: Destination file path (e.g. ``"report.json"``).
        """
        report = self.get_report()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("Report exported to '%s'", path)
