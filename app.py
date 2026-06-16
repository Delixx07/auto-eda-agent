"""AUDA — Gradio Web Interface

Run from the auto-eda-agent/ folder:
    python app.py
"""
from __future__ import annotations

import io
import os
import traceback
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from auto_eda_agent import AUDA
from examples.ml_evaluation import (
    build_dataset_with_target,
    preprocess_baseline,
    preprocess_auda,
    evaluate,
)


# ─────────────────────────── LLM (Gemini) setup ───────────────────────────────

def _load_env(paths=(".env", "../.env")):
    """Load KEY=VALUE pairs from a .env file into os.environ (no extra deps)."""
    for path in paths:
        f = Path(path)
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


def _make_llm_provider():
    """Return (provider, status_message). Gemini preferred, Groq fallback,
    else None (rule-based only). The provider only powers the AI feature-
    engineering suggestions shown in the Feature Recommendations tab."""
    _load_env()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gemini_key:
        try:
            from auto_eda_agent import GeminiProvider
            return GeminiProvider(api_key=gemini_key), "Gemini — AI suggestions ON"
        except Exception as e:
            return None, f"LLM off (Gemini failed: {e})"
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        try:
            from auto_eda_agent import GroqProvider
            return GroqProvider(api_key=groq_key), "Groq — AI suggestions ON"
        except Exception as e:
            return None, f"LLM off (Groq failed: {e})"
    return None, "LLM off — add GEMINI_API_KEY to .env for AI suggestions"


LLM_PROVIDER, LLM_STATUS = _make_llm_provider()


# ─────────────────────────── Report Formatters ────────────────────────────────

def fmt_profile(p: dict) -> str:
    lines = [
        "## Data Profile\n",
        f"| | |",
        f"|---|---|",
        f"| **Rows** | {p.get('n_rows', '?')} |",
        f"| **Columns** | {p.get('n_columns', '?')} |",
        f"| **Duplicate rows** | {p.get('duplicate_rows', '?')} |",
        f"| **Memory** | {p.get('memory_usage_mb', '?')} MB |",
        "",
        "### Column Types\n",
        "| Column | Type | Missing Count | Missing % |",
        "|--------|------|:-------------:|:---------:|",
    ]
    for col, ctype in p.get("column_types", {}).items():
        miss = p.get("missing_stats", {}).get(col, {})
        count = miss.get("count", 0)
        pct = miss.get("fraction", 0) * 100
        lines.append(f"| `{col}` | {ctype} | {count} | {pct:.1f}% |")
    return "\n".join(lines)


def fmt_missing(m: dict) -> str:
    lines = [
        "## Missing Value Handling\n",
        f"**Rows before → after cleaning:** `{m.get('rows_before','?')}` → `{m.get('rows_after','?')}`\n",
        "### Strategy per Column\n",
        "| Column | Strategy |",
        "|--------|----------|",
    ]
    for col, strat in m.get("strategy_map", {}).items():
        lines.append(f"| `{col}` | {strat} |")

    lines += ["", "### Actions Taken\n"]
    for action in m.get("actions_log", []):
        lines.append(f"- {action}")
    return "\n".join(lines)


def fmt_anomaly(a: dict) -> str:
    iqr    = a.get("iqr", {})
    zscore = a.get("zscore", {})
    iso    = a.get("isolation_forest", {})

    lines = ["## Anomaly Detection\n", "### IQR Method\n",
             "| Column | Outliers | Lower Fence | Upper Fence |",
             "|--------|:--------:|:-----------:|:-----------:|"]

    for col, info in iqr.items():
        n  = info.get("n_outliers", 0)
        lo = info.get("lower_fence", "?")
        hi = info.get("upper_fence", "?")
        if isinstance(lo, float): lo = f"{lo:.4g}"
        if isinstance(hi, float): hi = f"{hi:.4g}"
        lines.append(f"| `{col}` | **{n}** | {lo} | {hi} |")

    lines += ["", "### Z-Score Method\n",
              "| Column | Outliers | Mean | Std |",
              "|--------|:--------:|:----:|:---:|"]
    for col, info in zscore.items():
        n   = info.get("n_outliers", 0)
        mu  = info.get("mean", "?")
        std = info.get("std", "?")
        if isinstance(mu,  float): mu  = f"{mu:.4g}"
        if isinstance(std, float): std = f"{std:.4g}"
        lines.append(f"| `{col}` | **{n}** | {mu} | {std} |")

    lines.append("\n### Isolation Forest\n")
    if "error" in iso:
        lines.append(f"> {iso['error']}")
    else:
        n_iso = iso.get("n_anomalies", 0)
        lines.append(f"**{n_iso}** multivariate anomalie(s) detected across all numeric columns.")

    return "\n".join(lines)


def fmt_features(fr: list, llm_status: str = "") -> str:
    header = f"**AI mode:** {llm_status}\n\n" if llm_status else ""
    if not fr:
        return header + "_No feature recommendations available._"

    has_llm = any(
        str(s).startswith("[LLM] ")
        for item in fr for s in item.get("suggestions", [])
    )
    legend = "🤖 = AI-suggested (Gemini) &nbsp;•&nbsp; • = rule-based\n\n" if has_llm else ""

    lines = [header + legend]
    for item in fr:
        col  = item.get("column", "?")
        typ  = item.get("type", "?")
        sugg = item.get("suggestions", [])
        lines.append(f"### `{col}` — {typ}\n")

        meaning = item.get("llm_meaning")
        if meaning:
            lines.append(f"> 🤖 **Meaning:** {meaning}\n")

        for s in sugg:
            if str(s).startswith("[LLM] "):
                lines.append(f"- 🤖 {s[6:]}")
            else:
                lines.append(f"- {s}")

        for hint in item.get("llm_domain_hints", []):
            lines.append(f"- 🤖 _Domain hint:_ {hint}")

        lines.append("")
    return "\n".join(lines)


def build_eda_md(agent: AUDA) -> str:
    parts = []
    if agent.profile_report_:
        parts.append(fmt_profile(agent.profile_report_))
    if agent.missing_report_:
        parts.append("\n---\n" + fmt_missing(agent.missing_report_))
    if agent.anomaly_report_:
        parts.append("\n---\n" + fmt_anomaly(agent.anomaly_report_))
    return "\n".join(parts) if parts else "_No EDA report available._"


# ─────────────────────────── Charts ───────────────────────────────────────────

def bar_chart(base: dict, auda: dict) -> plt.Figure:
    labels = ["Accuracy", "F1-Score", "ROC-AUC"]
    bv = [base["accuracy"], base["f1_score"], base["roc_auc"]]
    av = [auda["accuracy"], auda["f1_score"], auda["roc_auc"]]

    x, w = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    b1 = ax.bar(x - w/2, bv, w, label="Baseline", color="#5B8DB8", alpha=0.9)
    b2 = ax.bar(x + w/2, av, w, label="AUDA",     color="#27AE60", alpha=0.9)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.006,
                f"{bar.get_height():.4f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Baseline vs AUDA — Score Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    return fig


def delta_chart(base: dict, auda: dict) -> plt.Figure:
    labels = ["Accuracy", "F1-Score", "ROC-AUC"]
    deltas = [
        auda["accuracy"] - base["accuracy"],
        auda["f1_score"] - base["f1_score"],
        auda["roc_auc"]  - base["roc_auc"],
    ]
    colors = ["#27AE60" if d >= 0 else "#E74C3C" for d in deltas]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    bars = ax.bar(labels, deltas, color=colors, alpha=0.9)
    for bar, d in zip(bars, deltas):
        sign = "+" if d >= 0 else ""
        y  = bar.get_height() + 0.0003 if d >= 0 else bar.get_height() - 0.0003
        va = "bottom" if d >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2, y,
                f"{sign}{d:.4f}", ha="center", va=va,
                fontsize=12, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("AUDA − Baseline", fontsize=11)
    ax.set_title("Delta: AUDA improvement over Baseline", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    return fig


# ─────────────────────────── Event Handlers ───────────────────────────────────

def on_upload(filepath):
    if filepath is None:
        return None, pd.DataFrame(), gr.Dropdown(choices=[], value=None)
    try:
        df   = pd.read_csv(filepath)
        cols = df.columns.tolist()
        return (
            df.to_json(orient="split"),
            df.head(8),
            gr.Dropdown(choices=cols, value=cols[-1], interactive=True),
        )
    except Exception as e:
        return None, pd.DataFrame(), gr.Dropdown(choices=[], value=None, label=f"Error: {e}")


def on_synthetic():
    df   = build_dataset_with_target(n=800, seed=42)
    cols = df.columns.tolist()
    return (
        df.to_json(orient="split"),
        df.head(8),
        gr.Dropdown(choices=cols, value="will_churn", interactive=True),
    )


def on_run(df_json: str, target_col: str):
    PLACEHOLDER = "_Run analysis to see results._"
    EMPTY = (PLACEHOLDER, PLACEHOLDER, PLACEHOLDER, None, None)

    if not df_json:
        return ("Please load a dataset first.",) + EMPTY[1:]
    if not target_col:
        return ("Please select a target column.",) + EMPTY[1:]

    try:
        df = pd.read_json(io.StringIO(df_json), orient="split")

        feat_df = df.drop(columns=[target_col], errors="ignore")
        agent   = AUDA(feat_df, verbose=False, llm_provider=LLM_PROVIDER)
        agent.run_full_pipeline()

        eda_md  = build_eda_md(agent)
        feat_md = fmt_features(agent.feature_report_ or [], LLM_STATUS)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        n_unique = int(y.dropna().nunique())
        if n_unique > 20:
            eval_md = (
                f"**Target column `{target_col}` has {n_unique} unique values.**\n\n"
                "Please select a **binary or low-cardinality classification** target column."
            )
            return eda_md, feat_md, eval_md, None, None

        if y.dtype == object or str(y.dtype) == "category":
            y = pd.Series(
                LabelEncoder().fit_transform(y.fillna("Unknown")),
                name=target_col,
            )
        else:
            y = y.fillna(0).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_tr_b, X_te_b = preprocess_baseline(X_train.copy(), X_test.copy())
        mb = evaluate(X_tr_b, y_train, X_te_b, y_test, "Baseline")

        X_tr_a, y_tr_a, X_te_a = preprocess_auda(
            X_train.copy(), y_train.copy(), X_test.copy()
        )
        ma = evaluate(X_tr_a, y_tr_a, X_te_a, y_test, "AUDA")

        def fmt_delta(v: float) -> str:
            return f"+{v:.4f}" if v >= 0 else f"{v:.4f}"

        winner      = "AUDA" if ma["roc_auc"] >= mb["roc_auc"] else "Baseline"
        winner_icon = "AUDA wins" if winner == "AUDA" else "Baseline wins"

        eval_md = f"""## ML Evaluation Results

**Dataset:** {df.shape[0]} rows &nbsp;|&nbsp; **Target:** `{target_col}` &nbsp;|&nbsp; **Test set:** {len(y_test)} rows

| Metric | Baseline | AUDA | Delta |
|--------|:--------:|:----:|:-----:|
| Accuracy | {mb['accuracy']:.4f} | {ma['accuracy']:.4f} | {fmt_delta(ma['accuracy']-mb['accuracy'])} |
| F1-Score | {mb['f1_score']:.4f} | {ma['f1_score']:.4f} | {fmt_delta(ma['f1_score']-mb['f1_score'])} |
| ROC-AUC | {mb['roc_auc']:.4f} | {ma['roc_auc']:.4f} | {fmt_delta(ma['roc_auc']-mb['roc_auc'])} |
| Training rows | {mb['train_samples']} | {ma['train_samples']} | {fmt_delta(ma['train_samples']-mb['train_samples'])} |

**Winner (by ROC-AUC): {winner_icon} — {winner}**

---
### What does this mean?
- **Baseline** uses `SimpleImputer(mean)` + `OrdinalEncoder` — fast but naive.
- **AUDA** uses the full 4-stage pipeline: KNN/median/mode imputation, outlier capping, log1p transform.
- **Delta** = AUDA score minus Baseline score. Positive = AUDA is better.
- **ROC-AUC** is the primary metric: measures how well the model separates classes (0.5 = random, 1.0 = perfect).
"""

        return eda_md, feat_md, eval_md, bar_chart(mb, ma), delta_chart(mb, ma)

    except Exception:
        tb  = traceback.format_exc()
        err = f"**An error occurred:**\n```\n{tb}\n```"
        return err, err, err, None, None


# ─────────────────────────── Gradio UI ────────────────────────────────────────

with gr.Blocks(title="AUDA — Automated EDA") as demo:

    df_store = gr.State(None)

    gr.Markdown(
        "# AUDA — Automated EDA & ML Evaluation\n"
        "Upload a CSV dataset or use the built-in synthetic dataset, "
        "select your target column, and compare **Baseline vs AUDA** preprocessing.\n\n"
        f"**AI feature suggestions:** {LLM_STATUS}"
    )

    with gr.Row():

        # ── Left panel: controls ──────────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):

            gr.Markdown("### 1. Load Dataset")
            file_in   = gr.File(label="Upload CSV", file_types=[".csv"], type="filepath")
            synth_btn = gr.Button("Use Synthetic Dataset", variant="secondary")

            gr.Markdown("### 2. Preview")
            preview_tbl = gr.Dataframe(
                label="First 8 rows",
                interactive=False,
                wrap=True,
            )

            gr.Markdown("### 3. Select Target Column")
            target_dd = gr.Dropdown(
                label="Target column (classification label)",
                choices=[],
                interactive=True,
            )

            gr.Markdown("### 4. Run")
            run_btn = gr.Button("Run Analysis", variant="primary", size="lg")

        # ── Right panel: results ──────────────────────────────────────────────
        with gr.Column(scale=3):
            with gr.Tabs():

                with gr.Tab("EDA Report"):
                    eda_out = gr.Markdown("_Load a dataset and click **Run Analysis**._")

                with gr.Tab("Feature Recommendations"):
                    feat_out = gr.Markdown("_Load a dataset and click **Run Analysis**._")

                with gr.Tab("ML Evaluation"):
                    eval_out = gr.Markdown("_Load a dataset and click **Run Analysis**._")
                    with gr.Row():
                        chart_bar   = gr.Plot(label="Score Comparison")
                        chart_delta = gr.Plot(label="Delta (AUDA - Baseline)")

    # ── Wire events ──────────────────────────────────────────────────────────
    file_in.change(
        fn=on_upload,
        inputs=[file_in],
        outputs=[df_store, preview_tbl, target_dd],
    )

    synth_btn.click(
        fn=on_synthetic,
        inputs=[],
        outputs=[df_store, preview_tbl, target_dd],
    )

    run_btn.click(
        fn=on_run,
        inputs=[df_store, target_dd],
        outputs=[eda_out, feat_out, eval_out, chart_bar, chart_delta],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
