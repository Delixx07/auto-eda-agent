"""Multi-seed benchmark + significance test for AUDA vs baseline.

Runs the SAME baseline-vs-AUDA comparison as the notebook, but across several
random train/test splits (seeds) so each delta comes with a mean ± std instead
of a single noisy number. Then runs a Wilcoxon signed-rank test across datasets
to ask: "is AUDA's effect statistically distinguishable from zero?"

Reuses the notebook's helper logic verbatim (run_baseline, run_auda_stages,
apply_feature_transformer, evaluate) so results are comparable.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from auto_eda_agent import AUDA, FeatureTransformer  # noqa: E402
from auto_eda_agent.profiler import DataProfiler  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────
SEEDS = [42, 0, 1, 7, 2023]          # >=5 recommended for a meaningful test
MS_ROW_CAP = 4000                     # smaller cap so 5 seeds finish quickly
USE_SATURATION = False                # False = AUDA always runs (no auto-skip)
SATURATION_THRESHOLD = 0.95
FAIR_BASELINE = True                  # True = strong baseline (median+OHE+scale for SVM)
OHE_MAX = 20                          # one-hot categoricals with <= this many uniques
DATASETS_DIR = os.path.join(REPO_ROOT, "auda_datasets", "datasets")
CONFIG_PATH = os.path.join(DATASETS_DIR, "config.json")
_llm_provider = None                  # rule-based only — LLM doesn't affect ML

RF_MODEL = lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
SVM_MODEL = lambda: SVC(kernel="rbf", probability=True, random_state=42)


# ── Helpers (copied verbatim from notebook cell code-3) ─────────────────────
def safe_encode_target(y):
    le = LabelEncoder()
    return pd.Series(le.fit_transform(y.fillna("__nan__").astype(str)), name=y.name)


def encode_categoricals(X_train, X_test):
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    if not cat_cols:
        return (X_train[num_cols].reset_index(drop=True),
                X_test[num_cols].reset_index(drop=True))
    for col in cat_cols:
        X_train[col] = X_train[col].fillna("Unknown").astype(str)
        X_test[col] = X_test[col].fillna("Unknown").astype(str)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_tr = pd.DataFrame(enc.fit_transform(X_train[cat_cols]), columns=cat_cols)
    cat_te = pd.DataFrame(enc.transform(X_test[cat_cols]), columns=cat_cols)
    X_tr = pd.concat([X_train[num_cols].reset_index(drop=True), cat_tr], axis=1)
    X_te = pd.concat([X_test[num_cols].reset_index(drop=True), cat_te], axis=1)
    return X_tr, X_te


def run_baseline(X_train, X_test):
    X_tr, X_te = X_train.copy(), X_test.copy()
    drop_cols = X_tr.columns[X_tr.isnull().mean() >= 0.6].tolist()
    X_tr = X_tr.drop(columns=drop_cols)
    X_te = X_te.drop(columns=drop_cols, errors="ignore")
    num_cols = X_tr.select_dtypes(include="number").columns.tolist()
    cat_cols = X_tr.select_dtypes(exclude="number").columns.tolist()
    if num_cols:
        imp = SimpleImputer(strategy="mean")
        X_tr[num_cols] = imp.fit_transform(X_tr[num_cols])
        X_te[num_cols] = imp.transform(X_te[num_cols])
    if cat_cols:
        imp = SimpleImputer(strategy="most_frequent")
        X_tr[cat_cols] = imp.fit_transform(X_tr[cat_cols])
        X_te[cat_cols] = imp.transform(X_te[cat_cols])
    return encode_categoricals(X_tr, X_te)


def run_strong_baseline(X_train, X_test, scale=False):
    """A competent manual-EDA baseline: median impute, OHE low-cardinality
    categoricals, frequency-encode high-cardinality, and (for SVM) StandardScaler.
    This is what a skilled analyst would actually do — unlike run_baseline()
    which ordinal-encodes everything and never scales."""
    from sklearn.preprocessing import StandardScaler
    X_tr, X_te = X_train.copy(), X_test.copy()
    drop_cols = X_tr.columns[X_tr.isnull().mean() >= 0.6].tolist()
    X_tr = X_tr.drop(columns=drop_cols)
    X_te = X_te.drop(columns=drop_cols, errors="ignore")

    num_cols = X_tr.select_dtypes(include="number").columns.tolist()
    cat_cols = X_tr.select_dtypes(exclude="number").columns.tolist()

    if num_cols:
        imp = SimpleImputer(strategy="median")
        X_tr[num_cols] = imp.fit_transform(X_tr[num_cols])
        X_te[num_cols] = imp.transform(X_te[num_cols])

    low_card, high_card = [], []
    for c in cat_cols:
        X_tr[c] = X_tr[c].fillna("Unknown").astype(str)
        X_te[c] = X_te[c].fillna("Unknown").astype(str)
        (low_card if X_tr[c].nunique() <= OHE_MAX else high_card).append(c)

    for c in high_card:  # frequency-encode (what a real analyst does for high card)
        vc = X_tr[c].value_counts().to_dict()
        X_tr[c] = X_tr[c].map(vc).fillna(0).astype(float)
        X_te[c] = X_te[c].map(vc).fillna(0).astype(float)

    if low_card:  # one-hot, then align test columns to train
        X_tr = pd.get_dummies(X_tr, columns=low_card)
        X_te = pd.get_dummies(X_te, columns=low_card)
        X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)

    X_tr = X_tr.reset_index(drop=True)
    X_te = X_te.reset_index(drop=True)

    if scale:  # critical for SVM — the whole point of a fair baseline
        cols = X_tr.columns
        sc = StandardScaler()
        X_tr = pd.DataFrame(sc.fit_transform(X_tr), columns=cols)
        X_te = pd.DataFrame(sc.transform(X_te), columns=cols)

    return X_tr, X_te


def run_auda_stages(X_train, y_train):
    X_in = X_train.copy().reset_index(drop=True)
    y_in = pd.Series(
        y_train.values if hasattr(y_train, "values") else list(y_train),
        dtype="int64",
    )
    agent = AUDA(X_in, verbose=False, llm_provider=_llm_provider)
    agent.profile_data()
    agent.handle_missing()
    agent.detect_anomalies()
    agent.recommend_features()
    cleaned_X = agent.cleaned_df_.copy()

    # Align y to the rows AUDA actually KEPT. AUDA's missing handler may drop
    # rows (drop_rows strategy) and then reset the index, so cleaned_X.index is
    # 0..n-1 and no longer maps to original positions. Using y_in[cleaned_X.index]
    # would grab the first n labels by POSITION (scrambling labels whenever the
    # dropped rows aren't at the tail). Reconstruct the survivor mask instead.
    smap = (agent.missing_report_ or {}).get("strategy_map", {})
    drop_row_cols = [c for c, s in smap.items() if s == "drop_rows"]
    if drop_row_cols:
        mask = X_in[drop_row_cols].notna().all(axis=1)
    else:
        mask = pd.Series(True, index=X_in.index)
    if int(mask.sum()) != len(cleaned_X):
        # Safety net: if row counts disagree, fall back to positional alignment
        # rather than returning misaligned labels silently.
        y_aligned = y_in.iloc[: len(cleaned_X)].reset_index(drop=True)
    else:
        y_aligned = y_in[mask.values].reset_index(drop=True)
    cleaned_X = cleaned_X.reset_index(drop=True)
    return cleaned_X, y_aligned


def apply_feature_transformer(cleaned_X, y_aligned, X_test, scale=False,
                              model_type="tree", max_output_features=None):
    keep_cols = [c for c in cleaned_X.columns if c in X_test.columns]
    X_te_c = X_test[keep_cols].copy()
    for col in cleaned_X.select_dtypes(include="number").columns:
        if col in X_te_c.columns and X_te_c[col].isnull().any():
            X_te_c[col] = X_te_c[col].fillna(cleaned_X[col].median())
    for col in cleaned_X.select_dtypes(exclude="number").columns:
        if col in X_te_c.columns and X_te_c[col].isnull().any():
            mode_val = cleaned_X[col].mode()
            X_te_c[col] = X_te_c[col].fillna(
                mode_val.iloc[0] if len(mode_val) else "Unknown")
    prof = DataProfiler(cleaned_X)
    prof.detect_column_types()
    ft = FeatureTransformer(
        cleaned_X, profiler=prof, skew_threshold=1.0, iqr_k=5.0, scale=scale,
        frequency_encode_threshold=20, model_type=model_type,
        max_output_features=max_output_features,
    )
    X_tr_t = ft.fit_transform(y=y_aligned)
    X_te_t = ft.transform(X_te_c)
    X_tr_enc, X_te_enc = encode_categoricals(X_tr_t.copy(), X_te_t.copy())
    return X_tr_enc, X_te_enc, y_aligned


def evaluate(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    all_y = np.concatenate([np.array(y_train), np.array(y_test)])
    n_cls = len(np.unique(all_y))
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred,
                  average="binary" if n_cls == 2 else "weighted", zero_division=0)
    try:
        auc = (roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
               if n_cls == 2 else
               roc_auc_score(y_test, model.predict_proba(X_test),
                             multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")
    return {"auc": auc}


# ── One (dataset, seed) evaluation → RF & SVM delta AUC ─────────────────────
def eval_one(df, tgt, seed):
    X = df.drop(columns=[tgt])
    y = safe_encode_target(df[tgt])
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed)

    n_classes = int(y.nunique())
    cleaned_X, y_aligned = run_auda_stages(X_tr.copy(), y_tr.copy())

    # Baselines. RF gets an unscaled baseline (trees don't need scaling);
    # SVM gets a scaled one. With FAIR_BASELINE the baseline is competent
    # (median + OHE + scale); otherwise it's the original weak baseline.
    if FAIR_BASELINE:
        Xb_tr_rf, Xb_te_rf = run_strong_baseline(X_tr.copy(), X_te.copy(), scale=False)
        Xb_tr_svm, Xb_te_svm = run_strong_baseline(X_tr.copy(), X_te.copy(), scale=True)
    else:
        Xb_tr_rf, Xb_te_rf = run_baseline(X_tr.copy(), X_te.copy())
        Xb_tr_svm, Xb_te_svm = Xb_tr_rf, Xb_te_rf

    use_auda = True
    if USE_SATURATION:
        try:
            scoring = "roc_auc" if n_classes == 2 else "roc_auc_ovr"
            base_cv = cross_val_score(
                RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
                Xb_tr_rf, y_tr, cv=3, scoring=scoring).mean()
            if base_cv >= SATURATION_THRESHOLD:
                use_auda = False
        except Exception:
            pass

    n_train_auda = len(cleaned_X)
    svm_max_feats = max(5, n_train_auda // 3)

    if use_auda:
        Xa_tr_rf, Xa_te_rf, y_a_rf = apply_feature_transformer(
            cleaned_X, y_aligned, X_te.copy(), scale=False, model_type="tree")
        Xa_tr_svm, Xa_te_svm, y_a_svm = apply_feature_transformer(
            cleaned_X, y_aligned, X_te.copy(), scale=True, model_type="kernel",
            max_output_features=svm_max_feats)
    else:
        Xa_tr_rf, Xa_te_rf, y_a_rf = Xb_tr_rf, Xb_te_rf, y_tr
        Xa_tr_svm, Xa_te_svm, y_a_svm = Xb_tr_svm, Xb_te_svm, y_tr

    rf_b = evaluate(Xb_tr_rf, y_tr, Xb_te_rf, y_te, RF_MODEL())["auc"]
    rf_a = evaluate(Xa_tr_rf, y_a_rf, Xa_te_rf, y_te, RF_MODEL())["auc"]
    sv_b = evaluate(Xb_tr_svm, y_tr, Xb_te_svm, y_te, SVM_MODEL())["auc"]
    sv_a = evaluate(Xa_tr_svm, y_a_svm, Xa_te_svm, y_te, SVM_MODEL())["auc"]
    return rf_a - rf_b, sv_a - sv_b


# ── Driver ──────────────────────────────────────────────────────────────────
def main():
    with open(CONFIG_PATH) as f:
        registry = json.load(f)

    print(f"Multi-seed benchmark | seeds={SEEDS} | row_cap={MS_ROW_CAP} "
          f"| saturation={'ON' if USE_SATURATION else 'OFF'} "
          f"| baseline={'FAIR' if FAIR_BASELINE else 'WEAK'}")
    print(f"{len(registry)} datasets\n")

    per_dataset = []  # {dataset, rf_deltas[], svm_deltas[]}
    seen_hashes = {}  # content hash -> first dataset name (dedup)
    t_start = time.time()

    import hashlib
    for idx, entry in enumerate(registry, 1):
        fname, tgt = entry["file"], entry["target"]
        fp = os.path.join(DATASETS_DIR, fname)
        name = os.path.basename(fname)
        if not os.path.exists(fp):
            print(f"[{idx:02d}] {name:<28} SKIP (missing)")
            continue
        # Dedup: skip files whose content is identical to one already seen
        with open(fp, "rb") as fh:
            content_hash = hashlib.md5(fh.read()).hexdigest()
        if content_hash in seen_hashes:
            print(f"[{idx:02d}] {name:<28} SKIP (duplicate of {seen_hashes[content_hash]})")
            continue
        seen_hashes[content_hash] = name
        df = pd.read_csv(fp, low_memory=False)
        if len(df) > MS_ROW_CAP:
            df = df.sample(n=MS_ROW_CAP, random_state=42).reset_index(drop=True)
        col_lower = {c.strip().lower(): c for c in df.columns}
        if tgt not in df.columns:
            if tgt.lower() in col_lower:
                tgt = col_lower[tgt.lower()]
            else:
                print(f"[{idx:02d}] {name:<28} SKIP (no target)")
                continue
        # Drop rows with a missing target — never train to predict "missing".
        # (Without this, NaN targets become a junk "__nan__" class, e.g. weatheraus.)
        df = df[df[tgt].notna()].reset_index(drop=True)
        n_classes = int(df[tgt].dropna().nunique())
        if n_classes < 2 or n_classes > 20 or len(df) < 40:
            print(f"[{idx:02d}] {name:<28} SKIP (classes/size)")
            continue

        rf_deltas, svm_deltas = [], []
        for seed in SEEDS:
            try:
                rf_d, sv_d = eval_one(df, tgt, seed)
                if not np.isnan(rf_d):
                    rf_deltas.append(rf_d)
                if not np.isnan(sv_d):
                    svm_deltas.append(sv_d)
            except Exception as e:
                print(f"     seed {seed} error: {str(e)[:60]}")
        if not rf_deltas and not svm_deltas:
            print(f"[{idx:02d}] {name:<28} all seeds failed")
            continue

        rf_arr, sv_arr = np.array(rf_deltas), np.array(svm_deltas)
        per_dataset.append({
            "dataset": name,
            "rf_mean": rf_arr.mean() if len(rf_arr) else np.nan,
            "rf_std": rf_arr.std() if len(rf_arr) else np.nan,
            "rf_seed42": rf_deltas[0] if rf_deltas else np.nan,
            "svm_mean": sv_arr.mean() if len(sv_arr) else np.nan,
            "svm_std": sv_arr.std() if len(sv_arr) else np.nan,
            "svm_seed42": svm_deltas[0] if svm_deltas else np.nan,
        })
        print(f"[{idx:02d}] {name:<28} "
              f"RF {rf_arr.mean():+.4f}±{rf_arr.std():.4f}  "
              f"SVM {sv_arr.mean():+.4f}±{sv_arr.std():.4f}")

    res = pd.DataFrame(per_dataset)
    out_csv = os.path.join(os.path.dirname(__file__), "multiseed_results.csv")
    res.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print("  MULTI-SEED SUMMARY (per-dataset mean across seeds)")
    print("=" * 70)
    for model, mean_col, std_col, s42_col in [
        ("RF", "rf_mean", "rf_std", "rf_seed42"),
        ("SVM", "svm_mean", "svm_std", "svm_seed42"),
    ]:
        d = res[mean_col].dropna()
        s42 = res[s42_col].dropna()
        if len(d) == 0:
            continue
        win_rate = (d > 0).mean() * 100
        s42_winrate = (s42 > 0).mean() * 100
        # Wilcoxon signed-rank: is the per-dataset delta distribution != 0?
        try:
            stat, p = wilcoxon(d)
            p_str = f"p={p:.4f}"
        except Exception as e:
            p_str = f"(wilcoxon failed: {e})"
        verdict = "SIGNIFICANT" if (p_str.startswith("p=") and p < 0.05) else "NOT significant"
        print(f"\n  [{model}]  n={len(d)} datasets")
        print(f"    Multi-seed mean delta AUC : {d.mean():+.4f}  (median {d.median():+.4f})")
        print(f"    Multi-seed win rate       : {win_rate:.0f}%")
        print(f"    Single-seed (42) win rate : {s42_winrate:.0f}%   <- what you reported before")
        print(f"    Avg within-dataset std    : ±{res[std_col].dropna().mean():.4f}  (noise level)")
        print(f"    Wilcoxon vs 0             : {p_str}  -> {verdict}")

    print(f"\nTotal runtime: {time.time() - t_start:.0f}s")
    print(f"Saved -> {out_csv}")


if __name__ == "__main__":
    main()
