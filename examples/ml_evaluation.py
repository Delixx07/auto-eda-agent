"""
Downstream ML Performance Evaluation
======================================
Membuktikan bahwa data yang dibersihkan AUDA menghasilkan model ML
lebih akurat dibanding data mentah (baseline naive preprocessing).

Alur:
  1. Generate dataset sintetis dengan target label (will_churn)
  2. Split train/test (sama persis untuk kedua pendekatan)
  3. Baseline : SimpleImputer(mean) + OrdinalEncoder + RandomForest
  4. AUDA     : AUDA.run_full_pipeline() + OrdinalEncoder + RandomForest
  5. Bandingkan Accuracy, F1-Score, ROC-AUC di test set yang sama

Jalankan:
  python -m examples.ml_evaluation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder 

from auto_eda_agent import AUDA, FeatureTransformer
from auto_eda_agent.profiler import DataProfiler
from examples.basic_usage import build_messy_dataframe


# ---------------------------------------------------------------------------
# 1. Build dataset with target
# ---------------------------------------------------------------------------

def build_dataset_with_target(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Buat dataset messy + kolom target 'will_churn'.

    Aturan target (deterministik + noise):
      - income rendah (< median) DAN umur tua (> 45) -> cenderung churn
      - Ditambah noise acak 30%
    """
    rng = np.random.default_rng(seed)
    df = build_messy_dataframe(n=n, seed=seed)

    income_filled = df["income"].fillna(df["income"].median())
    age_filled    = df["age"].fillna(df["age"].median())

    income_score = (income_filled < income_filled.median()).astype(float)
    age_score    = (age_filled > 45).astype(float)
    noise        = rng.uniform(0, 1, len(df))

    churn_prob = 0.4 * income_score + 0.3 * age_score + 0.3 * noise
    df["will_churn"] = (churn_prob > 0.5).astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. Preprocessing helpers
# ---------------------------------------------------------------------------

def _encode_categoricals(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """OrdinalEncoder untuk kolom non-numerik (string / datetime)."""
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
    num_cols = X_train.select_dtypes(include="number").columns.tolist()

    if not cat_cols:
        return X_train[num_cols], X_test[num_cols]

    # Pastikan string
    for col in cat_cols:
        X_train[col] = X_train[col].fillna("Unknown").astype(str)
        X_test[col]  = X_test[col].fillna("Unknown").astype(str)

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_train = pd.DataFrame(
        enc.fit_transform(X_train[cat_cols]),
        columns=cat_cols, index=X_train.index,
    )
    cat_test = pd.DataFrame(
        enc.transform(X_test[cat_cols]),
        columns=cat_cols, index=X_test.index,
    )

    X_tr = pd.concat([X_train[num_cols].reset_index(drop=True),
                       cat_train.reset_index(drop=True)], axis=1)
    X_te = pd.concat([X_test[num_cols].reset_index(drop=True),
                       cat_test.reset_index(drop=True)], axis=1)
    return X_tr, X_te


def preprocess_baseline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baseline: drop kolom >= 60% missing, SimpleImputer(mean/most_frequent)."""
    X_train = X_train.copy()
    X_test  = X_test.copy()

    miss_frac  = X_train.isnull().mean()
    drop_cols  = miss_frac[miss_frac >= 0.6].index.tolist()
    X_train    = X_train.drop(columns=drop_cols)
    X_test     = X_test.drop(columns=drop_cols, errors="ignore")

    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    if num_cols:
        num_imp = SimpleImputer(strategy="mean")
        X_train[num_cols] = num_imp.fit_transform(X_train[num_cols])
        X_test[num_cols]  = num_imp.transform(X_test[num_cols])

    if cat_cols:
        cat_imp = SimpleImputer(strategy="most_frequent")
        X_train[cat_cols] = cat_imp.fit_transform(X_train[cat_cols])
        X_test[cat_cols]  = cat_imp.transform(X_test[cat_cols])

    return _encode_categoricals(X_train, X_test)


def preprocess_auda(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """AUDA pipeline + FeatureTransformer pada training, lalu align test set.

    Tahap:
      1. AUDA.run_full_pipeline() -> imputasi missing, drop kolom/baris buruk
      2. FeatureTransformer.fit_transform() -> cap outlier, log1p skew, scaling
      3. Align test set dengan kolom & statistik training (no data leakage)
      4. Encode categoricals dengan OrdinalEncoder
    """
    df_train = X_train.copy()
    df_train["__target__"] = y_train.values

    agent = AUDA(df_train, verbose=False)
    agent.run_full_pipeline()
    cleaned = agent.cleaned_df_

    X_train_clean = cleaned.drop(columns=["__target__"])
    y_train_clean = cleaned["__target__"].astype(int)

    keep_cols = [c for c in X_train_clean.columns if c in X_test.columns]
    X_test_aligned = X_test[keep_cols].copy()

    num_cols = X_train_clean.select_dtypes(include="number").columns
    cat_cols = X_train_clean.select_dtypes(exclude="number").columns

    for col in num_cols:
        if col in X_test_aligned.columns and X_test_aligned[col].isnull().any():
            X_test_aligned[col] = X_test_aligned[col].fillna(X_train_clean[col].median())
    for col in cat_cols:
        if col in X_test_aligned.columns and X_test_aligned[col].isnull().any():
            mode_val = X_train_clean[col].mode()
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
            X_test_aligned[col] = X_test_aligned[col].fillna(fill_val)

    profiler_full = DataProfiler(X_train_clean)
    profiler_full.detect_column_types()

    transformer = FeatureTransformer(
        X_train_clean, profiler=profiler_full,
        skew_threshold=1.0, iqr_k=5.0, scale=False,
        frequency_encode_threshold=50,
    )
    X_train_t = transformer.fit_transform()
    X_test_t  = transformer.transform(X_test_aligned)

    X_tr, X_te = _encode_categoricals(X_train_t.copy(), X_test_t.copy())
    return X_tr, y_train_clean, X_te


# ---------------------------------------------------------------------------
# 3. Train & evaluate
# ---------------------------------------------------------------------------

def evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label: str,
) -> dict:
    """Latih RandomForest, kembalikan dict metrik."""
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "label":          label,
        "train_samples":  len(X_train),
        "test_samples":   len(X_test),
        "accuracy":       accuracy_score(y_test, y_pred),
        "f1_score":       f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":        roc_auc_score(y_test, y_prob),
    }


# ---------------------------------------------------------------------------
# 4. Print comparison
# ---------------------------------------------------------------------------

def print_comparison(base: dict, auda: dict) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("  ML PERFORMANCE COMPARISON: Baseline vs AUDA")
    print(sep)
    print(f"  {'Metric':<28} {'Baseline':>10} {'AUDA':>10} {'Delta':>10}")
    print("-" * 65)

    for key, label in [
        ("accuracy", "Accuracy"),
        ("f1_score", "F1-Score"),
        ("roc_auc",  "ROC-AUC"),
    ]:
        b, a = base[key], auda[key]
        delta = a - b
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:<28} {b:>10.4f} {a:>10.4f} {sign}{delta:>9.4f}")

    print("-" * 65)
    print(f"  {'Training samples':<28} {base['train_samples']:>10} {auda['train_samples']:>10}")
    print(sep)

    winner = "AUDA" if auda["roc_auc"] >= base["roc_auc"] else "Baseline"
    delta_auc = auda["roc_auc"] - base["roc_auc"]
    pct = abs(delta_auc) / base["roc_auc"] * 100

    print(f"\n  Pemenang (ROC-AUC)  : {winner}")
    if winner == "AUDA":
        print(f"  AUDA meningkatkan ROC-AUC sebesar {pct:.2f}%")
        extra = auda["train_samples"] - base["train_samples"]
        if extra > 0:
            print(f"  AUDA mempertahankan {extra} baris lebih banyak untuk training")
    else:
        print(f"  Baseline unggul dengan selisih {pct:.2f}%")
    print()


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def run_evaluation(df: pd.DataFrame, target_col: str) -> None:
    """Jalankan evaluasi ML pada dataframe dan kolom target yang diberikan.

    Args:
        df: Dataset lengkap (boleh ada missing values, outlier, dll).
        target_col: Nama kolom target/label (harus ada di df).
    """
    print("=" * 65)
    print("  DOWNSTREAM ML EVALUATION")
    print("=" * 65)

    if target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan di dataset.")

    print(f"\nDataset        : {df.shape[0]} baris x {df.shape[1]} kolom")
    print(f"Target kolom   : '{target_col}'")
    dist = df[target_col].value_counts()
    print(f"Distribusi target:\n{dist.to_string()}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain/Test     : {len(X_train)} / {len(X_test)}")

    print("\n[1/2] Baseline preprocessing ...")
    X_tr_b, X_te_b = preprocess_baseline(X_train, X_test)
    metrics_base = evaluate(X_tr_b, y_train, X_te_b, y_test, "Baseline")
    print(f"  Accuracy={metrics_base['accuracy']:.4f}  "
          f"F1={metrics_base['f1_score']:.4f}  "
          f"AUC={metrics_base['roc_auc']:.4f}  "
          f"(train={metrics_base['train_samples']})")

    print("\n[2/2] AUDA pipeline ...")
    X_tr_a, y_tr_a, X_te_a = preprocess_auda(X_train, y_train, X_test)
    metrics_auda = evaluate(X_tr_a, y_tr_a, X_te_a, y_test, "AUDA")
    print(f"  Accuracy={metrics_auda['accuracy']:.4f}  "
          f"F1={metrics_auda['f1_score']:.4f}  "
          f"AUC={metrics_auda['roc_auc']:.4f}  "
          f"(train={metrics_auda['train_samples']})")

    print_comparison(metrics_base, metrics_auda)


def main() -> None:
    # ---------------------------------------------------------------
    # UBAH BAGIAN INI SAJA
    # ---------------------------------------------------------------

    # Pilihan A: pakai dataset sintetis bawaan (default)
    df = build_dataset_with_target(n=800, seed=42)
    target_col = "will_churn"

    # Pilihan B: pakai file CSV sendiri -> hapus tanda # di bawah ini
    # df = pd.read_csv("train.csv")
    # target_col = "Survived"

    # Pilihan C: pakai dataset Titanic dari seaborn -> hapus tanda # di bawah ini
    # import seaborn as sns
    # df = sns.load_dataset("titanic").dropna(subset=["survived"])
    # target_col = "survived"
    # ---------------------------------------------------------------

    run_evaluation(df, target_col)


if __name__ == "__main__":
    main()
