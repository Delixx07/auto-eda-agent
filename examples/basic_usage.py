"""Pure rule-based EDA demo — no API key required.

Run with:
    python -m examples.basic_usage
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from auto_eda_agent import AUDA


def build_messy_dataframe(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic messy DataFrame for demonstration purposes.

    Features injected:

    * ``age``          — numeric, roughly normally distributed, some outliers
    * ``income``       — numeric, right-skewed (lognormal), some extreme outliers
    * ``city``         — categorical, 4 unique strings (low cardinality)
    * ``occupation``   — categorical, 6 unique strings
    * ``signup_date``  — datetime stored as string
    * ``is_premium``   — boolean
    * ``comment``      — long text column
    * ``almost_all_null`` — 100 % NaN → should trigger drop_column strategy
    * Various NaN rates injected per column
    * 5 duplicate rows appended

    Args:
        n: Number of base rows before duplicates (default: ``500``).
        seed: NumPy random seed for reproducibility (default: ``42``).

    Returns:
        A deliberately messy :class:`pandas.DataFrame`.
    """
    rng = np.random.default_rng(seed)

    age = rng.normal(35, 12, n).clip(18, 90).astype(float)
    # Inject extreme outliers
    age[rng.integers(0, n, 3)] = [200, -10, 150]

    income = rng.lognormal(mean=10.5, sigma=1.2, size=n)
    # Extreme income outlier
    income[rng.integers(0, n, 2)] = [1e8, 5e7]

    cities = ["New York", "London", "Tokyo", "Sydney"]
    city = rng.choice(cities, size=n)

    occupations = ["Engineer", "Teacher", "Doctor", "Artist", "Lawyer", "Nurse"]
    occupation = rng.choice(occupations, size=n)

    # Datetime as string
    base_date = pd.Timestamp("2020-01-01")
    days_offsets = rng.integers(0, 365 * 4, size=n)
    signup_date = [
        (base_date + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in days_offsets
    ]

    is_premium = rng.choice([True, False], size=n, p=[0.2, 0.8])

    comment_templates = [
        "This is a fairly long comment about the service experience and overall satisfaction.",
        "The product quality exceeded my expectations in every possible way.",
        "I had some issues but the support team resolved them quickly and professionally.",
        "Highly recommended to anyone looking for reliable and fast service.",
        "Average experience, nothing particularly notable but not bad either.",
    ]
    comment = rng.choice(comment_templates, size=n)

    almost_all_null = np.full(n, np.nan)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "city": city,
        "occupation": occupation,
        "signup_date": signup_date,
        "is_premium": is_premium,
        "comment": comment,
        "almost_all_null": almost_all_null,
    })

    # Inject NaN at varying rates — cast bool column to nullable first
    nan_rates = {
        "age": 0.03,
        "income": 0.10,
        "city": 0.02,
        "occupation": 0.07,
        "signup_date": 0.04,
        "is_premium": 0.01,
        "comment": 0.15,
    }
    # pandas 2.x bool dtype cannot hold NaN — use nullable boolean
    df["is_premium"] = df["is_premium"].astype(object)
    for col, rate in nan_rates.items():
        nan_idx = rng.choice(n, size=int(n * rate), replace=False)
        df.loc[nan_idx, col] = np.nan

    # Append 5 exact duplicate rows
    duplicate_rows = df.iloc[:5].copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    return df


def main() -> None:
    print("Building synthetic messy DataFrame ...")
    df = build_messy_dataframe(n=500, seed=42)
    print(f"DataFrame shape: {df.shape}")
    print(f"Missing values per column:\n{df.isna().sum()}\n")

    agent = AUDA(df, verbose=True)
    agent.run_full_pipeline()
    agent.print_report()


if __name__ == "__main__":
    main()
