"""Groq-augmented EDA example.

Setup:
  1. Get free API key from https://console.groq.com (no credit card)
  2. export GROQ_API_KEY=gsk_...          (Linux/macOS)
     set GROQ_API_KEY=gsk_...            (Windows CMD)
     $env:GROQ_API_KEY = "gsk_..."       (PowerShell)
  3. pip install groq
  4. python -m examples.groq_usage
"""

from __future__ import annotations

import json
import os

from auto_eda_agent import AUDA, GroqProvider, MockProvider
from examples.basic_usage import build_messy_dataframe


def demo_with_groq() -> None:
    """Run the full pipeline with Groq LLM augmentation.

    Skipped gracefully when GROQ_API_KEY is not set.
    """
    if not os.getenv("GROQ_API_KEY"):
        print("[SKIP] Set GROQ_API_KEY to run the Groq demo.")
        print("       Get a free key (no credit card) at https://console.groq.com")
        return

    print("=== Demo: Groq-augmented EDA ===")
    df = build_messy_dataframe(n=200)
    agent = AUDA(
        df,
        verbose=True,
        llm_provider=GroqProvider(model="llama-3.3-70b-versatile"),
    )
    agent.run_full_pipeline()
    agent.print_report()


def demo_with_mock() -> None:
    """Always works — no API key needed.

    Demonstrates the MockProvider for offline testing.
    """
    print("=== Demo: Mock LLM provider (no API key needed) ===")
    df = build_messy_dataframe(n=100)
    mock = MockProvider(json.dumps({
        "semantic_meaning": "Demo column for illustration purposes",
        "suggested_features": ["Mock suggestion 1", "Mock suggestion 2"],
        "domain_hints": [],
    }))
    agent = AUDA(df, verbose=False, llm_provider=mock)
    agent.run_full_pipeline()
    agent.print_report()


if __name__ == "__main__":
    demo_with_mock()
    demo_with_groq()
