"""LLM provider abstractions: base class, Groq implementation, and mock."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

from .utils import setup_logger

logger = setup_logger(__name__)


class LLMProvider(ABC):
    """Abstract base class for all LLM providers.

    Subclasses must implement :meth:`complete`.

    Attributes:
        name: Human-readable provider identifier.
    """

    name: str = "abstract"

    @abstractmethod
    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        """Send a prompt and return the model's text response.

        Args:
            prompt: The user-turn message.
            system: Optional system prompt.

        Returns:
            The model's text response as a plain string.
        """


class GroqProvider(LLMProvider):
    """LLM provider backed by the Groq cloud inference API.

    Groq offers a **free tier** with no credit card required:

    * 1,000 requests / day
    * 30 requests / minute (RPM)
    * 6,000 tokens / minute (TPM)

    Sign up and get your key at https://console.groq.com

    Recommended models:

    * ``llama-3.3-70b-versatile`` (**default** — best quality)
    * ``llama-3.1-8b-instant`` — faster, smaller
    * ``gemma2-9b-it`` — alternative

    Args:
        model: Groq model ID (default: ``"llama-3.3-70b-versatile"``).
        api_key: Groq API key. Falls back to ``GROQ_API_KEY`` env variable.
        max_tokens: Maximum tokens in the response (default: ``1024``).
        temperature: Sampling temperature (default: ``0.2``).

    Raises:
        ImportError: If the ``groq`` package is not installed.
        ValueError: If no API key is found.
    """

    name: str = "groq"

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> None:
        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError(
                "The 'groq' package is required for GroqProvider. "
                "Install it with: pip install groq"
            ) from exc

        resolved_key = api_key or os.getenv("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Groq API key provided. Pass api_key= or set the "
                "GROQ_API_KEY environment variable. "
                "Get a free key (no credit card) at https://console.groq.com"
            )

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = Groq(api_key=resolved_key)

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        """Call the Groq API and return the assistant's response text.

        Args:
            prompt: User message.
            system: Optional system message prepended to the conversation.

        Returns:
            Response text from the model.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content


class MockProvider(LLMProvider):
    """Deterministic mock LLM provider for unit tests and offline demos.

    Always returns a fixed canned response regardless of the prompt.

    Args:
        canned_response: The string to return from every :meth:`complete`
            call. Defaults to a minimal valid JSON payload.
    """

    name: str = "mock"

    _DEFAULT_RESPONSE = (
        '{"semantic_meaning": "Mock column", '
        '"suggested_features": ["Mock feature 1", "Mock feature 2"], '
        '"domain_hints": []}'
    )

    def __init__(self, canned_response: Optional[str] = None) -> None:
        self._response = canned_response if canned_response is not None else self._DEFAULT_RESPONSE

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        """Return the canned response, ignoring the actual prompt.

        Args:
            prompt: Ignored.
            system: Ignored.

        Returns:
            The canned response string set at construction time.
        """
        return self._response
