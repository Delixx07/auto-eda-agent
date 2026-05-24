"""Utility helpers: logger factory and DataFrame validation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Union

import pandas as pd

from .exceptions import InvalidDataError

_SupportedInput = Union[pd.DataFrame, str, os.PathLike, dict, list]

_FILE_READERS = {
    ".csv": pd.read_csv,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
    ".parquet": pd.read_parquet,
    ".json": pd.read_json,
}


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve an idempotent logger with a single StreamHandler.

    Calling this function multiple times with the same ``name`` returns the
    same logger without stacking additional handlers.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level (default: ``logging.INFO``).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def validate_dataframe(
    data: _SupportedInput,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Validate and coerce *data* into a :class:`pandas.DataFrame`.

    Accepted input types:

    * :class:`pandas.DataFrame` — returned as-is (copy not made).
    * ``str`` / :class:`pathlib.Path` — loaded from ``.csv``, ``.xlsx``,
      ``.xls``, ``.parquet``, or ``.json``.
    * ``dict`` — passed to :func:`pandas.DataFrame`.
    * ``list`` — passed to :func:`pandas.DataFrame`.

    Args:
        data: The data source to validate.
        allow_empty: If ``False`` (default), an empty DataFrame raises
            :class:`~auto_eda_agent.exceptions.InvalidDataError`.

    Returns:
        A validated :class:`pandas.DataFrame`.

    Raises:
        InvalidDataError: If *data* is ``None``, an unsupported type, a
            non-existent file path, an unsupported file extension, or an
            empty DataFrame when *allow_empty* is ``False``.
    """
    if data is None:
        raise InvalidDataError("Input data is None. Expected a DataFrame, file path, dict, or list.")

    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, (str, os.PathLike, Path)):
        path = Path(data)
        if not path.exists():
            raise InvalidDataError(f"File not found: '{path}'")
        suffix = path.suffix.lower()
        reader = _FILE_READERS.get(suffix)
        if reader is None:
            supported = ", ".join(_FILE_READERS.keys())
            raise InvalidDataError(
                f"Unsupported file extension '{suffix}'. "
                f"Supported formats: {supported}"
            )
        try:
            df = reader(path)
        except Exception as exc:
            raise InvalidDataError(f"Failed to read file '{path}': {exc}") from exc
    elif isinstance(data, dict):
        try:
            df = pd.DataFrame(data)
        except Exception as exc:
            raise InvalidDataError(f"Cannot convert dict to DataFrame: {exc}") from exc
    elif isinstance(data, list):
        try:
            df = pd.DataFrame(data)
        except Exception as exc:
            raise InvalidDataError(f"Cannot convert list to DataFrame: {exc}") from exc
    else:
        raise InvalidDataError(
            f"Unsupported data type: {type(data).__name__}. "
            "Expected: pd.DataFrame, file path (str/Path), dict, or list."
        )

    if not allow_empty and df.empty:
        raise InvalidDataError("DataFrame is empty (0 rows or 0 columns).")

    return df
