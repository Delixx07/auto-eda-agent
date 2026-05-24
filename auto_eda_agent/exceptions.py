"""Custom exception hierarchy for auto_eda_agent."""


class AutoEDAError(Exception):
    """Base exception for all auto_eda_agent errors.

    Catch this to handle any library error uniformly:

        try:
            agent.run_full_pipeline()
        except AutoEDAError as e:
            print(f"EDA failed: {e}")
    """


class InvalidDataError(AutoEDAError):
    """Raised when the input data is invalid, unreadable, or unsupported.

    Examples:
        - Passing None or a non-DataFrame type with no file path.
        - Providing a file path that does not exist.
        - Providing an empty DataFrame when allow_empty=False.
        - Invalid threshold values (outside [0, 1]).
    """


class ColumnNotFoundError(AutoEDAError):
    """Raised when a referenced column does not exist in the DataFrame.

    Attributes:
        column: The missing column name.
        available: List of column names that are present.
    """

    def __init__(self, column: str, available: list[str] | None = None) -> None:
        self.column = column
        self.available = available or []
        available_str = ", ".join(self.available[:10])
        if len(self.available) > 10:
            available_str += f" … ({len(self.available)} total)"
        msg = f"Column '{column}' not found."
        if self.available:
            msg += f" Available: {available_str}"
        super().__init__(msg)


class PipelineError(AutoEDAError):
    """Raised when the full EDA pipeline fails at a pipeline stage.

    This wraps the underlying exception so callers can distinguish
    pipeline-level failures from individual component errors.
    """
