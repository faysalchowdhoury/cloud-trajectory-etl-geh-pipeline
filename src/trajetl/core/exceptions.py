"""Domain exceptions."""

from __future__ import annotations


class PipelineError(Exception):
    def __init__(self, message: str, *, context: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, object] = dict(context or {})

    def __str__(self) -> str:
        if not self.context:
            return self.message
        return f"{self.message} ({', '.join(f'{k}={v!r}' for k, v in self.context.items())})"


class IngestError(PipelineError):
    """Download or raw-file problem."""


class DataQualityError(PipelineError):
    """Input violates schema or physical plausibility checks beyond repair."""


class ConfigurationError(PipelineError):
    """Invalid configuration."""
