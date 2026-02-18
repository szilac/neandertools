"""Custom errors for neandertools."""


class MissingDependencyError(RuntimeError):
    """Raised when an optional dependency is required but unavailable."""
