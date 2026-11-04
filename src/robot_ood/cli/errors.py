"""The one exception every subcommand fails through, carrying its own exit code."""

from __future__ import annotations

__all__ = ["CLIError"]


class CLIError(Exception):
    """Raised by any subcommand to fail cleanly.

    Attributes:
        exit_code: What the process should exit with. 2 means a pre-registered gate failed.
    """

    def __init__(self, message: str, exit_code: int = 1):
        """Bind the message and the exit code.

        Args:
            message: What went wrong.
            exit_code: Process exit code; 1 is a plain failure.
        """
        super().__init__(message)
        self.exit_code = exit_code
