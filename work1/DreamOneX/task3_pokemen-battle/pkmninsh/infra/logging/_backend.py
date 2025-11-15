from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from pkmninsh.infra.logging.types import LoggingConfig, Logger, _LogContextData


@runtime_checkable
class _LoggingBackend(Protocol):
    """Internal backend interface that bridges to a concrete logging library.

    Implementations are responsible for configuring the underlying logging
    library and returning Logger instances bound to that library. They may
    also manage turn-related logging context.
    """
    def configure(self, config: LoggingConfig) -> None:
        """Configures the backend with the given settings.

        Args:
            config: High-level logging configuration to apply.
        """
        ...

    def get_logger(self, name: str) -> Logger:
        """Returns a logger bound to this backend for the given name.

        Args:
            name: Logical logger name.

        Returns:
            A Logger instance backed by this backend.
        """
        ...

    def get_current_context(self) -> Optional[_LogContextData]:
        """Returns the current context stored by the backend, if any.

        Returns:
            The currently active _LogContextData, or None if no context is set.
        """
        ...

    def push_context(self, context: _LogContextData) -> None:
        """Pushes logging context onto the backend context stack.

        Args:
            context: Context data to make active.
        """
        ...

    def pop_context(self) -> None:
        """Pops the most recent logging context from the backend context stack.

        If no context is active, implementations may ignore the call or raise
        an internal error depending on their design.
        """
        ...
