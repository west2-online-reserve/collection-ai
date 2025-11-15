from __future__ import annotations

from typing import Any, ContextManager, Mapping, Optional

from pkmninsh.infra.logging.types import LoggingConfig, Logger, _LogContextData
from pkmninsh.infra.logging._backend import _LoggingBackend


class _TurnContext(ContextManager[None]):
    """Context manager that applies turn-related log context via a backend."""
    _backend: _LoggingBackend
    _context: _LogContextData

    def __init__(self, backend: _LoggingBackend, context: _LogContextData) -> None:
        """Initializes a new turn context manager.

        Args:
            backend: Backend used to push and pop context.
            context: Context data to apply for the duration of the scope.
        """
        self._backend = backend
        self._context = context

    def __enter__(self) -> None:
        """Enters the logging context scope."""
        self._backend.push_context(self._context)

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Any,
    ) -> None:
        """Exits the logging context scope.

        Args:
            exc_type: Exception type if an exception occurred, else None.
            exc: Exception instance if an exception occurred, else None.
            tb: Traceback object if an exception occurred, else None.
        """
        self._backend.pop_context()


class _LoggingManager:
    """Internal manager for logging and turn context handling."""
    _backend: _LoggingBackend
    _config: LoggingConfig

    def __init__(
        self,
        backend: _LoggingBackend,
        config: Optional[LoggingConfig] = None,
    ) -> None:
        """Initializes a new _LoggingManager.

        Args:
            backend: Logging backend implementation to use.
            config: Optional initial logging configuration.
        """
        self._backend = backend
        self._config = config if config else LoggingConfig()
        # Configure the backend with initial config
        self._backend.configure(self._config)

    @property
    def backend(self) -> _LoggingBackend:
        """Returns the underlying logging backend."""
        return self._backend

    @property
    def config(self) -> LoggingConfig:
        """Returns the current effective logging configuration."""
        return self._config

    def configure(self, config: LoggingConfig) -> None:
        """Configures the manager and underlying backend.

        Args:
            config: Logging configuration to apply.
        """
        self._config = config
        self._backend.configure(config)

    def get_logger(self, name: str) -> Logger:
        """Returns a logger with the given logical name.

        Args:
            name: Logical name of the logger.

        Returns:
            A Logger instance implementing the Logger protocol.
        """
        return self._backend.get_logger(name)

    def get_plugin_logger(self, plugin_name: str) -> Logger:
        """Returns a logger for a specific plugin.

        Args:
            plugin_name: Name of the plugin requesting a logger.

        Returns:
            A Logger instance scoped to the given plugin.
        """
        # Convention: plugin loggers use "plugin.<name>" naming
        logger_name = f"plugin.{plugin_name}"
        return self._backend.get_logger(logger_name)

    def get_current_context(self) -> Optional[_LogContextData]:
        """Gets the current effective _LogContextData for the running scope.

        Returns:
            The active _LogContextData, or None if no logging context is active.
        """
        return self._backend.get_current_context()

    def get_current_turn(self) -> Optional[int]:
        """Gets the current turn from the active logging context, if any.

        Returns:
            The current turn index, or None if no turn is set.
        """
        context = self.get_current_context()
        return context.turn if context else None

    def with_turn(self, turn: int) -> ContextManager[None]:
        """Creates a context manager that applies the given turn for a scope.

        Args:
            turn: Index of the current turn.

        Returns:
            A context manager that applies the turn context.
        """
        context = _LogContextData(turn=turn)
        return _TurnContext(self._backend, context)

    def log_domain_error(
        self,
        logger: Logger,
        error: BaseException,
        *,
        message: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Logs a domain-level error in a consistent way.

        Args:
            logger: Logger to use for emitting the record.
            error: Domain-level exception instance.
            message: Optional human-readable message.
            extra: Optional additional structured fields.
        """
        # Construct the log message
        msg = message if message else f"Domain error: {type(error).__name__}"

        # Merge extra fields
        merged_extra = dict(extra) if extra else {}
        merged_extra["error_type"] = type(error).__name__
        merged_extra["error_message"] = str(error)

        # Log the error with exception info
        logger.exception(msg, extra=merged_extra, exc=error)

    def log_content_error(
        self,
        logger: Logger,
        error: BaseException,
        *,
        message: Optional[str] = None,
        plugin_name: Optional[str] = None,
        manifest_path: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Logs a content or plugin-loading error in a consistent way.

        Args:
            logger: Logger to use for emitting the record.
            error: Exception raised during content or plugin loading.
            message: Optional human-readable message.
            plugin_name: Optional name of the plugin involved.
            manifest_path: Optional path to the offending manifest or content file.
            extra: Optional additional structured fields.
        """
        # Construct the log message
        msg = message if message else f"Content error: {type(error).__name__}"

        # Merge extra fields
        merged_extra = dict(extra) if extra else {}
        merged_extra["error_type"] = type(error).__name__
        merged_extra["error_message"] = str(error)

        if plugin_name:
            merged_extra["plugin_name"] = plugin_name
        if manifest_path:
            merged_extra["manifest_path"] = manifest_path

        # Log the error with exception info
        logger.exception(msg, extra=merged_extra, exc=error)
