from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any, Mapping, Optional

from rich.console import Console
from rich.logging import RichHandler

from pkmninsh.infra.logging.types import LogLevel, LoggingConfig, Logger, _LogContextData


# Context variable to store the current logging context
_log_context: ContextVar[Optional[_LogContextData]] = ContextVar(
    "_log_context", default=None
)


class _RichLoggerAdapter(Logger):
    """Adapter that wraps a standard library logger to implement the Logger protocol."""

    def __init__(self, logger: logging.Logger) -> None:
        """Initializes a new logger adapter.

        Args:
            logger: The underlying standard library logger.
        """
        self._logger = logger

    @property
    def name(self) -> str:
        """Returns the logger name."""
        return self._logger.name

    def _log(
        self,
        level: int,
        message: str,
        extra: Optional[Mapping[str, Any]] = None,
        exc_info: Any = None,
    ) -> None:
        """Internal helper to log a message with context."""
        # Merge context data into extra fields
        merged_extra = dict(extra) if extra else {}

        # Add turn context if available
        context = _log_context.get()
        if context and context.turn is not None:
            merged_extra["turn"] = context.turn
        if context and context.extra:
            merged_extra.update(context.extra)

        self._logger.log(level, message, extra=merged_extra, exc_info=exc_info)

    def trace(
        self,
        message: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Logs a trace-level message."""
        # TRACE is not standard, use DEBUG-1
        self._log(logging.DEBUG - 1, message, extra)

    def debug(
        self,
        message: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Logs a debug-level message."""
        self._log(logging.DEBUG, message, extra)

    def info(
        self,
        message: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Logs an info-level message."""
        self._log(logging.INFO, message, extra)

    def warning(
        self,
        message: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Logs a warning-level message."""
        self._log(logging.WARNING, message, extra)

    def error(
        self,
        message: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Logs an error-level message."""
        self._log(logging.ERROR, message, extra)

    def critical(
        self,
        message: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Logs a critical-level message."""
        self._log(logging.CRITICAL, message, extra)

    def exception(
        self,
        message: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        exc: Optional[BaseException] = None,
    ) -> None:
        """Logs an error-level message including exception information."""
        # Use exc_info=True to capture current exception, or pass explicit exception
        exc_info = exc if exc is not None else True
        self._log(logging.ERROR, message, extra, exc_info=exc_info)


class RichLoggingBackend:
    """Logging backend that uses Rich for formatted console output."""

    _console: Console
    _loggers: dict[str, _RichLoggerAdapter]

    def __init__(self) -> None:
        """Initializes a new Rich logging backend."""
        # Create console with wider width to avoid message truncation
        self._console = Console(width=120, soft_wrap=True)
        self._loggers = {}

    def configure(self, config: LoggingConfig) -> None:
        """Configures the backend with the given settings.

        Args:
            config: High-level logging configuration to apply.
        """
        # Map our LogLevel to standard library log levels
        level_map = {
            LogLevel.TRACE: logging.DEBUG - 1,
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }

        # Get root logger and set level
        root_logger = logging.getLogger()
        root_logger.setLevel(level_map[config.level])

        # Clear existing handlers
        root_logger.handlers.clear()

        # Configure handler based on settings
        if config.enable_rich:
            # Use RichHandler for pretty console output
            handler = RichHandler(
                console=self._console,
                rich_tracebacks=True,
                show_time=True,
                show_level=True,
                show_path=True,
                markup=True,
                omit_repeated_times=False,
            )
        else:
            # Use standard StreamHandler
            stream = sys.stderr if config.log_to_stderr else sys.stdout
            handler = logging.StreamHandler(stream)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)

        handler.setLevel(level_map[config.level])
        root_logger.addHandler(handler)

        # Add file handler if configured
        if config.log_file_path:
            file_handler = logging.FileHandler(config.log_file_path)
            file_handler.setLevel(level_map[config.level])
            if config.enable_json:
                # For JSON logging, we could use a JSONFormatter here
                # For now, just use standard formatter
                formatter = logging.Formatter(
                    '{"time":"%(asctime)s","name":"%(name)s","level":"%(levelname)s","message":"%(message)s"}'
                )
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    def get_logger(self, name: str) -> Logger:
        """Returns a logger bound to this backend for the given name.

        Args:
            name: Logical logger name.

        Returns:
            A Logger instance backed by this backend.
        """
        if name not in self._loggers:
            std_logger = logging.getLogger(name)
            self._loggers[name] = _RichLoggerAdapter(std_logger)
        return self._loggers[name]

    def get_current_context(self) -> Optional[_LogContextData]:
        """Returns the current context stored by the backend, if any.

        Returns:
            The currently active _LogContextData, or None if no context is set.
        """
        return _log_context.get()

    def push_context(self, context: _LogContextData) -> None:
        """Pushes logging context onto the backend context stack.

        Args:
            context: Context data to make active.
        """
        _log_context.set(context)

    def pop_context(self) -> None:
        """Pops the most recent logging context from the backend context stack."""
        _log_context.set(None)
