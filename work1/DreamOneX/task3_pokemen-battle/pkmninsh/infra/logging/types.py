from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, MutableMapping, Optional, Protocol, runtime_checkable


class LogLevel(str, Enum):
    """Log severity levels."""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass(slots=True)
class LoggingConfig:
    """Global logging configuration.

    This class is a configuration carrier only and does not contain any logic.

    Attributes:
        level: Global log level for the application.
        enable_rich: Whether to enable rich/pretty console output.
        enable_json: Whether to emit JSON/structured logs.
        log_to_stderr: Whether to log to stderr (otherwise stdout or file only).
        log_file_path: Optional file path for file logging.
        plugin_log_level: Optional log level override for plugin loggers.
        extra_defaults: Default extra fields attached to all log records.
    """
    level: LogLevel = LogLevel.INFO
    enable_rich: bool = True
    enable_json: bool = False
    log_to_stderr: bool = True
    log_file_path: Optional[str] = None
    plugin_log_level: Optional[LogLevel] = None
    extra_defaults: MutableMapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Logger(Protocol):
    """Abstract logger interface used by application and plugins."""
    @property
    def name(self) -> str:
        """Returns the logger name."""
    def trace(
        self,
        message: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Logs a trace-level message.

        Args:
            message: Human-readable log message.
            extra: Optional structured fields to attach to the log record.
        """
        ...

    def debug(
        self,
        message: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Logs a debug-level message.

        Args:
            message: Human-readable log message.
            extra: Optional structured fields to attach to the log record.
        """
        ...

    def info(
        self,
        message: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Logs an info-level message.

        Args:
            message: Human-readable log message.
            extra: Optional structured fields to attach to the log record.
        """
        ...

    def warning(
        self,
        message: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Logs a warning-level message.

        Args:
            message: Human-readable log message.
            extra: Optional structured fields to attach to the log record.
        """
        ...

    def error(
        self,
        message: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Logs an error-level message.

        Args:
            message: Human-readable log message.
            extra: Optional structured fields to attach to the log record.
        """
        ...

    def critical(
        self,
        message: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Logs a critical-level message.

        Args:
            message: Human-readable log message.
            extra: Optional structured fields to attach to the log record.
        """
        ...

    def exception(
        self,
        message: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        exc: Optional[BaseException] = None,
    ) -> None:
        """Logs an error-level message including exception information.

        Args:
            message: Human-readable log message.
            extra: Optional structured fields to attach to the log record.
            exc: Optional explicit exception to log. If omitted, the
                implementation may choose to use the current exception
                information from the runtime.
        """
        ...


@dataclass(slots=True)
class _LogContextData:
    """Internal context data attached to log records for a given scope.

    Currently only contains the dynamic turn identifier.

    Attributes:
        turn: Current turn or round index, if any.
        extra: Arbitrary additional context fields.
    """
    turn: Optional[int] = None
    extra: MutableMapping[str, Any] = field(default_factory=dict)
