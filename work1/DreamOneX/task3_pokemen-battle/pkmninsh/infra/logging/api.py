from __future__ import annotations

from typing import ContextManager, Mapping, Optional

from pkmninsh.infra.logging.types import LoggingConfig, Logger, LogLevel
from pkmninsh.infra.logging._manager import _LoggingManager
from pkmninsh.infra.logging._backend import _LoggingBackend


_default_manager: Optional[_LoggingManager] = None


def init_logging(
    backend: _LoggingBackend,
    config: Optional[LoggingConfig] = None,
) -> None:
    """Initializes the global logging manager and backend.

    This function should be called once at application startup.

    Args:
        backend: Backend to use for actual logging operations.
        config: Optional logging configuration. If None, a default
            configuration is used.
    """
    global _default_manager
    _default_manager = _LoggingManager(backend, config)


def _get_manager() -> _LoggingManager:
    """Returns the global logging manager.

    If not initialized, creates a default manager with RichLoggingBackend.

    Returns:
        The active _LoggingManager.
    """
    global _default_manager
    if _default_manager is None:
        # Auto-initialize with default backend
        from pkmninsh.infra.logging.rich_backend import RichLoggingBackend
        default_config = LoggingConfig(
            level=LogLevel.INFO,
            enable_rich=True,
        )
        _default_manager = _LoggingManager(RichLoggingBackend(), default_config)
    return _default_manager


def get_logger(name: str) -> Logger:
    """Returns a logger with the given logical name.

    Args:
        name: Logical name of the logger.

    Returns:
        A Logger instance.
    """
    return _get_manager().get_logger(name)


def get_plugin_logger(plugin_name: str) -> Logger:
    """Returns a plugin logger with a conventional name.

    Args:
        plugin_name: Name of the plugin requesting a logger.

    Returns:
        A Logger instance scoped to the given plugin.
    """
    return _get_manager().get_plugin_logger(plugin_name)


def get_current_turn() -> Optional[int]:
    """Gets the current turn from the active logging context, if any.

    Returns:
        The current turn index, or None if no turn is set.
    """
    return _get_manager().get_current_turn()


def with_turn(turn: int) -> ContextManager[None]:
    """Creates a logging context for the given turn using the global manager.

    Args:
        turn: Index of the current turn.

    Returns:
        A context manager that applies the turn context.
    """
    return _get_manager().with_turn(turn)


def log_domain_error(
    logger: Logger,
    error: BaseException,
    *,
    message: Optional[str] = None,
    extra: Optional[Mapping[str, object]] = None,
) -> None:
    """Logs a domain-level error using the global logging manager.

    Args:
        logger: Logger to use for emitting the record.
        error: Domain-level exception instance.
        message: Optional human-readable message.
        extra: Optional additional structured fields.
    """
    _get_manager().log_domain_error(logger, error, message=message, extra=extra)


def log_content_error(
    logger: Logger,
    error: BaseException,
    *,
    message: Optional[str] = None,
    plugin_name: Optional[str] = None,
    manifest_path: Optional[str] = None,
    extra: Optional[Mapping[str, object]] = None,
) -> None:
    """Logs a content or plugin-loading error using the global logging manager.

    Args:
        logger: Logger to use for emitting the record.
        error: Exception raised during content or plugin loading.
        message: Optional human-readable message.
        plugin_name: Optional name of the plugin involved.
        manifest_path: Optional path to the offending manifest or content file.
        extra: Optional additional structured fields.
    """
    _get_manager().log_content_error(
        logger,
        error,
        message=message,
        plugin_name=plugin_name,
        manifest_path=manifest_path,
        extra=extra,
    )

