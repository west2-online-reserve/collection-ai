"""Application initialization module for pkmninsh.

This module handles initialization of infrastructure components including
internationalization (i18n) and logging.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pkmninsh.infra.i18n import init_i18n, set_locale, translate
from pkmninsh.infra.i18n.types import I18nConfig
from pkmninsh.infra.logging import init_logging
from pkmninsh.infra.logging.rich_backend import RichLoggingBackend
from pkmninsh.infra.logging.types import LoggingConfig, LogLevel


def _detect_language(preferred: Optional[str] = None) -> str:
    """Detect language from environment or preferences.

    Args:
        preferred: Optional explicit language preference

    Returns:
        Detected language code (e.g., 'en', 'zh_cn')
    """
    # Normalize language code mapping
    canonical_mapping = {
        "zh": "zh_cn",
        "zh-cn": "zh_cn",
        "zh_cn": "zh_cn",
        "zh-hans": "zh_cn",
        "en": "en",
    }

    candidates = []

    # Add explicit preference
    if preferred:
        candidates.append(preferred)

    # Check environment variables
    for env_var in ("PKMNNISH_LANG", "LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.getenv(env_var)
        if value:
            candidates.append(value)

    # Normalize and return first valid candidate
    for candidate in candidates:
        # Normalize: lowercase, strip encoding
        normalized = candidate.lower().strip()
        normalized = normalized.split(".")[0]  # Remove encoding like .UTF-8
        normalized = normalized.replace("-", "_")

        if normalized in canonical_mapping:
            return canonical_mapping[normalized]

    # Default to zh_cn
    return "zh_cn"


def init_app(
    lang: Optional[str] = None,
    log_level: LogLevel = LogLevel.INFO,
    plugin_dirs: Optional[list[str]] = None,
) -> str:
    """Initialize application infrastructure.

    Sets up i18n and logging systems.

    Args:
        lang: Optional language code. If None, auto-detect from environment
        log_level: Logging level to use
        plugin_dirs: Optional list of plugin directories to scan for locales

    Returns:
        The language code that was set
    """
    # Initialize i18n
    infra_locales_path = Path(__file__).parent / "infra" / "locales"

    # Scan for plugin locales
    additional_locales = []
    if plugin_dirs:
        for plugin_dir in plugin_dirs:
            plugin_path = Path(plugin_dir)
            if plugin_path.exists() and plugin_path.is_dir():
                for subdir in plugin_path.iterdir():
                    if subdir.is_dir():
                        locale_dir = subdir / "locales"
                        if locale_dir.exists() and locale_dir.is_dir():
                            additional_locales.append(str(locale_dir))

    i18n_config = I18nConfig(
        default_locale="zh_cn",
        fallback_locale="en",
        locales_path=str(infra_locales_path),
        additional_locales_paths=additional_locales,
        domain="messages",
        available_locales=["en", "zh_cn"],
    )
    init_i18n(i18n_config)

    # Detect and set language
    detected_lang = _detect_language(lang)
    set_locale(detected_lang)

    # Initialize logging
    logging_config = LoggingConfig(
        level=log_level,
        enable_rich=True,
    )
    backend = RichLoggingBackend()
    init_logging(backend, logging_config)

    return detected_lang


__all__ = ["init_app", "translate"]
