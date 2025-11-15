from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional


@dataclass(slots=True)
class I18nConfig:
    '''Global internationalization configuration.

    This class is a configuration carrier only and does not contain any logic.

    Attributes:
        default_locale: Default locale to use when none is specified.
        fallback_locale: Fallback locale if a translation is missing.
        locales_path: Base directory containing translation files.
        additional_locales_paths: Additional directories to search for translations.
            Translations from these paths override those from locales_path.
        domain: Primary translation domain for the application.
        available_locales: List of supported locales.
        auto_reload: Whether to automatically reload translations on change.
        interpolation_defaults: Default variables for string interpolation.
    '''

    default_locale: str = "en"
    fallback_locale: Optional[str] = None
    locales_path: str = "locales"
    additional_locales_paths: list[str] = field(default_factory=list)
    domain: str = "messages"
    available_locales: list[str] = field(default_factory=list)
    auto_reload: bool = False
    interpolation_defaults: MutableMapping[str, Any] = field(default_factory=dict)
