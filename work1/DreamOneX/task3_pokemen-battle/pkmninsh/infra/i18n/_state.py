from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ContextManager, Mapping, MutableMapping, Optional

from pkmninsh.infra.i18n.types import I18nConfig


class _LocaleContext(ContextManager[None]):
    '''Context manager that temporarily overrides the active locale.'''

    _state: _I18nState
    _locale: str
    _previous_locale: Optional[str]

    def __init__(self, state: "_I18nState", locale: str) -> None:
        '''Initializes a new locale context manager.

        Args:
            state: Global i18n state instance.
            locale: Locale identifier to activate within this scope.
        '''
        self._state = state
        self._locale = locale
        self._previous_locale = None

    def __enter__(self) -> None:
        '''Enters the locale override scope.'''
        self._previous_locale = self._state.get_locale()
        self._state.set_locale(self._locale)

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Any,
    ) -> None:
        '''Exits the locale override scope.

        Args:
            exc_type: Exception type if an exception occurred, else None.
            exc: Exception instance if an exception occurred, else None.
            tb: Traceback object if an exception occurred, else None.
        '''
        if self._previous_locale is not None:
            self._state.set_locale(self._previous_locale)


class _I18nState:
    '''Internal state holder for internationalization.

    This class keeps track of configuration and the currently active locale.
    It also provides helper methods for translating messages.
    '''

    _config: Optional[I18nConfig]
    _current_locale: Optional[str]
    _translations_cache: MutableMapping[str, MutableMapping[str, str]]

    def __init__(self, config: Optional[I18nConfig] = None) -> None:
        '''Initializes a new _I18nState.

        Args:
            config: Optional initial i18n configuration.
        '''
        self._config = config
        self._current_locale = config.default_locale if config else "en"
        self._translations_cache = {}

    @property
    def config(self) -> Optional[I18nConfig]:
        '''Returns the current i18n configuration, if any.'''
        return self._config

    def configure(self, config: I18nConfig) -> None:
        '''Configures the i18n state.

        Args:
            config: Internationalization configuration to apply.
        '''
        self._config = config
        self._current_locale = config.default_locale
        # Clear cache when reconfiguring
        self._translations_cache.clear()

    def get_locale(self) -> str:
        '''Returns the currently active locale.

        Returns:
            The locale identifier that will be used for translations.
        '''
        return self._current_locale or "en"

    def set_locale(self, locale: str) -> None:
        '''Sets the currently active locale.

        Args:
            locale: Locale identifier to activate.
        '''
        self._current_locale = locale

    def with_locale(self, locale: str) -> ContextManager[None]:
        '''Creates a context manager that temporarily overrides the locale.

        Args:
            locale: Locale identifier to activate within the context.

        Returns:
            A context manager that applies the locale override.
        '''
        return _LocaleContext(self, locale)

    def _load_translations(self, locale: str) -> MutableMapping[str, str]:
        '''Loads translations for the given locale from disk.

        Loads from the main locales_path first, then overlays translations
        from additional_locales_paths (if configured), allowing plugins to
        override built-in translations.

        Args:
            locale: Locale identifier to load.

        Returns:
            Mapping of translation keys to translated strings.
        '''
        # Check cache first
        if locale in self._translations_cache:
            return self._translations_cache[locale]

        translations: MutableMapping[str, str] = {}

        if not self._config:
            self._translations_cache[locale] = translations
            return translations

        # Load from main locales path first
        locales_path = Path(self._config.locales_path)
        translation_file = locales_path / f"{locale}.properties"

        if translation_file.exists():
            self._load_properties_file(translation_file, translations)

        # Load and overlay translations from additional paths
        for additional_path in self._config.additional_locales_paths:
            additional_file = Path(additional_path) / f"{locale}.properties"
            if additional_file.exists():
                self._load_properties_file(additional_file, translations)

        self._translations_cache[locale] = translations
        return translations

    def _load_properties_file(
        self, file_path: Path, translations: MutableMapping[str, str]
    ) -> None:
        '''Load translations from a .properties file into the given dict.

        Args:
            file_path: Path to the .properties file
            translations: Dictionary to update with loaded translations
        '''
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    # Parse key=value pairs
                    if '=' in line:
                        key, _, value = line.partition('=')
                        translations[key.strip()] = value.strip()
        except Exception:
            # If loading fails, just skip this file
            pass

    def translate(
        self,
        key: str,
        *,
        locale: Optional[str] = None,
        variables: Optional[Mapping[str, Any]] = None,
    ) -> str:
        '''Resolves and formats a translation for the given key.

        Args:
            key: Translation key or message identifier.
            locale: Optional locale override. If None, the current locale is used.
            variables: Optional mapping of interpolation variables.

        Returns:
            The translated and formatted string.
        '''
        # Determine effective locale
        effective_locale = locale if locale is not None else self.get_locale()

        # Load translations for the locale
        translations = self._load_translations(effective_locale)

        # Try to get the translation
        message = translations.get(key)

        # If not found and we have a fallback locale, try that
        if message is None and self._config and self._config.fallback_locale:
            fallback_translations = self._load_translations(self._config.fallback_locale)
            message = fallback_translations.get(key)

        # If still not found, return the key itself
        if message is None:
            message = key

        # Apply variable interpolation if provided
        if variables:
            # Merge with interpolation defaults if configured
            effective_vars = dict(self._config.interpolation_defaults) if self._config else {}
            effective_vars.update(variables)

            # Simple string formatting using {var_name} syntax
            try:
                message = message.format(**effective_vars)
            except (KeyError, ValueError):
                # If formatting fails, return the unformatted message
                pass

        return message
