from __future__ import annotations

from typing import ContextManager, Mapping, Optional

from pkmninsh.infra.i18n.types import I18nConfig
from pkmninsh.infra.i18n._state import _I18nState


_state: Optional[_I18nState] = None


def init_i18n(config: Optional[I18nConfig] = None) -> None:
    '''Initializes the global i18n state.

    This function should be called once at application startup.

    Args:
        config: Optional i18n configuration. If None, a default configuration
            is used.
    '''
    global _state
    _state = _I18nState(config)


def _get_state() -> _I18nState:
    '''Returns the global i18n state.

    Returns:
        The active _I18nState.

    Raises:
        RuntimeError: If i18n has not been initialized yet.
    '''
    if _state is None:
        raise RuntimeError(
            "i18n has not been initialized. Call init_i18n() first."
        )
    return _state


def get_locale() -> str:
    '''Returns the currently active locale.

    Returns:
        The locale identifier that will be used for translations.
    '''
    return _get_state().get_locale()


def set_locale(locale: str) -> None:
    '''Sets the currently active locale.

    Args:
        locale: Locale identifier to activate.
    '''
    _get_state().set_locale(locale)


def with_locale(locale: str) -> ContextManager[None]:
    '''Creates a locale override context using the global state.

    Args:
        locale: Locale identifier to activate within the context.

    Returns:
        A context manager that applies the locale override.
    '''
    return _get_state().with_locale(locale)


def translate(
    key: str,
    *,
    locale: Optional[str] = None,
    variables: Optional[Mapping[str, object]] = None,
) -> str:
    '''Translates a message key using the global i18n state.

    Args:
        key: Translation key or message identifier.
        locale: Optional locale override.
        variables: Optional mapping of interpolation variables.

    Returns:
        The translated and formatted string.
    '''
    return _get_state().translate(key, locale=locale, variables=variables)

