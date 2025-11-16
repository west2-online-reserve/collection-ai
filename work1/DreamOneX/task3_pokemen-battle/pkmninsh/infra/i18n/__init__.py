from __future__ import annotations

from pkmninsh.infra.i18n.types import I18nConfig
from pkmninsh.infra.i18n.api import (
    init_i18n,
    get_locale,
    set_locale,
    with_locale,
    translate,
)


__all__ = [
    "I18nConfig",
    "init_i18n",
    "get_locale",
    "set_locale",
    "with_locale",
    "translate",
]
