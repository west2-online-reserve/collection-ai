from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from pkmninsh.engine.effects.handlers import BUILTIN_HANDLERS, EffectHandler, EffectSpec
from pkmninsh.engine.pipeline import ActionContext


__all__ = [
    "EffectHandler",
    "EffectRegistry",
    "EffectSpec",
    "effect_registry",
]


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _RegisteredHandler:
    handler: EffectHandler
    order: int


class EffectRegistry:
    """Registry responsible for resolving effect handlers."""

    def __init__(self, handlers: Iterable[EffectHandler] | None = None) -> None:
        self._handlers: list[_RegisteredHandler] = []
        self._namespace_index: dict[str, int] = {}
        if handlers:
            for handler in handlers:
                self.register(handler)

    def register(self, handler: EffectHandler, *, replace: bool = False) -> None:
        namespace = handler.namespace
        if namespace in self._namespace_index:
            if not replace:
                raise ValueError(f"Handler namespace '{namespace}' is already registered")
            index = self._namespace_index[namespace]
            self._handlers[index] = _RegisteredHandler(handler=handler, order=index)
            return

        if replace:
            raise ValueError(f"Cannot replace handler '{namespace}' because it is not registered")

        index = len(self._handlers)
        self._handlers.append(_RegisteredHandler(handler=handler, order=index))
        self._namespace_index[namespace] = index

    def unregister(self, namespace: str) -> None:
        if namespace not in self._namespace_index:
            return
        index = self._namespace_index.pop(namespace)
        self._handlers.pop(index)
        # Rebuild index to keep ordering consistent
        self._namespace_index = {
            entry.handler.namespace: i for i, entry in enumerate(self._handlers)
        }
        for i, entry in enumerate(self._handlers):
            entry.order = i

    def resolve(
        self, effects: Sequence[EffectSpec], ctx: ActionContext
    ) -> List[Optional[EffectHandler]]:
        resolved: list[Optional[EffectHandler]] = []
        move = ctx.get("move", {}) if isinstance(ctx, dict) else {}
        move_key = "<unknown>"
        if isinstance(move, dict):
            move_key = str(move.get("key") or move.get("name") or "<unknown>")

        for effect in effects:
            matches: list[tuple[int, int, EffectHandler]] = []
            for entry in self._handlers:
                handler = entry.handler
                try:
                    if handler.can_handle(effect, ctx):
                        handler_order = getattr(handler, "order", entry.order)
                        matches.append((handler_order, entry.order, handler))
                except Exception:  # pragma: no cover - defensive guard
                    logger.exception(
                        "Effect handler '%s' failed during can_handle for move '%s'",
                        getattr(handler, "namespace", repr(handler)),
                        move_key,
                    )

            if not matches:
                effect_type = str(effect.get("type", "<unknown>"))
                logger.warning(
                    "Unknown effect type '%s' in move '%s' — skipped",
                    effect_type,
                    move_key,
                )
                resolved.append(None)
                continue

            matches.sort(key=lambda item: (item[0], item[1]))
            best = matches[0]
            if len(matches) > 1:
                effect_type = str(effect.get("type", "<unknown>"))
                logger.warning(
                    "Multiple handlers matched effect '%s' in move '%s'; using %s",
                    effect_type,
                    move_key,
                    best[2].namespace,
                )
            resolved.append(best[2])

        return resolved

    def clear(self) -> None:
        self._handlers.clear()
        self._namespace_index.clear()

    def clone(self) -> "EffectRegistry":
        """Return a shallow copy preserving handler registration order."""

        return EffectRegistry(entry.handler for entry in self._handlers)


effect_registry: EffectRegistry = EffectRegistry(BUILTIN_HANDLERS)
