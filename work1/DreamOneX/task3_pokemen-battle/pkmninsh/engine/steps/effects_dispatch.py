from __future__ import annotations

import logging
from typing import Any, List, Mapping, Optional

from pkmninsh.engine.effects.registry import EffectRegistry, effect_registry as global_effect_registry
from pkmninsh.engine.pipeline import ActionContext


logger = logging.getLogger(__name__)


def _resolve_registry(ctx: ActionContext) -> EffectRegistry:
    ext = ctx.setdefault("ext", {})
    registry = ext.get("effect_registry")
    if isinstance(registry, EffectRegistry):
        return registry
    if registry is not None:
        logger.warning("Invalid effect registry override %r — falling back to global", registry)
    return global_effect_registry


def effects_dispatch(ctx: ActionContext) -> None:
    if ctx.get("abort"):
        return
    move = ctx.get("move", {})
    effects: List[Mapping[str, Any]] = list(move.get("effects", []) or [])
    if not effects:
        return
    registry = _resolve_registry(ctx)
    handlers = registry.resolve(effects, ctx)
    move_key = str(move.get("key") or move.get("name") or "<unknown>")
    for index, effect in enumerate(effects):
        if ctx.get("abort"):
            break
        handler: Optional[Any]
        if index < len(handlers):
            handler = handlers[index]
        else:
            handler = None
        if handler is None:
            continue
        try:
            handler.handle(effect, ctx)
        except Exception:  # pragma: no cover - defensive guard
            logger.exception(
                "Effect handler '%s' raised during move '%s'",
                getattr(handler, "namespace", repr(handler)),
                move_key,
            )
