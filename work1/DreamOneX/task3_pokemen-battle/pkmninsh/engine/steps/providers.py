"""Built-in step providers used by the default action assembler."""

from __future__ import annotations

from typing import List

from pkmninsh.engine.pipeline import ActionContext
from pkmninsh.engine.steps.catalog import Step, StepCatalog, StepProvider
from pkmninsh.engine.steps.effects_dispatch import effects_dispatch

__all__ = [
    "ResolveTargetProvider",
    "EffectsPreProvider",
    "DamageChainProvider",
    "EffectsPostProvider",
    "register_default_providers",
]


def _resolve_targets(ctx: ActionContext) -> None:
    candidates = list(ctx.get("candidates", ()))
    ctx.setdefault("targets", [])
    if candidates:
        ctx["targets"] = candidates


def _run_pre_effects(_: ActionContext) -> None:
    """Placeholder for pre-resolution effects.

    The current milestone does not implement concrete behaviour here; the hook
    exists so that future providers can insert their own logic without touching
    the core pipeline structure.
    """


def _run_post_effects(ctx: ActionContext) -> None:
    _ = ctx  # Placeholder for future post-processing hooks


class ResolveTargetProvider(StepProvider):
    namespace = "core.resolve_targets"

    def priority(self) -> int:
        return 10

    def match(self, _: ActionContext) -> bool:
        return True

    def steps(self, _: ActionContext) -> List[Step]:
        return [_resolve_targets]


class EffectsPreProvider(StepProvider):
    namespace = "core.effects.pre"

    def priority(self) -> int:
        return 20

    def match(self, ctx: ActionContext) -> bool:
        return bool(ctx["move"].get("effects"))

    def steps(self, _: ActionContext) -> List[Step]:
        return [_run_pre_effects]


class DamageChainProvider(StepProvider):
    namespace = "core.damage.chain"

    def priority(self) -> int:
        return 30

    def match(self, ctx: ActionContext) -> bool:
        return bool(ctx["move"].get("effects"))

    def steps(self, _: ActionContext) -> List[Step]:
        return [effects_dispatch]


class EffectsPostProvider(StepProvider):
    namespace = "core.effects.post"

    def priority(self) -> int:
        return 40

    def match(self, ctx: ActionContext) -> bool:
        return bool(ctx["move"].get("effects"))

    def steps(self, _: ActionContext) -> List[Step]:
        return [_run_post_effects]


def register_default_providers(catalog: StepCatalog) -> None:
    """Register built-in providers on the supplied catalog."""

    catalog.register(ResolveTargetProvider())
    catalog.register(EffectsPreProvider())
    catalog.register(DamageChainProvider())
    catalog.register(EffectsPostProvider())
