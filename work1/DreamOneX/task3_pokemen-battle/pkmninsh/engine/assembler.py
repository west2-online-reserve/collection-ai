"""Action assembler responsible for context and pipeline creation."""

from __future__ import annotations

from typing import Callable, Protocol

from pkmninsh.engine.events import EventBus
from pkmninsh.engine.effects.registry import EffectRegistry, effect_registry as global_effect_registry
from pkmninsh.engine.ops import Ops
from pkmninsh.engine.pipeline import ActionContext, ActionPipeline
from pkmninsh.engine.registries import move_registry
from pkmninsh.engine.rng import RNG
from pkmninsh.engine.steps.catalog import StepCatalog
from pkmninsh.engine.steps.effects_dispatch import effects_dispatch
from pkmninsh.engine.turns import Action, TurnManager

__all__ = ["ActionAssembler", "DefaultAssembler"]


class ActionAssembler(Protocol):
    """Protocol describing how an action is assembled for execution."""

    def build_context(self, action: Action) -> ActionContext: ...

    def build_pipeline(self, ctx: ActionContext) -> ActionPipeline: ...


class DefaultAssembler:
    """Default implementation that relies on a :class:`StepCatalog`."""

    def __init__(
        self,
        catalog: StepCatalog,
        *,
        effect_registry: EffectRegistry | None = None,
    ) -> None:
        self._catalog = catalog
        self._bus: EventBus | None = None
        self._rng: RNG | None = None
        self._tm: TurnManager | None = None
        self._log: Callable[[str], None] | None = None
        base_registry = effect_registry or global_effect_registry.clone()
        self._effect_registry = base_registry
        self._ops: Ops | None = None

    @property
    def catalog(self) -> StepCatalog:
        """Return the underlying catalog used by the assembler."""

        return self._catalog

    @property
    def effect_registry(self) -> EffectRegistry:
        """Expose the effect registry used by this assembler."""

        return self._effect_registry

    def configure(
        self,
        *,
        bus: EventBus,
        rng: RNG,
        tm: TurnManager,
        log: Callable[[str], None],
    ) -> None:
        """Bind runtime dependencies required to assemble contexts."""

        self._bus = bus
        self._rng = rng
        self._tm = tm
        self._log = log

    def use_ops(self, ops: Ops) -> None:
        """Reuse a shared :class:`Ops` facade for assembled contexts."""

        self._ops = ops

    def build_context(self, action: Action) -> ActionContext:
        if self._bus is None or self._rng is None or self._tm is None or self._log is None:
            raise RuntimeError("DefaultAssembler is not configured")

        if not move_registry.has(action.move_key):
            raise KeyError(f"Move '{action.move_key}' not found in registry")

        move = move_registry.get(action.move_key)

        ctx: ActionContext = {
            "actor": action.actor,
            "move": move,
            "candidates": (action.target,),
            "targets": [],
            "turn_no": None,
            "abort": False,
            "flags": {},
            "errors": [],
            "rng": self._rng,
            "log": self._log,
            "bus": self._bus,
            "ops": self._ops or Ops(self._tm, self._bus),
            "scratch": {},
            "tags": set(),
            "ext": {"effect_registry": self._effect_registry},
        }
        return ctx

    def build_pipeline(self, ctx: ActionContext) -> ActionPipeline:
        pipeline = ActionPipeline()
        steps = list(self._catalog.resolve(ctx))
        if ctx.get("move", {}).get("effects") and effects_dispatch not in steps:
            if self._log is not None:
                self._log(
                    "[WARN] Effect pipeline produced by %s lacked effects_dispatch; appending fallback"
                    % self._catalog.__class__.__name__
                )
            steps.append(effects_dispatch)
        for step in steps:
            pipeline.use(step)
        return pipeline
