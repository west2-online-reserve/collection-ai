from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

from pkmninsh.engine.events import EventBus
from pkmninsh.engine.rng import RNG
from pkmninsh.model.creature import Creature

__all__ = ["ActionContext", "ActionPipeline"]


class ActionContext(TypedDict, total=False):
    """

    (AI Generated Documentation)

    Context object passed through ActionPipeline steps.

    Design principles
    - Keep facts separate from flow-control and infra.
    - Facts are read-only by convention; steps may mutate flow and intermediates.
    - Prefer descriptive `tags` for "what happened", and narrow `flags` for
      imperative switches. Use `ext` to carry opt-in integrations to avoid
      expanding the core schema.

    Keys
    actor : Creature
        The acting creature for this action. Read-only by convention.

    move : dict
        Canonical move spec used by providers/handlers. It should include
        a stable identifier (e.g. "key") inside the dict if needed for logs.

    candidates : Tuple[Creature, ...]
        Immutable selection domain for targeting resolution. Providers may
        choose from this domain but must not mutate it.

    targets : List[Creature]
        Resolved targets for execution. Providers write here; later steps read.

    turn_no : Optional[int]
        Turn index if available (e.g. 1-based). Optional and used for timing
        decisions (charge, cooldown windows, etc.).

    abort : bool
        When set to True, remaining steps should stop early. Steps that set
        this must do so intentionally and document the reason (e.g. charge step).

    flags : Dict[str, Any]
        Narrow, imperative switches that influence how the rest of the pipeline
        should behave (e.g. {"nonlethal": True, "ignore_shield": False}).
        Flags are transient and scoped to this action only.

    errors : List[str]
        Non-fatal validation or runtime issues collected during assembly/run.
        The pipeline should log and proceed when safe.

    rng : RNG
        Deterministic RNG facade injected by the assembler. Always use
        `rng.chance(p)` instead of accessing a raw random source.

    log : Callable[[str], None]
        Logger for user-facing battle logs. Keep messages concise and
        deterministic under a fixed RNG seed.

    bus : EventBus
        Engine-wide event bus. Steps and ops publish structured events here.

    ops : Any
        Side-effect gateway (damage, shield, status, enqueue, schedule, ...).
        Steps/handlers must use `ops` instead of mutating models directly.

    scratch : Dict[str, Any]
        Ephemeral workspace for intermediate values computed by steps
        (e.g. "hit_roll", "base_damage"). Cleared after the action.

    tags : set[str]
        Idempotent labels describing outcomes or facts of this action, such as
        {"miss", "crit", "element:fire"}. Prefer tags for classification that
        does not change control flow.

    ext : Dict[str, Any]
        Extension slot for optional integrations and overrides without
        changing the schema. Recommended to namespace keys, e.g.:
        ext["effect_registry"] = EffectRegistry instance override.

    Exclusions (intentional)
    - move_key: read from `move.get("key")` instead of duplicating schema.
    - action: keep the raw Action outside the pipeline to avoid tight coupling.
    - effect_registry: when an override is needed, pass it via
      `ext["effect_registry"]`; steps must otherwise fallback to the global one.
    """

    # facts (read-only by convention)
    actor: Creature
    move: dict
    candidates: Tuple[Creature, ...]
    targets: List[Creature]
    turn_no: Optional[int]

    # control (flow)
    abort: bool
    flags: Dict[str, Any]
    errors: List[str]

    # infra (tools)
    rng: RNG
    log: Callable[[str], None]
    bus: EventBus
    ops: Any

    # extension (for intermediates)
    scratch: Dict[str, Any]
    tags: set[str]
    ext: Dict[str, Any]


Step = Callable[[ActionContext], None]


class ActionPipeline:
    """Order-preserving pipeline for action processing."""

    __slots__ = ("_steps",)

    def __init__(self) -> None:
        self._steps: List[Step] = []

    def use(self, step: Step) -> None:
        if not callable(step):
            raise TypeError("step must be callable(ctx)")
        self._steps.append(step)

    def clear(self) -> None:
        self._steps.clear()

    def run(self, ctx: ActionContext) -> None:
        for step in list(self._steps):
            if ctx.get("abort"):
                break
            step(ctx)
