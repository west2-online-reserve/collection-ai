"""
(AI Generated Documentation)

Event contract and payload policy
=================================
This module documents the payload contracts for events emitted through
`EventBus.emit(...)`. The goal is to keep payloads uniform across
providers/handlers/ops while remaining forward-compatible.

Terminology & presence policy
-----------------------------
- MUST: the key MUST be present in the emitted payload AND MUST be non-None,
  unless the event explicitly says "may be None" for that key.
- MAY: the key is optional. Default policy: **omit the key** when not applicable
  or not computed. If included, it MUST be non-None unless explicitly documented
  as Optional (`T | None` / `Optional[T]`).
- MAY be None: explicitly allowed to be present with None; such cases are marked
  as Optional in the event spec.
- SHOULD: recommended practice but not strictly required.

Emitter rules
-------------
1) Provide all MUST keys, non-None, with the documented runtime types.
2) Prefer **omission** over `None` for all MAY keys (unless Optional is stated).
3) Key names MUST match the spec.

Subscriber rules
----------------
1) Prefer signatures that accept `**kwargs` for forward compatibility.
2) Treat MAY keys as absent by default (use `kwargs.get("...")`).
3) Do not rely on undocumented keys.

Shared vocabulary (types)
-------------------------
- actor: Creature
- move: dict               # If a stable identifier is needed, use move.get("key").
- candidates: Tuple[Creature, ...]
- targets: List[Creature]
- target: Creature         # Per-target phase (when applicable)
- turn_no: int | None
- rng: RNG
- log: Callable[[str], None]
- bus: EventBus
- ops: Any
- scratch: Dict[str, Any]
- tags: Set[str]
- errors: List[str]
- hit_index: int           # 0-based for multi-hit phases (when applicable)
- element: str
- cause: str               # Reason for miss: "evade" or "accuracy"
- base, raw, final, dealt: int
- multiplier: float
- reduction: float
- absorbed: int
- chance: float
- nonlethal, critical, hit: bool

Event payloads (MUST / MAY)
---------------------------

TURN_START
  MUST: turn_no, rng, log, bus
  MAY:  tags

ACTION_BEGIN
  MUST: actor, move, candidates, targets, log
  MAY:  rng, bus, turn_no, tags, scratch

BEFORE_ACCURACY_CHECK
  MUST: actor, move, targets, log
  MAY:  rng, target, tags

ACCURACY_CHECK
  MUST: actor, move, targets, log
  MAY:  rng, target, chance: float, hit: bool, tags

ACTION_MISS
  MUST: actor, move, targets, log, cause: str
  MAY:  target, tags, errors

ACTION_SUCCEEDED
  MUST: actor, move, targets, log
  MAY:  target, tags

DAMAGE_PIPELINE_BEGIN
  (Deprecated – kept for compatibility)
  MUST: actor, move, log
  MAY:  target, targets, tags

BEFORE_BASE_DAMAGE
  MUST: actor, move, target, log
  MAY:  scratch, tags

BASE_DAMAGE
  MUST: actor, move, target, log, base: int
  MAY:  raw: int, critical: bool, hit_index: int, tags

APPLY_TYPE_MULTIPLIER
  MUST: actor, move, target, log, element: str, multiplier: float
  MAY:  base: int, hit_index: int, tags

APPLY_SHIELDS
  MUST: actor, move, target, log, absorbed: int
  MAY:  incoming: int, reduction: float, hit_index: int, tags

APPLY_REDUCTIONS
  MUST: actor, move, target, log, reduced: int
  MAY:  incoming: int, sources: List[str], hit_index: int, tags

FINALIZE_DAMAGE
  MUST: actor, move, target, log, final: int
  MAY:  nonlethal: bool, hit_index: int, tags

AFTER_DAMAGE
  MUST: actor, move, target, log, dealt: int
  MAY:  tags

EFFECTS_RESOLVE
  MUST: actor, move, log, effect: dict
  MAY:  targets, target, effect_index: int, tags

CHECK_FAINT
  MUST: target, log
  MAY:  actor, move, tags

ON_FAINT
  MUST: target, log
  MAY:  actor, move, cause: str, tags

SWITCH
  MUST: out_creature: Creature, in_creature: Creature, log
  MAY:  reason: str, tags

ACTION_END
  MUST: actor, move, log
  MAY:  targets, tags, errors

TURN_END
  MUST: turn_no, log
  MAY:  tags

OpsEvent payloads (MUST / MAY)
------------------------------
Emitted exclusively by `pkmninsh.engine.ops.Ops`.

ACTION_ENQUEUED
  MUST: action: Any, log
  MAY:  when: str | int, tags

ACTION_INTERRUPTED
  MUST: action: Any, reason: str, log
  MAY:  tags

QUEUE_CLEARED
  MUST: log
  MAY:  tags

STATUS_APPLIED
  MUST: target, key: str, state: dict, log
  MAY:  reason: str, tags

STATUS_STACKED
  MUST: target, key: str, state: dict, log
  MAY:  previous_stacks: int, added: int, tags

STATUS_REFRESHED
  MUST: target, key: str, state: dict, log
  MAY:  duration: int | None, tags

STATUS_TICKED
  MUST: target, key: str, state: dict, phase: str, log
  MAY:  amount: int | float | None, tags

STATUS_EXPIRED
  MUST: target, key: str, state: dict, log
  MAY:  phase: str, tags

STATUS_REMOVED
  MUST: target, key: str, state: dict | None, log
  MAY:  reason: str, tags

SHIELD_GRANTED
  MUST: target, log
  MAY:  reduction: float | None, amount: int | None, once: bool, duration_turns: int | None, tags

SHIELD_CONSUMED
  MUST: target, absorbed: int, log
  MAY:  remaining_shield: int, tags

STAT_MODIFIED
  MUST: target, stat: str, old: Any, new: Any, log
  MAY:  source: Any, mode: str, tags

RESOURCE_CONSUMED
  MUST: target, resource: str, amount: int, log
  MAY:  tags

COOLDOWN_SET
  MUST: target, move_key: str, value: int, log
  MAY:  tags

COOLDOWN_REDUCED
  MUST: target, move_key: str, delta: int, log
  MAY:  tags

DAMAGE_APPLIED
  MUST: target, amount: int, remaining_hp: int, log
  MAY:  nonlethal: bool, lethal_blocked: bool, source: Any, tags

HEAL_APPLIED
  MUST: target, amount: int, new_hp: int, log
  MAY:  source: Any, tags

LIFESTEAL
  MUST: dealer: Creature, amount: int, log
  MAY:  source: Any, tags

REDIRECT_SET
  MUST: source: Creature, to: Creature, log
  MAY:  duration_turns: int | None, tags

UNTARGETABLE_SET
  MUST: target: Creature, value: bool, log
  MAY:  duration_turns: int | None, tags

FLAG_SET
  MUST: target: Any, name: str, value: Any, log
  MAY:  tags

FLAG_CLEARED
  MUST: target: Any, name: str, log
  MAY:  tags

TIMER_SCHEDULED
  MUST: timer_id: str, due_turn: int, log
  MAY:  payload: dict, tags

LOG_ENTRY
  MUST: message: str
  MAY:  level: str, tags

Emission order (typical)
------------------------
Success path:
  TURN_START → ACTION_BEGIN → BEFORE_ACCURACY_CHECK → ACCURACY_CHECK → ACTION_SUCCEEDED
  → EFFECTS_RESOLVE (per effect)
  → [for damage effects: BEFORE_BASE_DAMAGE → BASE_DAMAGE → APPLY_TYPE_MULTIPLIER
     → APPLY_SHIELDS → APPLY_REDUCTIONS → FINALIZE_DAMAGE → AFTER_DAMAGE]
  → CHECK_FAINT → ON_FAINT? → ACTION_END → TURN_END

On miss:
  TURN_START → ACTION_BEGIN → BEFORE_ACCURACY_CHECK → ACCURACY_CHECK → ACTION_MISS → ACTION_END → TURN_END

Per-target / multi-hit:
  Phase events that operate on a concrete target SHOULD include `target`.
  Repeated phases SHOULD include `hit_index` (0-based).

Backwards compatibility
-----------------------
DAMAGE_PIPELINE_BEGIN is deprecated. New code SHOULD NOT rely on it.
"""

from __future__ import annotations
from enum import Enum, auto
from typing import Any, Callable, Union

__all__ = ["Event", "OpsEvent", "EventBus", "AnyEvent"]


class Event(Enum):
    TURN_START            = auto()
    ACTION_BEGIN          = auto()
    BEFORE_ACCURACY_CHECK = auto()
    ACCURACY_CHECK        = auto()
    ACTION_MISS           = auto()
    ACTION_SUCCEEDED      = auto()
    DAMAGE_PIPELINE_BEGIN = auto()
    BEFORE_BASE_DAMAGE    = auto()
    BASE_DAMAGE           = auto()
    APPLY_TYPE_MULTIPLIER = auto()
    APPLY_SHIELDS         = auto()
    APPLY_REDUCTIONS      = auto()
    FINALIZE_DAMAGE       = auto()
    AFTER_DAMAGE          = auto()
    EFFECTS_RESOLVE       = auto()
    CHECK_FAINT           = auto()
    ON_FAINT              = auto()
    SWITCH                = auto()
    ACTION_END            = auto()
    TURN_END              = auto()


class OpsEvent(Enum):
    """Events emitted by the :class:`pkmninsh.engine.ops.Ops` façade."""

    ACTION_ENQUEUED    = auto()
    ACTION_INTERRUPTED = auto()
    QUEUE_CLEARED      = auto()

    STATUS_APPLIED     = auto()
    STATUS_STACKED     = auto()
    STATUS_REFRESHED   = auto()
    STATUS_TICKED      = auto()
    STATUS_EXPIRED     = auto()
    STATUS_REMOVED     = auto()

    SHIELD_GRANTED     = auto()
    SHIELD_CONSUMED    = auto()

    STAT_MODIFIED      = auto()
    RESOURCE_CONSUMED  = auto()
    COOLDOWN_SET       = auto()
    COOLDOWN_REDUCED   = auto()

    DAMAGE_APPLIED     = auto()
    HEAL_APPLIED       = auto()
    LIFESTEAL          = auto()

    REDIRECT_SET       = auto()
    UNTARGETABLE_SET   = auto()

    FLAG_SET           = auto()
    FLAG_CLEARED       = auto()
    TIMER_SCHEDULED    = auto()

    LOG_ENTRY          = auto()


AnyEvent = Union[Event, OpsEvent, str]


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[AnyEvent, list[Callable[..., None]]] = {}

    def subscribe(self, event_key: AnyEvent, fn: Callable[..., None]) -> None:
        self._subs.setdefault(event_key, []).append(fn)

    def emit(self, event_key: AnyEvent, **ctx: Any) -> None:
        for fn in tuple(self._subs.get(event_key, [])):
            fn(**ctx)
