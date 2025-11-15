"""Operational façade for battle side effects.

The :class:`Ops` class centralises all direct mutations that happen during a
battle.  Steps, pipelines and plugins interact with the battle state only via
this layer which keeps the side effects observable and easier to reason about.

The implementation provided here favours clarity over hyper-optimisation.  The
goal is to expose a coherent API surface that future systems can build upon
without being tightly coupled to the underlying data structures.  Everything is
tracked in lightweight dictionaries so that tests – and, eventually, gameplay –
can introspect the state that has been altered by effects.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import heapq
from typing import Any, Deque, DefaultDict, Iterable, Mapping, Sequence, Literal

from pkmninsh.engine.events import EventBus, AnyEvent, OpsEvent
from pkmninsh.engine.turns import Action, TurnManager
from pkmninsh.engine.statuses import (
    StatusContext,
    StatusSpec,
    StatusState,
    ModifierSpec,
    get_status_spec,
    get_status_handler,
    iter_status_targets,
    track_status_target,
    untrack_status_target,
)

__all__ = ["Ops", "OpsEvent", "ScheduledEvent"]


@dataclass(slots=True)
class ScheduledEvent:
    """Small helper structure describing a delayed callback emission."""

    turn: int
    event: AnyEvent
    payload: dict[str, Any]


@dataclass(slots=True)
class FlagEntry:
    """Track a flag value and its optional expiry/metadata."""

    value: Any
    expires_at: int | None
    tags: set[str]


@dataclass(slots=True)
class RedirectRule:
    """Describe how targets should be redirected until a given moment."""

    to: Any | None
    until: str | int
    reason: str | None
    set_at: int


@dataclass(slots=True)
class UntargetableEntry:
    """Detail why and until when a target cannot be selected."""

    until: str | int | None
    reason: str | None
    set_at: int


class Ops:
    """High level façade for all side effects produced by battle logic."""

    def __init__(
        self,
        tm: TurnManager,
        bus: EventBus,
        *,
        turn_no: int = 0,
        rules: Any | None = None,
        context_id: str | None = None,
    ) -> None:
        self.tm = tm
        self.bus = bus
        self.turn_no = turn_no
        self.rules = rules
        self.context_id = context_id

        # Internal bookkeeping
        self._delayed_actions: list[tuple[int, int, int, Action]] = []
        self._action_seq: int = 0
        self._shields: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._resources: DefaultDict[int, dict[str, float]] = defaultdict(dict)
        self._cooldowns: DefaultDict[int, dict[str, int]] = defaultdict(dict)
        self._flags: DefaultDict[str, dict[str, FlagEntry]] = defaultdict(dict)
        self._redirects: dict[str, RedirectRule] = {}
        self._untargetable: dict[int, UntargetableEntry] = {}
        # Heap of (turn, seq, ScheduledEvent) awaiting emission.
        self._scheduled_events_heap: list[tuple[int, int, ScheduledEvent]] = []
        self._schedule_seq: int = 0
        self._logs: list[str] = []
        # Remember the last damage dealt by a source actor for lifesteal math.
        self._last_damage_by_source: dict[int, int] = defaultdict(int)

    # Helpers

    def _scope_key(self, scope: Any) -> str:
        if isinstance(scope, str):
            return scope
        return f"obj:{id(scope)}"

    def _emit(self, event_key: AnyEvent, **payload: Any) -> None:
        payload.setdefault("turn", self.turn_no)
        if self.context_id is not None:
            payload.setdefault("context_id", self.context_id)
        self.bus.emit(event_key, **payload)

    def _add_log(self, message: str) -> None:
        self._logs.append(message)

    # A. Action scheduling

    def enqueue_action(
        self,
        actor: Any,
        target: Any,
        move_key: str,
        *,
        delay_turns: int = 0,
        priority: int = 0,
        reason: str | None = None,
    ) -> Action:
        action = Action(actor=actor, target=target, move_key=move_key, reason=reason or "ops")

        if delay_turns <= 0:
            if priority > 0:
                self.tm.enqueue_front(action)
            else:
                self.tm.enqueue(action)
        else:
            run_turn = self.turn_no + max(0, delay_turns)
            heapq.heappush(
                self._delayed_actions,
                (run_turn, -priority, self._action_seq, action),
            )
            self._action_seq += 1

        self._emit(
            OpsEvent.ACTION_ENQUEUED,
            action=action,
            delay_turns=delay_turns,
            priority=priority,
            reason=reason,
        )
        return action

    def interrupt_actions(
        self,
        *,
        of_actor: Any | None = None,
        against_target: Any | None = None,
        reason: str | None = None,
    ) -> list[Action]:
        removed: list[Action] = []
        new_queue: Deque[Action] = deque()
        while self.tm.queue:
            action = self.tm.queue.popleft()
            matches_actor = of_actor is None or action.actor is of_actor
            matches_target = against_target is None or action.target is against_target
            if matches_actor and matches_target:
                removed.append(action)
            else:
                new_queue.append(action)
        self.tm.queue = new_queue

        if removed:
            self._emit(
                OpsEvent.ACTION_INTERRUPTED,
                actions=tuple(removed),
                reason=reason,
            )
        return removed

    def clear_queue(
        self,
        scope: str = "all",
        *,
        actor: Any | None = None,
        target: Any | None = None,
    ) -> None:
        scope = scope.lower()
        if scope == "all":
            cleared = list(self.tm.queue)
            self.tm.queue.clear()
        elif scope == "actor":
            cleared = self.interrupt_actions(of_actor=actor)
        elif scope == "target":
            cleared = self.interrupt_actions(against_target=target)
        else:
            raise ValueError(f"Unknown scope: {scope}")

        self._emit(OpsEvent.QUEUE_CLEARED, scope=scope, actions=tuple(cleared))

    # B. Status & passives

    def _status_map(self, target: Any) -> dict[str, StatusState]:
        status_map = getattr(target, "statuses", None)
        if status_map is None:
            raise AttributeError("Target does not expose a 'statuses' mapping")
        return status_map

    def _status_snapshot(self, state: StatusState) -> StatusState:
        return {
            "key": state.get("key"),
            "source_id": state.get("source_id"),
            "stacks": state.get("stacks", 0),
            "max_stacks": state.get("max_stacks", 1),
            "expires_at_turn": state.get("expires_at_turn"),
            "dispellable": state.get("dispellable", True),
            "tags": set(state.get("tags", set())),
            "params": dict(state.get("params", {})),
            "created_turn": state.get("created_turn", self.turn_no),
        }

    def _build_status_context(
        self,
        target: Any,
        spec: StatusSpec,
        state: StatusState,
        *,
        phase: str | None = None,
        attacker: Any | None = None,
        defender: Any | None = None,
        amount: int | float | None = None,
    ) -> StatusContext:
        ctx: StatusContext = {
            "ops": self,
            "target": target,
            "spec": spec,
            "log": self.log,
            "source": state.get("params", {}).get("source"),
            "phase": phase,
            "attacker": attacker,
            "defender": defender,
            "amount": amount,
        }
        return ctx

    def _refresh_status_duration(
        self,
        state: StatusState,
        spec: StatusSpec,
        duration_override: int | None,
    ) -> bool:
        params = state.setdefault("params", {})
        duration = spec.get("duration")
        if not duration:
            state["expires_at_turn"] = None
            params.pop("remaining_turns", None)
            params.pop("remaining_hits", None)
            return False

        dtype = duration.get("type", "turns")
        refreshed = False
        if dtype == "turns":
            value = duration_override if duration_override is not None else duration.get("value")
            if value is None:
                params.pop("remaining_turns", None)
                state["expires_at_turn"] = None
                return False
            turns = max(0, int(value))
            params["remaining_turns"] = turns
            state["expires_at_turn"] = self.turn_no + turns
            refreshed = True
        elif dtype == "hits":
            value = duration_override if duration_override is not None else duration.get("value")
            if value is not None:
                params["remaining_hits"] = max(0, int(value))
                refreshed = True
        else:  # until_end
            state["expires_at_turn"] = None
            params.pop("remaining_turns", None)
            params.pop("remaining_hits", None)
        return refreshed

    def _decrement_status_duration(
        self,
        state: StatusState,
        spec: StatusSpec,
        phase: str,
    ) -> bool:
        duration = spec.get("duration")
        if not duration:
            return False

        params = state.setdefault("params", {})
        dtype = duration.get("type", "turns")
        if dtype == "turns":
            if phase != "turn_end":
                return False
            remaining = params.get("remaining_turns")
            if remaining is None:
                base = duration.get("value")
                if base is None:
                    return False
                remaining = max(0, int(base))
            if remaining > 0:
                remaining -= 1
            params["remaining_turns"] = remaining
            future_turn = remaining if remaining is not None else 0
            state["expires_at_turn"] = self.turn_no + max(0, future_turn)
            return remaining is not None and remaining <= 0
        if dtype == "hits":
            remaining_hits = params.get("remaining_hits")
            if remaining_hits is None:
                base_hits = duration.get("value")
                if base_hits is None:
                    return False
                remaining_hits = max(0, int(base_hits))
            params["remaining_hits"] = remaining_hits
            return remaining_hits <= 0
        return False

    def _consume_status_hit(self, state: StatusState, spec: StatusSpec) -> bool:
        duration = spec.get("duration")
        if not duration or duration.get("type") != "hits":
            return False
        params = state.setdefault("params", {})
        remaining = params.get("remaining_hits")
        if remaining is None:
            base = duration.get("value")
            if base is None:
                return False
            remaining = max(0, int(base))
        if remaining > 0:
            remaining -= 1
        params["remaining_hits"] = remaining
        return remaining <= 0

    def _run_on_hit(
        self,
        attacker: Any | None,
        amount: float,
        defender: Any | None,
    ) -> float:
        if attacker is None:
            return float(amount)

        try:
            status_map = self._status_map(attacker)
        except AttributeError:
            return float(amount)

        new_amount = float(amount)
        for key, state in list(status_map.items()):
            handler = get_status_handler(key)
            if handler is None:
                continue
            on_hit = getattr(handler, "on_hit", None)
            if not callable(on_hit):
                continue
            spec = get_status_spec(key)
            ctx = self._build_status_context(
                attacker,
                spec,
                state,
                phase="on_hit",
                attacker=attacker,
                defender=defender,
                amount=new_amount,
            )
            result = on_hit(ctx, state, amount=new_amount)
            if result is not None:
                new_amount = float(result)
                ctx["amount"] = new_amount
            snapshot = self._status_snapshot(state)
            self._emit(
                OpsEvent.STATUS_TICKED,
                target=attacker,
                key=key,
                state=snapshot,
                phase="on_hit",
                amount=ctx.get("amount"),
                defender=defender,
            )
            if self._consume_status_hit(state, spec):
                self._emit(
                    OpsEvent.STATUS_EXPIRED,
                    target=attacker,
                    key=key,
                    state=self._status_snapshot(state),
                    phase="on_hit",
                )
                self.remove_status(attacker, key, reason="consumed")
        return float(max(0.0, new_amount))

    def _run_on_being_hit(
        self,
        target: Any,
        amount: float,
        attacker: Any | None,
    ) -> float:
        try:
            status_map = self._status_map(target)
        except AttributeError:
            return float(amount)
        new_amount = float(amount)
        for key, state in list(status_map.items()):
            handler = get_status_handler(key)
            if handler is None:
                continue
            on_hit = getattr(handler, "on_being_hit", None)
            if not callable(on_hit):
                continue
            spec = get_status_spec(key)
            ctx = self._build_status_context(
                target,
                spec,
                state,
                phase="on_being_hit",
                attacker=attacker,
                defender=target,
                amount=new_amount,
            )
            result = on_hit(ctx, state, amount=new_amount)
            if result is not None:
                new_amount = max(0.0, float(result))
                ctx["amount"] = new_amount
            snapshot = self._status_snapshot(state)
            self._emit(
                OpsEvent.STATUS_TICKED,
                target=target,
                key=key,
                state=snapshot,
                phase="on_being_hit",
                amount=ctx.get("amount"),
            )
            if self._consume_status_hit(state, spec):
                self._emit(
                    OpsEvent.STATUS_EXPIRED,
                    target=target,
                    key=key,
                    state=self._status_snapshot(state),
                    phase="on_being_hit",
                )
                self.remove_status(target, key, reason="consumed")
        return new_amount

    def add_status(
        self,
        target: Any,
        status_key: str,
        *,
        stacks: int = 1,
        duration: int | None = None,
        source: Any | None = None,
        params: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> StatusState:
        spec = get_status_spec(status_key)
        status_map = self._status_map(target)
        state = status_map.get(status_key)
        created = state is None
        max_stacks = spec.get("stacking", {}).get("max", 1)

        if created:
            state = {
                "key": status_key,
                "source_id": str(source) if source is not None else None,
                "stacks": 0,
                "max_stacks": max_stacks,
                "expires_at_turn": None,
                "dispellable": bool(spec.get("dispellable", True)),
                "tags": set(spec.get("tags", [])),
                "params": {},
                "created_turn": self.turn_no,
            }
        else:
            state = status_map[status_key]
            if source is not None:
                state["source_id"] = str(source)

        state_tags = state.setdefault("tags", set())
        for tag in spec.get("tags", []):
            state_tags.add(tag)
        if tags:
            for tag in tags:
                state_tags.add(tag)

        state_params = dict(state.get("params", {}))
        if params:
            state_params.update(params)
        if source is not None:
            state_params.setdefault("source", source)
        state["params"] = state_params

        existing_stacks = state.get("stacks", 0)
        requested = max(1, int(stacks))
        mode = spec.get("stacking", {}).get("mode", "replace")
        if mode == "add":
            new_stacks = min(max_stacks, existing_stacks + requested)
        elif mode == "cap":
            new_stacks = min(max_stacks, max(existing_stacks, requested))
        else:  # replace
            new_stacks = min(max_stacks, requested)
        state["stacks"] = new_stacks
        state["max_stacks"] = max_stacks

        refreshed = self._refresh_status_duration(state, spec, duration)

        status_map[status_key] = state
        track_status_target(target)

        snapshot = self._status_snapshot(state)
        if created:
            self._emit(OpsEvent.STATUS_APPLIED, target=target, key=status_key, state=snapshot)
        else:
            delta = new_stacks - existing_stacks
            if delta > 0:
                self._emit(
                    OpsEvent.STATUS_STACKED,
                    target=target,
                    key=status_key,
                    state=snapshot,
                    previous_stacks=existing_stacks,
                    added=delta,
                )
            if refreshed:
                remaining = state.get("params", {}).get("remaining_turns")
                self._emit(
                    OpsEvent.STATUS_REFRESHED,
                    target=target,
                    key=status_key,
                    state=snapshot,
                    duration=remaining,
                )

        handler = get_status_handler(status_key)
        if handler is not None:
            on_apply = getattr(handler, "on_apply", None)
            if callable(on_apply) and created:
                ctx = self._build_status_context(target, spec, state, phase="on_apply")
                on_apply(ctx, state)

        return state

    def remove_status(
        self,
        target: Any,
        status_key: str,
        *,
        reason: str | None = None,
    ) -> bool:
        status_map = self._status_map(target)
        state = status_map.pop(status_key, None)
        if state is None:
            return False

        spec = get_status_spec(status_key)
        handler = get_status_handler(status_key)
        if handler is not None:
            on_remove = getattr(handler, "on_remove", None)
            if callable(on_remove):
                ctx = self._build_status_context(target, spec, state, phase="on_remove")
                on_remove(ctx, state, reason)

        snapshot = self._status_snapshot(state)
        self._emit(OpsEvent.STATUS_REMOVED, target=target, key=status_key, state=snapshot, reason=reason)

        if not status_map:
            untrack_status_target(id(target))
        return True

    def has_status(
        self,
        target: Any,
        status_key: str,
        *,
        tags: Sequence[str] | None = None,
    ) -> bool:
        status_map = target.statuses if hasattr(target, "statuses") else {}
        state = status_map.get(status_key)
        if not state:
            return False
        if not tags:
            return True
        state_tags = state.get("tags", set())
        return bool(state_tags.intersection(set(tags)))

    def query_status(
        self,
        target: Any,
        key: str | None = None,
    ) -> StatusState | list[StatusState] | None:
        status_map = target.statuses if hasattr(target, "statuses") else {}
        if key is None:
            return [self._status_snapshot(state) for state in status_map.values()]
        state = status_map.get(key)
        if state is None:
            return None
        return self._status_snapshot(state)

    def query_modifiers(self, target: Any) -> list[ModifierSpec]:
        status_map = target.statuses if hasattr(target, "statuses") else {}
        modifiers: list[ModifierSpec] = []
        for key, state in status_map.items():
            spec = get_status_spec(key)
            stacks = max(1, int(state.get("stacks", 1)))
            for mod in spec.get("modifiers", []) or []:
                stat_key = mod.get("stat")
                if not stat_key:
                    continue
                mode = mod.get("mode", "add")
                base_value = float(mod.get("value", 0.0))
                value = base_value
                if mode == "add":
                    value = base_value * stacks
                elif mode == "mul":
                    value = base_value ** stacks
                entry: ModifierSpec = {
                    "stat": stat_key,
                    "mode": mode,
                    "value": value,
                }
                if "caps" in mod:
                    entry["caps"] = mod.get("caps")
                if "when" in mod:
                    entry["when"] = mod.get("when")
                modifiers.append(entry)
        return modifiers

    def tick_statuses(self, phase: Literal["turn_start", "turn_end", "action_end"]) -> None:
        for target in list(iter_status_targets()):
            status_map = target.statuses if hasattr(target, "statuses") else {}
            for key, state in list(status_map.items()):
                spec = get_status_spec(key)
                handler = get_status_handler(key)
                ctx = self._build_status_context(target, spec, state, phase=phase)
                if handler is not None:
                    on_tick = getattr(handler, "on_tick", None)
                    if callable(on_tick):
                        on_tick(ctx, state, phase)
                snapshot = self._status_snapshot(state)
                self._emit(
                    OpsEvent.STATUS_TICKED,
                    target=target,
                    key=key,
                    state=snapshot,
                    phase=phase,
                    amount=ctx.get("amount"),
                )
                if self._decrement_status_duration(state, spec, phase):
                    self._emit(
                        OpsEvent.STATUS_EXPIRED,
                        target=target,
                        key=key,
                        state=self._status_snapshot(state),
                        phase=phase,
                    )
                    self.remove_status(target, key, reason="expired")

    # C. Shields & mitigation

    def grant_shield(
        self,
        target: Any,
        reduction: int | float,
        *,
        once: bool = True,
        duration_turns: int | None = None,
        tags: Iterable[str] | None = None,
        mode: str = "flat",
    ) -> dict[str, Any]:
        entry = {
            "reduction": float(reduction),
            "remaining": float(reduction),
            "once": once,
            "applied_at": self.turn_no,
            "expires_at": None
            if duration_turns is None
            else self.turn_no + max(0, duration_turns),
            "tags": set(tags or ()),
            "mode": mode,
        }
        self._shields[id(target)].append(entry)
        self._emit(
            OpsEvent.SHIELD_GRANTED,
            target=target,
            shield=entry,
        )
        return entry

    def consume_shield(
        self,
        target: Any,
        *,
        amount: int | None = None,
        tags: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        tags = set(tags or ())
        pool = self._shields.get(id(target), [])
        removed: list[dict[str, Any]] = []
        keep: list[dict[str, Any]] = []
        remaining = amount
        for shield in pool:
            if tags and not shield["tags"].intersection(tags):
                keep.append(shield)
                continue
            if remaining is not None and remaining <= 0:
                keep.append(shield)
                continue
            removed.append(shield)
            if remaining is not None:
                remaining -= 1
        self._shields[id(target)] = keep
        if not keep:
            self._shields.pop(id(target), None)

        if removed:
            self._emit(
                OpsEvent.SHIELD_CONSUMED,
                target=target,
                shields=tuple(removed),
                tags=tags,
                amount=amount,
            )
        return removed

    def _apply_shields(self, target: Any, amount: float) -> float:
        if amount <= 0:
            return 0.0

        pool = list(self._shields.get(id(target), []))
        if not pool:
            return float(amount)

        updated: list[dict[str, Any]] = []
        consumed: list[dict[str, Any]] = []
        remaining_damage = float(amount)
        absorbed_total = 0.0

        for shield in pool:
            expires_at = shield.get("expires_at")
            if expires_at is not None and expires_at <= self.turn_no:
                consumed.append(shield)
                continue

            mode = str(shield.get("mode", "flat"))
            if mode == "ratio":
                ratio = max(0.0, min(1.0, float(shield.get("reduction", 0.0))))
                if ratio > 0:
                    before = remaining_damage
                    remaining_damage *= 1.0 - ratio
                    absorbed_total += max(0.0, before - remaining_damage)
                if shield.get("once", True):
                    consumed.append(shield)
                else:
                    updated.append(shield)
                continue

            remaining_value = float(shield.get("remaining", shield.get("reduction", 0.0)))
            if remaining_value <= 0 or remaining_damage <= 0:
                updated.append(shield)
                continue

            absorb = min(remaining_damage, remaining_value)
            remaining_damage -= absorb
            absorbed_total += absorb

            if shield.get("once", True):
                consumed.append(shield)
            else:
                new_remaining = remaining_value - absorb
                if new_remaining <= 0:
                    consumed.append(shield)
                else:
                    shield["remaining"] = new_remaining
                    updated.append(shield)

        if updated:
            self._shields[id(target)] = updated
        else:
            self._shields.pop(id(target), None)

        if absorbed_total > 0:
            self._emit(
                OpsEvent.SHIELD_CONSUMED,
                target=target,
                shields=tuple(consumed),
                absorbed=absorbed_total,
            )

        return max(0.0, remaining_damage)

    # D. Stats & resources

    def modify_stat(
        self,
        target: Any,
        stat: str,
        delta: int | float,
        *,
        clamp: bool = True,
        tags: Iterable[str] | None = None,
    ) -> float | int:
        current = getattr(target, stat)
        new_value = current + delta
        if clamp and isinstance(new_value, (int, float)):
            floor = 0
            ceiling = None
            if stat == "hp" and hasattr(target, "max_hp"):
                ceiling = target.max_hp
            if ceiling is not None:
                new_value = max(floor, min(ceiling, new_value))
            else:
                new_value = max(floor, new_value)

        setattr(target, stat, new_value)
        self._emit(
            OpsEvent.STAT_MODIFIED,
            target=target,
            stat=stat,
            delta=delta,
            value=new_value,
            tags=set(tags or ()),
        )
        return new_value

    def consume_resource(
        self,
        target: Any,
        resource: str,
        amount: int | float,
        *,
        allow_negative: bool = False,
    ) -> float:
        pool = self._resources[id(target)]
        current = pool.get(resource, 0.0)
        new_value = current - amount
        if not allow_negative and new_value < 0:
            new_value = 0.0
        pool[resource] = new_value
        self._emit(
            OpsEvent.RESOURCE_CONSUMED,
            target=target,
            resource=resource,
            amount=amount,
            remaining=new_value,
            allow_negative=allow_negative,
        )
        return new_value

    def set_cooldown(self, actor: Any, move_key: str, turns: int) -> int:
        expires = self.turn_no + max(0, turns)
        self._cooldowns[id(actor)][move_key] = expires
        self._emit(
            OpsEvent.COOLDOWN_SET,
            actor=actor,
            move_key=move_key,
            expires_at=expires,
        )
        return expires

    def reduce_cooldown(self, actor: Any, move_key: str, turns: int) -> int | None:
        if move_key not in self._cooldowns[id(actor)]:
            return None
        self._cooldowns[id(actor)][move_key] = max(
            self.turn_no,
            self._cooldowns[id(actor)][move_key] - max(0, turns),
        )
        expires = self._cooldowns[id(actor)][move_key]
        self._emit(
            OpsEvent.COOLDOWN_REDUCED,
            actor=actor,
            move_key=move_key,
            expires_at=expires,
            reduced_by=turns,
        )
        return expires

    # E. Damage & healing

    def apply_damage(
        self,
        target: Any,
        amount: int,
        *,
        nonlethal: bool = False,
        source: Any | None = None,
        tags: Iterable[str] | None = None,
    ) -> int:
        if amount <= 0:
            return 0
        adjusted = float(amount)
        adjusted = self._run_on_hit(source, adjusted, target)
        adjusted = self._run_on_being_hit(target, adjusted, attacker=source)
        mitigated = self._apply_shields(target, adjusted)
        amount = int(round(mitigated))
        if nonlethal and hasattr(target, "hp") and target.hp - amount <= 0:
            amount = max(0, target.hp - 1)
        if amount <= 0:
            return 0
        target.apply_damage(int(amount))
        if source is not None:
            self._last_damage_by_source[id(source)] = int(amount)
        self._emit(
            OpsEvent.DAMAGE_APPLIED,
            target=target,
            amount=int(amount),
            nonlethal=nonlethal,
            source=source,
            tags=set(tags or ()),
        )
        return int(amount)

    def heal(
        self,
        target: Any,
        amount: int,
        *,
        clamp: bool = True,
        source: Any | None = None,
        tags: Iterable[str] | None = None,
    ) -> int:
        if amount <= 0:
            return 0
        before = target.hp
        target.heal(int(amount))
        healed = int(amount)
        if clamp:
            healed = target.hp - before
        self._emit(
            OpsEvent.HEAL_APPLIED,
            target=target,
            amount=healed,
            clamp=clamp,
            source=source,
            tags=set(tags or ()),
        )
        return healed

    def lifesteal(
        self,
        source_actor: Any,
        target: Any,
        percent_or_amount: float,
        *,
        mode: str = "percent_of_damage",
    ) -> int:
        healed = 0
        if mode == "percent_of_damage":
            damage = self._last_damage_by_source.get(id(source_actor), 0)
            if 0 <= percent_or_amount <= 1:
                healed = int(round(damage * percent_or_amount))
            else:
                healed = int(round(percent_or_amount))
        else:
            healed = int(round(percent_or_amount))
        healed = max(0, healed)
        if healed:
            self.heal(source_actor, healed, source=target, tags={"lifesteal"})
        self._emit(
            OpsEvent.LIFESTEAL,
            source=source_actor,
            target=target,
            healed=healed,
            mode=mode,
        )
        return healed

    # F. Target control

    def redirect(
        self,
        from_actor: Any | None = None,
        to_target: Any | None = None,
        *,
        until: str | int = "end_of_turn",
        reason: str | None = None,
    ) -> None:
        key = "global" if from_actor is None else f"actor:{id(from_actor)}"
        self._redirects[key] = RedirectRule(
            to=to_target,
            until=until,
            reason=reason,
            set_at=self.turn_no,
        )
        self._emit(
            OpsEvent.REDIRECT_SET,
            from_actor=from_actor,
            to_target=to_target,
            until=until,
            reason=reason,
        )

    def mark_untargetable(
        self,
        target: Any,
        *,
        until: str | int | None = None,
        reason: str | None = None,
    ) -> None:
        self._untargetable[id(target)] = UntargetableEntry(
            until=until,
            reason=reason,
            set_at=self.turn_no,
        )
        self._emit(
            OpsEvent.UNTARGETABLE_SET,
            target=target,
            until=until,
            reason=reason,
        )

    # G. Flags & timers

    def set_flag(
        self,
        scope: Any,
        name: str,
        value: Any = True,
        *,
        ttl_turns: int | None = None,
        tags: Iterable[str] | None = None,
    ) -> None:
        scope_key = self._scope_key(scope)
        expires = None if ttl_turns is None else self.turn_no + max(0, ttl_turns)
        self._flags[scope_key][name] = FlagEntry(
            value=value,
            expires_at=expires,
            tags=set(tags or ()),
        )
        self._emit(
            OpsEvent.FLAG_SET,
            scope=scope,
            name=name,
            value=value,
            ttl_turns=ttl_turns,
            tags=set(tags or ()),
        )

    def get_flag(self, scope: Any, name: str, default: Any | None = None) -> Any:
        scope_key = self._scope_key(scope)
        data = self._flags.get(scope_key, {}).get(name)
        if not data:
            return default
        expires_at = data.expires_at
        if expires_at is not None and expires_at <= self.turn_no:
            self._flags[scope_key].pop(name, None)
            if not self._flags[scope_key]:
                self._flags.pop(scope_key, None)
            return default
        return data.value

    def clear_flag(self, scope: Any, name: str) -> None:
        scope_key = self._scope_key(scope)
        if name in self._flags.get(scope_key, {}):
            self._flags[scope_key].pop(name, None)
            if not self._flags[scope_key]:
                self._flags.pop(scope_key, None)
            self._emit(OpsEvent.FLAG_CLEARED, scope=scope, name=name)

    def schedule(
        self,
        callback_event: AnyEvent,
        at_turn: int,
        **payload: Any,
    ) -> ScheduledEvent:
        event = ScheduledEvent(turn=at_turn, event=callback_event, payload=dict(payload))
        heapq.heappush(self._scheduled_events_heap, (at_turn, self._schedule_seq, event))
        self._schedule_seq += 1
        self._emit(
            OpsEvent.TIMER_SCHEDULED,
            event=callback_event,
            at_turn=at_turn,
            payload=payload,
        )
        return event

    # H. Events & logging

    def emit(self, event: AnyEvent, **kwargs: Any) -> None:
        self._emit(event, **kwargs)

    def log(self, message: str, *, level: str = "info", tags: Iterable[str] | None = None) -> None:
        self._add_log(message)
        self._emit(
            OpsEvent.LOG_ENTRY,
            message=message,
            level=level,
            tags=set(tags or ()),
        )

    # Utilities for consumers (not part of public blueprint but practical)

    def advance_turn(self, turns: int = 1) -> None:
        """Advance internal clock and release delayed actions/timers."""

        for _ in range(turns):
            self.turn_no += 1
            self._release_delayed_actions()
            self._release_scheduled_events()

    def _release_delayed_actions(self) -> None:
        while self._delayed_actions and self._delayed_actions[0][0] <= self.turn_no:
            _, neg_priority, _, action = heapq.heappop(self._delayed_actions)
            if -neg_priority > 0:
                self.tm.enqueue_front(action)
            else:
                self.tm.enqueue(action)

    def _release_scheduled_events(self) -> None:
        while self._scheduled_events_heap and self._scheduled_events_heap[0][0] <= self.turn_no:
            _, _, scheduled = heapq.heappop(self._scheduled_events_heap)
            self._emit(scheduled.event, **scheduled.payload)
