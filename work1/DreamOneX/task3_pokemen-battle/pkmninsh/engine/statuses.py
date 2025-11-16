from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Mapping, MutableMapping, Protocol, TypedDict, runtime_checkable
from typing import NotRequired
import weakref

from pkmninsh.engine.registries import status_registry

__all__ = [
    "ModifierSpec",
    "StatusSpec",
    "StatusState",
    "StatusContext",
    "StatusHandler",
    "register_status_spec",
    "get_status_spec",
    "register_status_handler",
    "get_status_handler",
    "iter_status_targets",
    "track_status_target",
    "untrack_status_target",
]


class ModifierBounds(TypedDict, total=False):
    min: float | None
    max: float | None


class ModifierSpec(TypedDict, total=False):
    stat: str
    mode: Literal["add", "mul", "override"]
    value: float
    caps: NotRequired[ModifierBounds | None]
    when: NotRequired[dict[str, Any] | None]


class StatusDurationSpec(TypedDict, total=False):
    type: Literal["turns", "hits", "until_end"]
    value: int | None


class StatusStackingSpec(TypedDict, total=False):
    mode: Literal["add", "replace", "cap"]
    max: int


class StatusSpec(TypedDict, total=False):
    key: str
    stacking: StatusStackingSpec
    duration: NotRequired[StatusDurationSpec | None]
    dispellable: NotRequired[bool]
    tags: NotRequired[list[str]]
    modifiers: NotRequired[list[ModifierSpec]]
    hooks: NotRequired[list[str]]


class StatusState(TypedDict, total=False):
    key: str
    source_id: str | None
    stacks: int
    max_stacks: int
    expires_at_turn: int | None
    dispellable: bool
    tags: set[str]
    params: dict[str, Any]
    created_turn: int


class StatusContext(TypedDict, total=False):
    ops: Any
    target: Any
    spec: StatusSpec
    log: Callable[[str], None]
    attacker: Any | None
    defender: Any | None
    source: Any | None
    phase: str | None
    amount: int | float | None


@runtime_checkable
class StatusHandler(Protocol):
    """Protocol implemented by status handlers.

    Handlers may optionally define any of the hook methods below. Hooks that are
    not implemented are simply ignored at runtime.
    """

    def on_apply(self, ctx: StatusContext, state: StatusState) -> None: ...

    def on_tick(self, ctx: StatusContext, state: StatusState, phase: str) -> None: ...

    def on_remove(self, ctx: StatusContext, state: StatusState, reason: str | None) -> None: ...

    def on_hit(
        self,
        ctx: StatusContext,
        state: StatusState,
        *,
        amount: int | float,
    ) -> int | float: ...

    def on_being_hit(
        self,
        ctx: StatusContext,
        state: StatusState,
        *,
        amount: int | float,
    ) -> int | float: ...


@dataclass(slots=True)
class _StatusTrackerEntry:
    target_ref: weakref.ReferenceType[Any]


_STATUS_HANDLERS: dict[str, StatusHandler] = {}
_STATUS_TARGETS: dict[int, _StatusTrackerEntry] = {}


def _normalise_stacking(spec: Mapping[str, Any]) -> StatusStackingSpec:
    raw = spec.get("stacking")
    if isinstance(raw, Mapping):
        mode = str(raw.get("mode", "replace")).lower()
        max_stacks = int(raw.get("max", raw.get("max_stacks", 1) or 1))
    else:
        mode = str(spec.get("stack_mode", "replace")).lower()
        max_stacks = int(spec.get("max_stacks", 1) or 1)
    if mode not in {"add", "replace", "cap"}:
        mode = "replace"
    max_stacks = max(1, max_stacks)
    return StatusStackingSpec(mode=mode, max=max_stacks)


def _normalise_duration(spec: Mapping[str, Any]) -> StatusDurationSpec | None:
    raw = spec.get("duration")
    if isinstance(raw, Mapping):
        duration_type = str(raw.get("type", "turns")).lower()
        value = raw.get("value")
    else:
        duration_type = str(spec.get("duration_type", "turns") or "turns").lower()
        value = spec.get("duration_value", spec.get("duration_turns"))
    if duration_type not in {"turns", "hits", "until_end"}:
        duration_type = "turns"
    if duration_type == "until_end":
        return StatusDurationSpec(type="until_end", value=None)
    if value is None:
        return StatusDurationSpec(type=duration_type, value=None)
    try:
        duration_value = int(value)
    except (TypeError, ValueError):
        duration_value = None
    if duration_value is not None and duration_value < 0:
        duration_value = None
    return StatusDurationSpec(type=duration_type, value=duration_value)


def register_status_spec(spec: Mapping[str, Any]) -> StatusSpec:
    """Register a canonical status specification in the global registry."""

    key = str(spec.get("key")) if spec.get("key") is not None else None
    if not key:
        raise ValueError("Status spec must define a non-empty 'key'")

    canonical: StatusSpec = StatusSpec(
        key=key,
        stacking=_normalise_stacking(spec),
    )
    duration = _normalise_duration(spec)
    if duration is not None:
        canonical["duration"] = duration
    if "dispellable" in spec:
        canonical["dispellable"] = bool(spec.get("dispellable"))
    tags = spec.get("tags")
    if isinstance(tags, Iterable) and not isinstance(tags, (str, bytes)):
        canonical["tags"] = [str(tag) for tag in tags]
    modifiers = spec.get("modifiers")
    if isinstance(modifiers, Iterable):
        canonical["modifiers"] = [
            ModifierSpec(
                stat=str(mod.get("stat")),
                mode=str(mod.get("mode", "add")).lower(),
                value=float(mod.get("value", 0.0)),
                caps=mod.get("caps"),
                when=mod.get("when"),
            )
            for mod in modifiers
            if isinstance(mod, Mapping) and mod.get("stat")
        ]
    hooks = spec.get("hooks")
    if isinstance(hooks, Iterable) and not isinstance(hooks, (str, bytes)):
        canonical["hooks"] = [str(h) for h in hooks]

    status_registry.register(key, canonical)
    return canonical


def get_status_spec(key: str) -> StatusSpec:
    """Retrieve a status specification by key."""

    if not status_registry.has(key):
        raise KeyError(f"Status '{key}' is not registered")
    return status_registry.get(key)


def register_status_handler(key: str, handler: StatusHandler) -> None:
    """Associate a status handler with a registered status."""

    if not key:
        raise ValueError("Status handler key must be non-empty")
    _STATUS_HANDLERS[key] = handler


def get_status_handler(key: str) -> StatusHandler | None:
    """Return the handler registered for a status key (if any)."""

    return _STATUS_HANDLERS.get(key)


def track_status_target(target: Any) -> None:
    """Remember a creature that currently has at least one status."""

    if target is None:
        return
    try:
        target_ref = weakref.ref(target)
    except TypeError:
        return
    _STATUS_TARGETS[id(target)] = _StatusTrackerEntry(target_ref=target_ref)


def untrack_status_target(target_id: int) -> None:
    """Stop tracking a creature once it no longer has statuses."""

    _STATUS_TARGETS.pop(target_id, None)


def iter_status_targets() -> Iterable[Any]:
    """Yield tracked creatures that still exist."""

    stale: list[int] = []
    for key, entry in _STATUS_TARGETS.items():
        target = entry.target_ref()
        if target is None:
            stale.append(key)
            continue
        yield target
    for key in stale:
        _STATUS_TARGETS.pop(key, None)
