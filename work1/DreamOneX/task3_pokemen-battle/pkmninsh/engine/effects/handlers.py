from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Mapping, Any

from pkmninsh.engine.events import Event
from pkmninsh.engine.ops import Ops
from pkmninsh.engine.pipeline import ActionContext
from pkmninsh.engine.registries import type_registry
from pkmninsh.infra.i18n import translate as t

EffectSpec = Mapping[str, Any]


class EffectHandler(Protocol):
    """Protocol implemented by all effect handlers."""

    namespace: str
    order: int

    def can_handle(self, effect: EffectSpec, ctx: ActionContext) -> bool:
        """Return ``True`` when this handler can process ``effect``."""

    def handle(self, effect: EffectSpec, ctx: ActionContext) -> None:
        """Execute the handler for the provided ``effect``."""


def _normalise_type(effect: EffectSpec) -> str:
    return str(effect.get("type", "")).strip().lower()


def _type_multiplier(attacker_element: str | None, defender_element: str | None) -> float:
    if not attacker_element or not defender_element:
        return 1.0
    if not type_registry.has(attacker_element):
        return 1.0
    spec = type_registry.get(attacker_element)
    strong = set(spec.get("strong", []) or ())
    weak = set(spec.get("weak", []) or ())
    if defender_element in strong:
        return 2.0
    if defender_element in weak:
        return 0.5
    return 1.0


@dataclass(slots=True)
class DamageHandler:
    namespace: str = "builtin.damage"
    order: int = 100

    def can_handle(self, effect: EffectSpec, ctx: ActionContext) -> bool:  # noqa: D401
        return _normalise_type(effect) == "damage"

    def handle(self, effect: EffectSpec, ctx: ActionContext) -> None:  # noqa: D401
        targets = list(ctx.get("targets", []))
        if not targets:
            return

        actor = ctx["actor"]
        move = ctx["move"]
        move_key = ctx.get("move_key", move.get("key", "move"))
        rng = ctx["rng"]
        log = ctx["log"]
        bus = ctx.get("bus")
        ops: Ops = ctx["ops"]
        scratch = ctx.setdefault("scratch", {})
        tags = ctx.setdefault("tags", set())

        base_accuracy = float(effect.get("accuracy", move.get("accuracy", 1.0)))
        accuracy_bonus = float(scratch.get("accuracy_bonus", 0.0))
        chance = max(0.0, min(1.0, base_accuracy + accuracy_bonus))

        defender = targets[0]
        dodge = float(defender.dodge)
        dodge += float(scratch.get("defender_dodge_bonus", 0.0))

        if bus is not None:
            # BEFORE_ACCURACY_CHECK: MUST have actor, move, targets, log
            bus.emit(
                Event.BEFORE_ACCURACY_CHECK,
                actor=actor,
                move=move,
                targets=targets,
                target=defender,
                log=log,
                ops=ops,
            )
            # ACCURACY_CHECK: MUST have actor, move, targets, log; MAY have chance, hit
            bus.emit(
                Event.ACCURACY_CHECK,
                actor=actor,
                move=move,
                targets=targets,
                target=defender,
                chance=chance,
                log=log,
                ops=ops,
            )

        if not rng.chance(chance):
            log(t("battle.move_missed", variables={"actor": t(f"creature.{actor.spec_key}"), "move": t(f"move.{move_key}")}))
            if bus is not None:
                # Determine cause: "evade" if defender has dodge, otherwise "accuracy"
                cause = "evade" if dodge > 0 else "accuracy"
                # ACTION_MISS: MUST have actor, move, targets, log, cause
                bus.emit(
                    Event.ACTION_MISS,
                    actor=actor,
                    move=move,
                    targets=targets,
                    target=defender,
                    cause=cause,
                    log=log,
                    ops=ops,
                )
            tags.add("miss")
            return

        if bus is not None:
            # ACTION_SUCCEEDED: MUST have actor, move, targets, log
            bus.emit(
                Event.ACTION_SUCCEEDED,
                actor=actor,
                move=move,
                targets=targets,
                target=defender,
                log=log,
                ops=ops,
            )

        ratio = float(effect.get("ratio", move.get("ratio", 1.0)))
        element = effect.get("element") or move.get("element") or actor.element
        crit_spec = effect.get("crit") or {}
        nonlethal = bool(effect.get("nonlethal", False))

        for target in targets:
            base_damage = round(float(actor.attack) * ratio) - int(target.defense)
            base_damage = max(0, base_damage)

            did_crit = False
            crit_chance = float(crit_spec.get("chance", 0.0) or 0.0)
            crit_mult = float(crit_spec.get("mult", 1.5) or 1.5)
            if crit_chance > 0 and rng.chance(max(0.0, min(1.0, crit_chance))):
                base_damage = int(round(base_damage * crit_mult))
                did_crit = True
                tags.add("crit")
                log(t("battle.critical_hit"))

            multiplier = _type_multiplier(str(element), target.element)
            final_damage = int(max(0, round(base_damage * multiplier)))

            if multiplier > 1.0:
                log(t("battle.super_effective"))
            elif 0 < multiplier < 1.0:
                log(t("battle.not_effective"))

            dealt = ops.apply_damage(
                target,
                final_damage,
                nonlethal=nonlethal,
                source=actor,
                tags=tags if did_crit else None,
            )
            log(t("battle.damage_dealt", variables={
                "target": t(f"creature.{target.spec_key}"),
                "damage": dealt,
                "hp": target.hp,
                "max_hp": target.max_hp
            }))


@dataclass(slots=True)
class StatusApplyHandler:
    namespace: str = "builtin.status_apply"
    order: int = 200

    def can_handle(self, effect: EffectSpec, ctx: ActionContext) -> bool:  # noqa: D401
        return _normalise_type(effect) in {"status", "status_apply"}

    def handle(self, effect: EffectSpec, ctx: ActionContext) -> None:  # noqa: D401
        # Determine targets based on 'target' parameter
        target_spec = str(effect.get("target", "targets")).lower()
        if target_spec in ("actor", "self"):
            # Apply to actor (self)
            targets = [ctx["actor"]]
        else:
            # Default: apply to targets
            targets = list(ctx.get("targets", []))

        if not targets:
            return
        status_key = str(effect.get("status"))
        if not status_key:
            return
        chance = float(effect.get("chance", 1.0))
        params = dict(effect.get("params", {}))
        duration_turns = effect.get("duration_turns", params.pop("duration_turns", None))
        stacks = effect.get("stacks", params.pop("stacks", None))
        allow_duplicates = bool(effect.get("allow_duplicates", False))
        effect_tags = list(effect.get("tags", []) or [])
        param_tags = params.pop("tags", [])
        if isinstance(param_tags, (list, tuple, set)):
            effect_tags.extend(param_tags)
        elif param_tags:
            effect_tags.append(str(param_tags))
        tags = tuple(dict.fromkeys(effect_tags)) if effect_tags else None
        rng = ctx["rng"]
        ops: Ops = ctx["ops"]
        log = ctx["log"]

        for target in targets:
            if not rng.chance(max(0.0, min(1.0, chance))):
                log(t("battle.status_resisted", variables={"target": t(f"creature.{target.spec_key}"), "status": t(f"status.{status_key}")}))
                continue
            if not allow_duplicates:
                already = (
                    ops.has_status(target, status_key, tags=tags)
                    if tags
                    else ops.has_status(target, status_key)
                )
                if already:
                    log(t("battle.status_already", variables={"target": t(f"creature.{target.spec_key}"), "status": t(f"status.{status_key}")}))
                    continue
            status_params = dict(params)
            ops.add_status(
                target,
                status_key,
                params=status_params,
                duration=duration_turns,
                stacks=stacks or 1,
                tags=tags,
                source=ctx.get("actor"),
            )
            log(t("battle.status_afflicted", variables={"target": t(f"creature.{target.spec_key}"), "status": t(f"status.{status_key}")}))


@dataclass(slots=True)
class ChargeHandler:
    namespace: str = "builtin.charge"
    order: int = 150

    def can_handle(self, effect: EffectSpec, ctx: ActionContext) -> bool:  # noqa: D401
        return _normalise_type(effect) == "charge"

    def handle(self, effect: EffectSpec, ctx: ActionContext) -> None:  # noqa: D401
        actor = ctx["actor"]
        ops: Ops = ctx["ops"]
        log = ctx["log"]
        move_key = ctx.get("move_key", ctx["move"].get("key", "move"))
        prepare_turns = int(effect.get("prepare_turns", 1))
        dodge_bonus = float(effect.get("defender_dodge_bonus", 0.0) or 0.0)
        flag_name = f"charge:{move_key}"
        ext_state = actor.ext.setdefault("charge_state", {})
        flag = ext_state.get(flag_name) or ops.get_flag(actor, flag_name)

        if not flag:
            ext_state[flag_name] = {"dodge_bonus": dodge_bonus}
            ops.set_flag(actor, flag_name, value={"ready": False, "dodge_bonus": dodge_bonus}, ttl_turns=prepare_turns)
            target = ctx.get("targets", [ctx.get("action").target if ctx.get("action") else None])[0]
            if target is not None:
                ops.enqueue_action(
                    actor,
                    target,
                    move_key,
                    delay_turns=max(1, prepare_turns),
                    priority=1,
                    reason="charge",
                )
            log(t("battle.charging", variables={"actor": t(f"creature.{actor.spec_key}"), "move": t(f"move.{move_key}")}))
            ctx["abort"] = True
            return

        ext_state.pop(flag_name, None)
        ops.clear_flag(actor, flag_name)
        scratch = ctx.setdefault("scratch", {})
        if dodge_bonus:
            scratch["defender_dodge_bonus"] = scratch.get("defender_dodge_bonus", 0.0) - float(dodge_bonus)
        log(t("battle.charge_released", variables={"actor": t(f"creature.{actor.spec_key}"), "move": t(f"move.{move_key}")}))


@dataclass(slots=True)
class MultiHitHandler:
    namespace: str = "builtin.multi_hit"
    order: int = 250

    def can_handle(self, effect: EffectSpec, ctx: ActionContext) -> bool:  # noqa: D401
        return _normalise_type(effect) in {"multi_hit", "multihit"}

    def handle(self, effect: EffectSpec, ctx: ActionContext) -> None:  # noqa: D401
        extra_hits = int(effect.get("extra_hits", 0))
        if extra_hits <= 0:
            return
        chance = float(effect.get("chance", 1.0))
        rng = ctx["rng"]
        if not rng.chance(max(0.0, min(1.0, chance))):
            return
        move_key = str(effect.get("move_key") or ctx.get("move_key") or ctx["move"].get("key", "move"))
        actor = ctx["actor"]
        targets = ctx.get("targets", [])
        target = targets[0] if targets else ctx.get("action").target if ctx.get("action") else None
        if target is None:
            return
        ops: Ops = ctx["ops"]
        log = ctx["log"]
        for _ in range(extra_hits):
            ops.enqueue_action(
                actor,
                target,
                move_key,
                delay_turns=0,
                priority=1,
                reason="multi_hit",
            )
        log(t("battle.multi_hit", variables={"actor": t(f"creature.{actor.spec_key}"), "move": t(f"move.{move_key}"), "hits": extra_hits + 1}))


BUILTIN_HANDLERS: tuple[EffectHandler, ...] = (
    ChargeHandler(),
    DamageHandler(),
    StatusApplyHandler(),
    MultiHitHandler(),
)
