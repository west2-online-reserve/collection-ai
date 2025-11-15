from __future__ import annotations

from typing import Any

from pkmninsh.engine.events import Event
from pkmninsh.engine.plugin_loader import register_status_handler
from pkmninsh.engine.statuses import StatusContext, StatusState
from pkmninsh.engine.turns import Action
from pkmninsh.model.creature import Creature


class SimpleStatusHandler:
    def on_apply(self, ctx: StatusContext, state: StatusState) -> None:  # noqa: D401
        return None

    def on_tick(self, ctx: StatusContext, state: StatusState, phase: str) -> None:  # noqa: D401
        return None

    def on_remove(self, ctx: StatusContext, state: StatusState, reason: str | None) -> None:  # noqa: D401
        return None

    def on_hit(self, ctx: StatusContext, state: StatusState, *, amount: int | float) -> int | float:  # noqa: D401
        return amount

    def on_being_hit(self, ctx: StatusContext, state: StatusState, *, amount: int | float) -> int | float:  # noqa: D401
        return amount


class BurnHandler(SimpleStatusHandler):
    def on_tick(self, ctx: StatusContext, state: StatusState, phase: str) -> None:
        if phase != "turn_end":
            return
        ops = ctx.get("ops")
        target = ctx.get("target")
        if not ops or not target or not target.is_alive():
            return
        amount = int(state.get("params", {}).get("damage", 10))
        dealt = ops.apply_damage(target, amount, source=state.get("params", {}).get("source"))
        if dealt > 0:
            ops.log(f"{target.name} 因烧伤损失 {dealt} HP。")


class PoisonHandler(SimpleStatusHandler):
    def on_tick(self, ctx: StatusContext, state: StatusState, phase: str) -> None:
        if phase != "turn_end":
            return
        ops = ctx.get("ops")
        target = ctx.get("target")
        if not ops or not target or not target.is_alive():
            return
        percent = float(state.get("params", {}).get("percent", 0.10))
        amount = max(1, int(round(target.max_hp * percent)))
        dealt = ops.apply_damage(target, amount, source=state.get("params", {}).get("source"))
        if dealt > 0:
            ops.log(f"{target.name} 因中毒损失 {dealt} HP。")


class LeechSeedHandler(SimpleStatusHandler):
    def on_tick(self, ctx: StatusContext, state: StatusState, phase: str) -> None:
        if phase != "turn_end":
            return
        ops = ctx.get("ops")
        target = ctx.get("target")
        if not ops or not target or not target.is_alive():
            return
        params = state.get("params", {})
        percent = float(params.get("percent", 0.10))
        amount = max(1, int(round(target.max_hp * percent)))
        dealt = ops.apply_damage(target, amount, source=params.get("source"))
        source_creature = params.get("source")
        if dealt > 0:
            if source_creature and source_creature.is_alive():
                healed = ops.heal(source_creature, dealt, source=target)
                ops.log(f"{target.name} 被寄生种子吸取 {dealt} HP，{source_creature.name} 获得治疗 {healed} HP。")
            else:
                ops.log(f"{target.name} 被寄生种子吸取 {dealt} HP。")


class ShieldHandler(SimpleStatusHandler):
    def on_being_hit(self, ctx: StatusContext, state: StatusState, *, amount: int | float) -> int | float:
        ops = ctx.get("ops")
        target = ctx.get("target")
        if not ops or not target:
            return amount
        params = state.get("params", {})
        ratio = max(0.0, min(1.0, float(params.get("ratio", 0.5))))
        original = float(amount)
        reduced = original * (1.0 - ratio)
        if reduced < original:
            absorbed = int(round(original - reduced))
            ops.log(f"{target.name} 的护盾抵消了 {absorbed} 点伤害。")
        return max(0.0, reduced)


def register_passives(bus, tm):
    register_status_handler("burn", BurnHandler())
    register_status_handler("poison", PoisonHandler())
    register_status_handler("leech_seed", LeechSeedHandler())
    register_status_handler("shield", ShieldHandler())

    def on_miss(actor=None, targets=None, target=None, move=None, cause=None, log=None, rng=None, ops=None, **_):
        defender = target or (targets[0] if targets else None)
        attacker = actor
        if not (defender and attacker and ops):
            return
        if cause != "evade":
            return
        if defender.element != "electric":
            return
        if ops.get_flag(defender, "electric_evade_react"):
            return
        if not defender.moves:
            return
        ops.set_flag(defender, "electric_evade_react", True, ttl_turns=1)
        mkey = defender.moves[0]
        tm.enqueue_front(Action(actor=defender, target=attacker, move_key=mkey, reason="electric_evade"))
        if log:
            log(f"{defender.name} 闪避成功，立刻获得一次行动。")

    bus.subscribe(Event.ACTION_MISS, on_miss)

    def fire_stack(actor=None, move=None, log=None, ops=None, **_):
        """火系被动：每次成功行动后，攻击力+10%（最多4层）"""
        attacker = actor
        if not attacker or attacker.element != "fire":
            return
        if not ops:
            return
        # 使用 ops flags 管理层数
        stacks = ops.get_flag(attacker, "fire_stacks", default=0)
        if stacks >= 4:
            return
        ops.set_flag(attacker, "fire_stacks", stacks + 1)
        if log:
            log(f"{attacker.name} 火系被动：攻击力层数 +1 → {stacks + 1}")

    bus.subscribe(Event.ACTION_SUCCEEDED, fire_stack)

    def fire_damage_boost(actor=None, scratch=None, ops=None, **_):
        """在计算伤害前，根据火系层数增加攻击力"""
        if not actor or not scratch or not ops:
            return
        if actor.element != "fire":
            return
        # 从 ops flags 读取层数
        stacks = ops.get_flag(actor, "fire_stacks", default=0)
        if stacks > 0:
            # 在 scratch 中设置伤害倍率加成
            bonus = scratch.get("damage_multiplier_bonus", 1.0)
            scratch["damage_multiplier_bonus"] = bonus * (1.0 + stacks * 0.10)

    # 订阅伤害计算前的事件
    bus.subscribe(Event.BEFORE_BASE_DAMAGE, fire_damage_boost)

    def grass_heal(log=None, ops=None, player_actor=None, enemy_actor=None, **_):
        """草系被动：回合末回复10% HP"""
        if not ops:
            return
        # 从 TURN_END 事件获取战斗中的生物
        combatants = []
        if player_actor:
            combatants.append(player_actor)
        if enemy_actor:
            combatants.append(enemy_actor)

        for creature in combatants:
            if creature and creature.element == "grass" and creature.is_alive():
                heal = max(1, int(round(creature.max_hp * 0.10)))
                healed = ops.heal(creature, heal, source=creature)
                if log:
                    log(f"{creature.name} 草系被动：回合末回复 {healed} HP。")

    bus.subscribe(Event.TURN_END, grass_heal)

