from __future__ import annotations

from typing import Any, Iterable, List

from pkmninsh.engine.ops import Ops
from pkmninsh.engine.statuses import ModifierSpec

__all__ = ["StatResolver"]


class StatResolver:
    """Resolve effective stats after passive, status, and flag modifiers."""

    def __init__(self, ops: Ops | None = None) -> None:
        self._ops = ops

    def _collect_passive_modifiers(self, creature: Any) -> List[ModifierSpec]:
        mods = creature.ext.get("passive_modifiers") if hasattr(creature, "ext") else None
        if not isinstance(mods, Iterable):
            return []
        result: List[ModifierSpec] = []
        for mod in mods:
            if isinstance(mod, dict) and mod.get("stat") and mod.get("mode"):
                result.append(dict(mod))  # type: ignore[arg-type]
        return result

    def _collect_status_modifiers(self, creature: Any) -> List[ModifierSpec]:
        if self._ops is None:
            return []
        try:
            return list(self._ops.query_modifiers(creature))
        except AttributeError:
            return []

    def _collect_flag_modifiers(self, creature: Any) -> List[ModifierSpec]:
        if self._ops is None:
            return []
        try:
            data = self._ops.get_flag(creature, "stat_modifiers")
        except AttributeError:
            return []
        if not isinstance(data, Iterable):
            return []
        result: List[ModifierSpec] = []
        for mod in data:
            if isinstance(mod, dict) and mod.get("stat") and mod.get("mode"):
                result.append(dict(mod))  # type: ignore[arg-type]
        return result

    def _combine_caps(self, modifiers: Iterable[ModifierSpec]) -> tuple[float | None, float | None]:
        min_cap: float | None = None
        max_cap: float | None = None
        for mod in modifiers:
            caps = mod.get("caps")
            if not isinstance(caps, dict):
                continue
            if caps.get("min") is not None:
                value = float(caps["min"])
                min_cap = value if min_cap is None else max(min_cap, value)
            if caps.get("max") is not None:
                value = float(caps["max"])
                max_cap = value if max_cap is None else min(max_cap, value)
        return min_cap, max_cap

    def get_effective_stat(self, creature: Any, stat: str) -> float | int:
        base_value = getattr(creature, stat)
        value = float(base_value)

        passive_mods = self._collect_passive_modifiers(creature)
        status_mods = self._collect_status_modifiers(creature)
        flag_mods = self._collect_flag_modifiers(creature)

        ordered_mods: List[ModifierSpec] = []
        ordered_mods.extend(passive_mods)
        ordered_mods.extend(status_mods)
        ordered_mods.extend(flag_mods)

        add_total = sum(mod.get("value", 0.0) for mod in ordered_mods if mod.get("mode") == "add")
        value += add_total

        for mod in ordered_mods:
            if mod.get("mode") == "mul":
                value *= float(mod.get("value", 1.0))

        override_values = [float(mod.get("value", value)) for mod in ordered_mods if mod.get("mode") == "override"]
        if override_values:
            value = override_values[-1]

        min_cap, max_cap = self._combine_caps(ordered_mods)
        if min_cap is not None:
            value = max(min_cap, value)
        if max_cap is not None:
            value = min(max_cap, value)

        if isinstance(base_value, int):
            return int(round(value))
        return value
