from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from pkmninsh.engine.registries import creatures_registry

__all__ = ["CreatureSpec", "Creature", "build_creature"]


@dataclass(frozen=True, slots=True)
class CreatureSpec:
    """
    Immutable creature specification loaded from plugin YAML.

    Represents the static definition of a creature type, including
    its stats, element type, and available moves.

    YAML structure:
        key: creature_id
        name: Display Name
        element: fire
        stats:
          max_hp: 100
          attack: 50
          defense: 30
          dodge: 0.1
        moves: [move1, move2, move3]
    """
    key: str
    name: str
    element: str
    stats: Mapping[str, int | float]
    moves: list[str]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> CreatureSpec:
        """
        Create CreatureSpec from dictionary data.

        Args:
            data: Dictionary containing creature specification

        Returns:
            Immutable CreatureSpec instance

        Raises:
            KeyError: If required fields are missing
            ValueError: If field types are invalid
        """
        return cls(
            key=str(data["key"]),
            name=str(data["name"]),
            element=str(data["element"]),
            stats=dict(data["stats"]),
            moves=list(data["moves"]),
        )

    @property
    def max_hp(self) -> int:
        """Maximum hit points."""
        return int(self.stats["max_hp"])

    @property
    def attack(self) -> int:
        """Attack stat for damage calculation."""
        return int(self.stats["attack"])

    @property
    def defense(self) -> int:
        """Defense stat for damage reduction."""
        return int(self.stats["defense"])

    @property
    def dodge(self) -> float:
        """Dodge chance (0.0 to 1.0)."""
        return float(self.stats["dodge"])


@dataclass(slots=True, weakref_slot=True)
class Creature:
    """
    Runtime battle entity instance.

    Mutable instance created from CreatureSpec for actual battle usage.
    Tracks current HP, status, and plugin-specific extensions.

    Attributes:
        spec_key: Reference to original CreatureSpec for serialization
        name: Display name
        element: Element type (fire, water, etc.)
        max_hp: Maximum hit points
        attack: Attack stat
        defense: Defense stat
        dodge: Dodge chance (0.0 to 1.0)
        moves: Available move keys
        hp: Current hit points (initialized to max_hp)
        fainted: Whether creature has fainted
        ext: Extension dictionary for plugin data (shields, buffs, etc.)
    """
    spec_key: str
    name: str
    element: str
    max_hp: int
    attack: int
    defense: int
    dodge: float
    moves: list[str]

    hp: int = field(init=False)
    fainted: bool = field(default=False, init=False)
    ext: dict[str, Any] = field(default_factory=dict)
    statuses: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize HP to maximum on creation."""
        self.hp = self.max_hp

    def is_alive(self) -> bool:
        """
        Check if creature is still able to battle.

        Returns:
            True if creature has not fainted and has HP remaining
        """
        return not self.fainted and self.hp > 0

    def apply_damage(self, amount: int) -> None:
        """
        Apply damage to the creature.

        Reduces HP by the specified amount, minimum 0.
        Automatically marks creature as fainted if HP reaches 0.

        Args:
            amount: Damage amount (negative or zero values are ignored)
        """
        if amount <= 0:
            return

        self.hp = max(0, self.hp - amount)

        if self.hp == 0:
            self.fainted = True

    def heal(self, amount: int) -> None:
        """
        Restore hit points to the creature.

        Increases HP by the specified amount, maximum max_hp.
        Only works if creature is still alive.

        Args:
            amount: Healing amount (negative or zero values are ignored)
        """
        if amount <= 0 or not self.is_alive():
            return

        self.hp = min(self.max_hp, self.hp + amount)


def build_creature(spec_key: str) -> Creature:
    """
    Create a runtime Creature instance from registry specification.

    Factory function that loads creature data from the global registry
    and instantiates a battle-ready Creature object.

    Args:
        spec_key: Creature identifier in creature_registry

    Returns:
        Fully initialized Creature instance ready for battle

    Raises:
        KeyError: If spec_key not found in registry
        ValueError: If creature data is invalid

    Example:
        >>> alpha = build_creature("alpha")
        >>> print(f"{alpha.name}: {alpha.hp}/{alpha.max_hp} HP")
    """
    raw_data = creatures_registry.get(spec_key)

    # Handle both CreatureSpec and raw dictionary formats
    if isinstance(raw_data, CreatureSpec):
        spec = raw_data
    else:
        spec = CreatureSpec.from_mapping(raw_data)

    return Creature(
        spec_key=spec.key,
        name=spec.name,
        element=spec.element,
        max_hp=spec.max_hp,
        attack=spec.attack,
        defense=spec.defense,
        dodge=spec.dodge,
        moves=list(spec.moves),
    )
