from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Deque

from pkmninsh.model.creature import Creature

__all__ = ["Action", "TurnManager"]


@dataclass(slots=True)
class Action:
    """
    Represents a single action in battle.

    Attributes:
        actor: The creature performing the action
        target: The creature receiving the action
        move_key: Identifier of the move being used
        reason: Context for why this action was queued (e.g., "normal", "counter")
    """
    actor: Creature
    target: Creature
    move_key: str
    reason: str = "normal"


class TurnManager:
    """
    Manages action queue and per-turn flags.

    Supports reactive actions (counters, chains) via a double-ended queue.
    Turn-scoped flags allow plugins to implement once-per-turn effects.
    """

    def __init__(self) -> None:
        """Initialize empty action queue and flag storage."""
        self.queue: Deque[Action] = deque()
        self._flags: dict[tuple[int, str], bool] = {}

    @property
    def flags(self) -> dict[tuple[int, str], bool]:
        """Expose internal flag storage for legacy integrations."""

        return self._flags

    def enqueue(self, action: Action) -> None:
        """
        Add an action to the end of the queue.

        Args:
            action: Action to append
        """
        self.queue.append(action)

    def enqueue_front(self, action: Action) -> None:
        """
        Add an action to the front of the queue (priority action).

        Args:
            action: Action to prepend
        """
        self.queue.appendleft(action)

    def set_flag(self, obj: object, name: str, value: bool = True) -> None:
        """
        Set a per-turn flag on an object.

        Args:
            obj: Object to associate the flag with
            name: Flag name
            value: Optional flag value, defaults to True
        """
        self._flags[(id(obj), name)] = value

    def get_flag(self, obj: object, name: str) -> bool:
        """
        Retrieve a flag value.

        Args:
            obj: Object to query
            name: Flag name

        Returns:
            The stored flag value, or False if not found
        """
        return self._flags.get((id(obj), name), False)

    def has_flag(self, obj: object, name: str) -> bool:
        """
        Check whether a flag exists on an object.

        Args:
            obj: Object to query
            name: Flag name

        Returns:
            True if flag exists, False otherwise
        """
        return (id(obj), name) in self._flags

    def unset_flag(self, obj: object, name: str) -> None:
        """
        Remove a specific flag from an object.

        Args:
            obj: Object whose flag should be removed
            name: Flag name
        """
        self._flags.pop((id(obj), name), None)

    def unset_flags_by_name(self, name: str) -> None:
        """
        Remove all flags that match a specific name.

        Args:
            name: Flag name to remove
        """
        to_del = [k for k in self._flags.keys() if k[1] == name]
        for k in to_del:
            self._flags.pop(k, None)

    def unset_all_flags(self) -> None:
        """Remove all flags for all objects."""
        self._flags.clear()
