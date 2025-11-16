from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from pkmninsh.model.creature import Creature
from pkmninsh.engine.rng import RNG


__all__ = ["Player", "LocalPlayer", "Computer"]


class Player(ABC):
    """
    Abstract interface for all player types.

    Implementations include local human players, AI opponents, network players,
    and LLM-based players.

    Of course, currently only LocalPlayer and Computer are available. XD
    """

    @abstractmethod
    def choose_creature(self, candidates: Sequence[Creature]) -> int:
        """
        Select a creature from available candidates.

        Args:
            candidates: List of creatures to choose from

        Returns:
            Index of the selected creature (0-based)
        """
        ...

    @abstractmethod
    def choose_move(self, me: Creature) -> int:
        """
        Select a move to use in battle.

        Args:
            me: The creature performing the move

        Returns:
            Index of the selected move (0-based)
        """
        ...

    def notify(self, message: str) -> None:
        """
        Receive a notification message (optional to override).

        Default implementation does nothing. Subclasses can override
        to display messages to the player.

        Args:
            message: Notification text to display
        """
        pass


class LocalPlayer(Player):
    """
    Command-line interface for local human players.

    Prompts the user for input via stdin and displays information to stdout.
    Uses the i18n system for localized prompts.
    """

    def choose_creature(self, candidates: Sequence[Creature]) -> int:
        """
        Prompt user to select a creature via console input.

        Displays numbered list of creatures with their stats and validates input.

        Args:
            candidates: Available creatures to choose from

        Returns:
            Index of selected creature (0-based)
        """
        from pkmninsh.infra.i18n import translate as t

        if not candidates:
            raise ValueError("No creatures available to choose from")

        print(t("prompt.choose_creature"))

        # Display creature options with 1-based numbering for users
        for i, creature in enumerate(candidates, start=1):
            creature_name = t(f"creature.{creature.spec_key}")
            print(
                f"{i}. {creature_name} ({creature.element}) "
                f"HP {creature.hp}/{creature.max_hp}"
            )

        # Input validation loop
        while True:
            user_input = input(t("prompt.input_number")).strip()

            if not user_input.isdigit():
                print(t("prompt.invalid_input"))
                continue

            # Convert from 1-based user input to 0-based index
            index = int(user_input) - 1

            if 0 <= index < len(candidates):
                return index

            print(t("prompt.invalid_input"))

    def choose_move(self, me: Creature) -> int:
        """
        Prompt user to select a move via console input.

        Displays numbered list of available moves and validates input.

        Args:
            me: The creature whose moves to choose from

        Returns:
            Index of selected move (0-based)

        Raises:
            ValueError: If creature has no moves available
        """
        from pkmninsh.infra.i18n import translate as t

        if not me.moves:
            raise ValueError(f"Creature {me.name} has no moves available")

        # Display move options with 1-based numbering for users
        for i, move_key in enumerate(me.moves, start=1):
            move_name = t(f"move.{move_key}")
            print(f"{i}. {move_name}")

        # Input validation loop
        while True:
            user_input = input(t("prompt.choose_skill")).strip()

            if not user_input.isdigit():
                print(t("prompt.invalid_input"))
                continue

            # Convert from 1-based user input to 0-based index
            index = int(user_input) - 1

            if 0 <= index < len(me.moves):
                return index

            print(t("prompt.invalid_input"))

    def notify(self, message: str) -> None:
        """
        Display a notification message to the user.

        Args:
            message: Text to display
        """
        print(message)


class Computer(Player):
    """
    Simple AI player that makes random decisions.

    Uses a random number generator to select creatures and moves uniformly.
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        Initialize the AI player.

        Args:
            seed: Optional random seed for deterministic behavior
        """
        try:
            self.rng = RNG(seed=seed)
        except RuntimeError:
            self.rng = RNG()

    def choose_creature(self, candidates: Sequence[Creature]) -> int:
        """
        Randomly select a creature.

        Args:
            candidates: Available creatures to choose from

        Returns:
            Random valid index

        Raises:
            ValueError: If no creatures are available
        """
        if not candidates:
            raise ValueError("No creatures available to choose from")

        return self.rng.randint(0, len(candidates) - 1)

    def choose_move(self, me: Creature) -> int:
        """
        Randomly select a move.

        Args:
            me: The creature whose moves to choose from

        Returns:
            Random valid move index, or 0 if no moves available
        """
        if not me.moves:
            # Fallback to index 0 if no moves (should not happen in normal play)
            return 0

        return self.rng.randint(0, len(me.moves) - 1)
