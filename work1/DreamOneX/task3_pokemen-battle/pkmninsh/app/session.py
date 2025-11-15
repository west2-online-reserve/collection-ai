from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from collections import deque

from pkmninsh.engine.core import Battle, Action

from pkmninsh.engine.events import EventBus, Event
from pkmninsh.engine.registries import creatures_registry
from pkmninsh.model.creature import Creature
from pkmninsh.app.player import Player


__all__ = ["Snapshot", "GameSession"]


@dataclass(slots=True)
class Snapshot:
    """
    Immutable snapshot of current battle state.

    Provides read-only view for UI rendering without exposing mutable objects.
    """
    player_hp: int
    player_max_hp: int
    player_name: str
    enemy_hp: int
    enemy_max_hp: int
    enemy_name: str


class GameSession:
    """
    High-level game session orchestration.

    Manages the battle lifecycle and defines turn boundaries:
    - Owns and injects EventBus into Battle for unified event handling
    - One turn = player action + opponent action + TURN_END event
    - Provides UI-friendly API wrapping battle engine
    """

    def __init__(self, plugin_dirs: list[str], seed: int | None = None) -> None:
        """
        Initialize a new game session.

        Args:
            plugin_dirs: Directories to load plugin content from
            seed: Optional random seed for deterministic battles
        """
        self.bus = EventBus()
        self.battle = Battle(plugin_dirs=plugin_dirs, seed=seed, bus=self.bus, log=self.log)
        self._logs: deque[str] = deque()

        self.player_actor: Creature | None = None
        self.enemy_actor: Creature | None = None
        self.turn_no: int = 0

        # Pre-instantiate all available creatures for selection
        self._creatures: list[Creature] = [
            self.battle.make_creature(key)
            for key, _ in creatures_registry.items()
        ]

    def list_creatures(self) -> Sequence[Creature]:
        """
        Get read-only list of available creatures.

        Returns:
            Tuple of creature instances
        """
        return tuple(self._creatures)

    def snapshot(self) -> Snapshot:
        """
        Capture current battle state.

        Returns:
            Snapshot with HP and names of both combatants

        Raises:
            RuntimeError: If battle hasn't been set up yet
        """
        if self.player_actor is None or self.enemy_actor is None:
            raise RuntimeError("Battle not set up - call setup() first")

        from pkmninsh.infra.i18n import translate as t

        player = self.player_actor
        enemy = self.enemy_actor

        return Snapshot(
            player_hp=player.hp,
            player_max_hp=player.max_hp,
            player_name=t(f"creature.{player.spec_key}"),
            enemy_hp=enemy.hp,
            enemy_max_hp=enemy.max_hp,
            enemy_name=t(f"creature.{enemy.spec_key}"),
        )

    def is_over(self) -> bool:
        """
        Check if battle has ended.

        Returns:
            True if either combatant is defeated

        Raises:
            RuntimeError: If battle hasn't been set up yet
        """
        if self.player_actor is None or self.enemy_actor is None:
            raise RuntimeError("Battle not set up - call setup() first")

        return not self.player_actor.is_alive() or not self.enemy_actor.is_alive()

    def winner(self) -> str | None:
        """
        Determine battle winner.

        Returns:
            "player" if player won, "enemy" if opponent won, None if ongoing
        """
        if self.player_actor is None or self.enemy_actor is None:
            return None

        player_alive = self.player_actor.is_alive()
        enemy_alive = self.enemy_actor.is_alive()

        if player_alive and not enemy_alive:
            return "player"
        if enemy_alive and not player_alive:
            return "enemy"

        return None

    def log(self, message: str) -> None:
        """
        Record a log message.

        Args:
            message: Text to log
        """
        self._logs.append(message)

    def pop_log(self) -> str | None:
        """
        Retrieve the next log message.

        Returns:
            Next log message, or None if no logs available
        """
        if self._logs:
            return self._logs.popleft()
        return None

    def setup(self, player: Player, opponent: Player) -> None:
        """
        Creature selection phase.

        Both players choose their creatures, then TURN_BEGIN event is emitted.

        Args:
            player: First player (typically human)
            opponent: Second player (typically AI)

        Raises:
            RuntimeError: If fewer than 2 creatures available
            ValueError: If invalid creature indices returned
        """
        if len(self._creatures) < 2:
            raise RuntimeError(
                f"Cannot start battle: only {len(self._creatures)} creature(s) loaded, need at least 2"
            )

        # Player selects from all creatures
        player_index = player.choose_creature(self._creatures)
        if not 0 <= player_index < len(self._creatures):
            raise ValueError(f"Invalid creature index from player: {player_index}")

        # Opponent selects from remaining creatures
        remaining = [
            creature
            for i, creature in enumerate(self._creatures)
            if i != player_index
        ]
        opponent_index = opponent.choose_creature(remaining)
        if not 0 <= opponent_index < len(remaining):
            raise ValueError(f"Invalid creature index from opponent: {opponent_index}")

        self.player_actor = self._creatures[player_index]
        self.enemy_actor = remaining[opponent_index]

        # Initialize turn counter
        self.turn_no = 1
        self.battle.ops.turn_no = self.turn_no

        # Emit turn begin event for plugins
        # TURN_START: MUST have turn_no, rng, log, bus
        self.bus.emit(
            Event.TURN_START,
            turn_no=self.turn_no,
            rng=self.battle.rng,
            log=self.log,
            bus=self.bus,
            ops=self.battle.ops,
        )

    def step(self, player: Player, opponent: Player) -> None:
        """
        Execute one complete turn.

        Turn flow:
        1. Release delayed actions (e.g., charge moves)
        2. Player acts first (or executes pending charge)
        3. If battle not over, opponent acts
        4. Emit TURN_END event (even if battle ended mid-turn)

        Args:
            player: First player to act
            opponent: Second player to act

        Raises:
            RuntimeError: If battle hasn't been set up yet
            ValueError: If invalid move indices returned
        """
        if self.player_actor is None or self.enemy_actor is None:
            raise RuntimeError("Battle not set up - call setup() first")

        self.battle.ops.turn_no = self.turn_no
        self.battle.ops.tick_statuses("turn_start")

        # Release delayed actions (e.g., charge moves)
        self.battle.ops._release_delayed_actions()

        # Check if player has a pending action (e.g., from charge)
        pending_player_action = None
        for action in self.battle.tm.queue:
            if action.actor is self.player_actor and action.reason == "charge":
                pending_player_action = action
                break

        if pending_player_action:
            # Remove from queue and execute
            self.battle.tm.queue.remove(pending_player_action)
            from pkmninsh.infra.i18n import translate as t
            self.log(t("battle.charge_continuing", variables={
                "actor": t(f"creature.{self.player_actor.spec_key}"),
                "move": t(f"move.{pending_player_action.move_key}")
            }))
            self.battle.resolve_action(pending_player_action)
        else:
            # Player's normal action
            player_move_index = player.choose_move(self.player_actor)
            if not 0 <= player_move_index < len(self.player_actor.moves):
                raise ValueError(f"Invalid move index from player: {player_move_index}")

            player_move_key = self.player_actor.moves[player_move_index]
            self.battle.resolve_action(
                Action(
                    actor=self.player_actor,
                    target=self.enemy_actor,
                    move_key=player_move_key,
                )
            )

        # Check if battle ended after player's action
        if self.is_over():
            # TURN_END: MUST have turn_no, log; MAY have player_actor, enemy_actor, ops
            self.bus.emit(
                Event.TURN_END,
                turn_no=self.turn_no,
                log=self.log,
                player_actor=self.player_actor,
                enemy_actor=self.enemy_actor,
                ops=self.battle.ops,
            )
            self.battle.ops.tick_statuses("turn_end")
            return

        # Check if opponent has a pending action (e.g., from charge)
        pending_opponent_action = None
        for action in self.battle.tm.queue:
            if action.actor is self.enemy_actor and action.reason == "charge":
                pending_opponent_action = action
                break

        if pending_opponent_action:
            # Remove from queue and execute
            self.battle.tm.queue.remove(pending_opponent_action)
            from pkmninsh.infra.i18n import translate as t
            self.log(t("battle.charge_continuing", variables={
                "actor": t(f"creature.{self.enemy_actor.spec_key}"),
                "move": t(f"move.{pending_opponent_action.move_key}")
            }))
            self.battle.resolve_action(pending_opponent_action)
        else:
            # Opponent's normal action
            opponent_move_index = opponent.choose_move(self.enemy_actor)
            if not 0 <= opponent_move_index < len(self.enemy_actor.moves):
                raise ValueError(f"Invalid move index from opponent: {opponent_move_index}")

            opponent_move_key = self.enemy_actor.moves[opponent_move_index]
            self.battle.resolve_action(
                Action(
                    actor=self.enemy_actor,
                    target=self.player_actor,
                    move_key=opponent_move_key,
                )
            )

        # Increment turn counter and emit turn end for end-of-turn effects
        self.turn_no += 1
        self.battle.ops.turn_no = self.turn_no
        # TURN_END: MUST have turn_no, log; MAY have player_actor, enemy_actor, ops
        self.bus.emit(
            Event.TURN_END,
            turn_no=self.turn_no,
            log=self.battle.log,
            player_actor=self.player_actor,
            enemy_actor=self.enemy_actor,
            ops=self.battle.ops,
        )
        self.battle.ops.tick_statuses("turn_end")
