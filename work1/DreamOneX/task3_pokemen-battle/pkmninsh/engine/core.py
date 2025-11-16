from __future__ import annotations
from typing import Callable

from pkmninsh.engine.assembler import ActionAssembler, DefaultAssembler
from pkmninsh.engine.events import EventBus, Event
from pkmninsh.engine.effects.registry import EffectRegistry, effect_registry as global_effect_registry
from pkmninsh.engine.registries import (
    move_registry,
    creatures_registry,
)
from pkmninsh.engine.rng import RNG
from pkmninsh.engine.turns import Action, TurnManager
from pkmninsh.engine.plugin_loader import load_plugins
from pkmninsh.engine.steps.catalog import StepCatalog
from pkmninsh.engine.steps.providers import (
    register_default_providers,
)
from pkmninsh.engine.ops import Ops
from pkmninsh.model.creature import Creature, build_creature


__all__ = ["Battle"]


class Battle:
    """
    Core battle engine handling combat resolution.

    Features:
    - Event-driven architecture via EventBus
    - Pluggable damage pipeline
    - Strict RNG singleton for reproducibility
    - Support for reactive actions (counters, chains)
    - Lazy plugin loading
    """

    def __init__(
        self,
        plugin_dirs: list[str] | None = None,
        seed: int | None = None,
        bus: EventBus | None = None,
        log: Callable[[str], None] = lambda msg: None,
        *,
        assembler: ActionAssembler | None = None,
        use_default_catalog: bool = True,
    ) -> None:
        """
        Initialize battle engine.

        Args:
            plugin_dirs: Directories to load plugin content from
            seed: Optional random seed for deterministic battles
            bus: Optional shared event bus
            log (Callable[[str], None]): logging function; note that if not provided, no logs will be recorded
            assembler: Optional custom ActionAssembler
            use_default_catalog: Whether to register default step providers if no assembler is provided
        Raises:
            ValueError: If RNG is already initialized with a different seed
        """
        if seed is not None:
            try:
                RNG(seed)
            except RuntimeError:
                # RNG already initialized for this process; ignore on re-seed attempts
                pass

        self.bus = bus or EventBus()
        self.rng = RNG()
        self.tm = TurnManager()
        self.log = log
        self.ops = Ops(self.tm, self.bus)

        catalog_for_plugins: StepCatalog | None = None
        effect_registry_for_plugins: EffectRegistry | None = None
        if assembler is None:
            catalog_for_plugins = StepCatalog()
            if use_default_catalog:
                register_default_providers(catalog_for_plugins)
            effect_registry_for_plugins = global_effect_registry.clone()
        elif isinstance(assembler, DefaultAssembler):
            catalog_for_plugins = assembler.catalog
            effect_registry_for_plugins = assembler.effect_registry
        else:
            candidate = getattr(assembler, "catalog", None)
            if isinstance(candidate, StepCatalog):
                catalog_for_plugins = candidate
            effect_registry_for_plugins = getattr(assembler, "effect_registry", None)

        # Lazy plugin loading to avoid import-time side effects
        if plugin_dirs:
            load_plugins(
                plugin_dirs,
                self.bus,
                self.tm,
                catalog_for_plugins,
                effect_registry_for_plugins,
            )

        if assembler is None:
            assembler = DefaultAssembler(
                catalog_for_plugins or StepCatalog(),
                effect_registry=effect_registry_for_plugins,
            )
        self.assembler = assembler
        if isinstance(self.assembler, DefaultAssembler):
            self.assembler.configure(bus=self.bus, rng=self.rng, tm=self.tm, log=self.log)
            self.assembler.use_ops(self.ops)
        elif hasattr(self.assembler, "configure"):
            self.assembler.configure(bus=self.bus, rng=self.rng, tm=self.tm, log=self.log)  # type: ignore[attr-defined]
            if hasattr(self.assembler, "use_ops"):
                getattr(self.assembler, "use_ops")(self.ops)  # type: ignore[misc]

    def make_creature(self, key: str) -> Creature:
        """
        Factory method to create a creature from registry data.

        Args:
            key: Creature identifier in registry

        Returns:
            Fully initialized Creature instance

        Raises:
            KeyError: If creature key not found in registry
        """
        return build_creature(key)

    def resolve_action(self, action: Action) -> None:
        """
        Resolve a single action, applying move effects and damage.

        Emits:
            Event.ACTION_BEGIN, Event.ACTION_HIT / Event.ACTION_MISS, Event.ACTION_END
        """
        actor = action.actor
        target = action.target
        move_key = action.move_key

        if not actor.is_alive() or not target.is_alive():
            return

        if not move_registry.has(move_key):
            raise KeyError(f"Move '{move_key}' not found in registry")

        move = move_registry.get(move_key)

        # ACTION_BEGIN: MUST have actor, move, candidates, targets, log
        self.bus.emit(
            Event.ACTION_BEGIN,
            actor=actor,
            move=move,
            candidates=(target,),  # In this simplified battle, target is the only candidate
            targets=[target],
            log=self.log,
            ops=self.ops,
        )
        ctx = self.assembler.build_context(action)
        ctx["ops"] = self.ops
        pipeline = self.assembler.build_pipeline(ctx)
        pipeline.run(ctx)

        # ACTION_END: MUST have actor, move, log; MAY have targets
        self.bus.emit(
            Event.ACTION_END,
            actor=actor,
            move=move,
            targets=[target],
            log=self.log,
            ops=self.ops,
        )

        # Process reactive actions (counters, chains, etc.)
        self._drain_reaction_queue()

    def _drain_reaction_queue(self) -> None:
        """
        Process all queued reactive actions.

        Continues until queue is empty, allowing chained reactions.
        Plugins should use TurnManager.flags to prevent infinite loops.
        """
        while self.tm.queue:
            action = self.tm.queue.popleft()
            if action.actor.is_alive() and action.target.is_alive():
                self.resolve_action(action)
