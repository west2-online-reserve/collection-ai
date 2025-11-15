from __future__ import annotations

import importlib
import importlib.util
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from pkmninsh.content.loader import load_yaml
from pkmninsh.engine.effects.registry import EffectRegistry
from pkmninsh.engine.registries import (
    type_registry,
    move_registry,
    creatures_registry,
    passive_registry,
)
from pkmninsh.engine.events import EventBus
from pkmninsh.engine.turns import TurnManager
from pkmninsh.engine.steps.catalog import StepCatalog, StepProvider
from pkmninsh.engine.statuses import (
    StatusHandler,
    StatusSpec,
    register_status_spec as engine_register_status_spec,
    register_status_handler as engine_register_status_handler,
)
from pkmninsh.infra.i18n import translate as t
from pkmninsh.infra.logging import get_logger

__all__ = [
    "load_plugins",
    "PluginEnvironment",
    "register_status_spec",
    "register_status_handler",
]


logger = get_logger(__name__)


@dataclass(slots=True)
class PluginEnvironment:
    bus: EventBus
    tm: TurnManager
    step_catalog: StepCatalog | None
    effect_registry: EffectRegistry | None


def register_status_spec(spec: Mapping[str, Any]) -> StatusSpec:
    """Expose status spec registration for plugins."""

    return engine_register_status_spec(spec)


def register_status_handler(key: str, handler: StatusHandler) -> None:
    """Expose status handler registration for plugins."""

    engine_register_status_handler(key, handler)

def _warn(msg: str) -> None:
    """Emit a warning via the module logger."""
    logger.warning(msg)


def _info(msg: str) -> None:
    """Emit an informational message via the module logger."""
    logger.info(msg)


def _load_types(plugin_dir: Path, types_file: str) -> None:
    """
    Load element/type definitions from plugin.
    
    Expected YAML structure:
        { elements: { <element_key>: {strong:[], weak:[], ...} } }
    
    Args:
        plugin_dir: Root directory of the plugin
        types_file: Relative path to types YAML file
    """
    path = plugin_dir / types_file
    if not path.exists():
        return
    
    data = load_yaml(path) or {}
    elements = data.get("elements") or {}
    
    for element_key, spec in elements.items():
        try:
            type_registry.register(element_key, spec)
        except Exception as e:
            _warn(t("plugin.warn_register_type", variables={"key": element_key, "err": str(e)}))


def _load_moves(plugin_dir: Path, moves_dir_name: str) -> None:
    """
    Load move definitions from plugin.
    
    Args:
        plugin_dir: Root directory of the plugin
        moves_dir_name: Name of the moves directory
    """
    moves_dir = plugin_dir / moves_dir_name
    if not moves_dir.exists() or not moves_dir.is_dir():
        return
    
    for yaml_file in moves_dir.glob("*.yaml"):
        try:
            spec = load_yaml(yaml_file) or {}
            move_key = spec.get("key") or yaml_file.stem
            move_registry.register(move_key, spec)
        except Exception as e:
            _warn(t("plugin.warn_load_move", variables={"file": yaml_file.name, "err": str(e)}))


def _load_creatures(plugin_dir: Path, creatures_dir_name: str) -> None:
    """
    Load creature definitions from plugin.
    
    Args:
        plugin_dir: Root directory of the plugin
        creatures_dir_name: Name of the creatures directory
    """
    creatures_dir = plugin_dir / creatures_dir_name
    if not creatures_dir.exists() or not creatures_dir.is_dir():
        return
    
    for yaml_file in creatures_dir.glob("*.yaml"):
        try:
            spec = load_yaml(yaml_file) or {}
            key = spec.get("key") or yaml_file.stem
            creatures_registry.register(key, spec)
        except Exception as e:
            _warn(t("plugin.warn_load_creature", variables={"file": yaml_file.name, "err": str(e)}))


def _load_statuses(plugin_dir: Path, statuses_dir_name: str) -> None:
    """
    Load status effect definitions from plugin.
    
    Args:
        plugin_dir: Root directory of the plugin
        statuses_dir_name: Name of the statuses directory
    """
    statuses_dir = plugin_dir / statuses_dir_name
    if not statuses_dir.exists() or not statuses_dir.is_dir():
        return
    
    for yaml_file in statuses_dir.glob("*.yaml"):
        try:
            raw_spec = load_yaml(yaml_file) or {}
            if "key" not in raw_spec:
                raw_spec["key"] = yaml_file.stem
            engine_register_status_spec(raw_spec)
        except Exception as e:
            _warn(t("plugin.warn_load_status", variables={"file": yaml_file.name, "err": str(e)}))


def _import_func_from_plugin(
    plugin_dir: Path,
    module_ref: str,
    func_name: str,
) -> Callable:
    """
    Import a function from a plugin module.
    
    Strategy:
    1. Try standard import (for installed packages/namespace packages)
    2. If that fails, try loading relative to plugin directory:
       - "passives" -> <plugin_dir>/passives.py
       - "pkg.mod" -> <plugin_dir>/pkg/mod.py or <plugin_dir>/pkg/mod/__init__.py
    
    Args:
        plugin_dir: Root directory of the plugin
        module_ref: Module path (e.g., "passives" or "pkg.hooks")
        func_name: Name of the function to import
        
    Returns:
        The imported function
        
    Raises:
        ImportError: If module cannot be found
        AttributeError: If function doesn't exist in module
    """
    # Try standard import first
    try:
        mod = importlib.import_module(module_ref)
        if not hasattr(mod, func_name):
            raise AttributeError(
                f"Function '{func_name}' not found in module '{module_ref}'"
            )
        return getattr(mod, func_name)
    except (ImportError, AttributeError):
        pass  # Fall through to relative loading

    # Try loading relative to plugin directory
    module_path = module_ref.replace(".", "/")
    candidate_file = plugin_dir / f"{module_path}.py"
    candidate_pkg_init = plugin_dir / module_path / "__init__.py"

    target: Path | None = None
    if candidate_file.exists():
        target = candidate_file
    elif candidate_pkg_init.exists():
        target = candidate_pkg_init

    if target is None:
        raise ImportError(
            f"Module '{module_ref}' not found in plugin directory or sys.path"
        )

    # Load module from file
    spec = importlib.util.spec_from_file_location(module_ref, target)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot create module spec for '{module_ref}' at {target}"
        )

    mod = importlib.util.module_from_spec(spec)
    
    # Add to sys.modules to support cross-imports within plugin
    sys.modules[module_ref] = mod
    
    try:
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    except Exception as e:
        # Clean up sys.modules on failure
        sys.modules.pop(module_ref, None)
        raise ImportError(f"Failed to execute module '{module_ref}': {e}") from e

    if not hasattr(mod, func_name):
        raise AttributeError(
            f"Function '{func_name}' not found in module '{module_ref}'"
        )

    return getattr(mod, func_name)


def _load_passives(
    plugin_dir: Path,
    plugin_name: str,
    passive_paths: list[str],
    bus: EventBus,
    tm: TurnManager,
) -> int:
    """
    Load and register passive ability functions from plugin.
    
    Passive paths should be in format: "module.path:function_name"
    Example: ["passives:register_electric", "pkg.hooks:setup"]
    
    Args:
        plugin_dir: Root directory of the plugin
        plugin_name: Name of the plugin (for registration keys)
        passive_paths: List of module:function paths to load
        bus: Event bus to pass to registration functions
        tm: Turn manager to pass to registration functions
        
    Returns:
        Number of successfully registered passives
    """
    success_count = 0
    
    for ref in passive_paths:
        try:
            if ":" not in ref:
                _warn(t("plugin.warn_passive_format", variables={"ref": ref}))
                continue

            module_ref, func_name = ref.split(":", 1)

            # Import the registration function
            registration_func = _import_func_from_plugin(
                plugin_dir,
                module_ref,
                func_name,
            )

            # Call the function to register with event system
            registration_func(bus, tm)

            # Track in passive registry for debugging/statistics
            registry_key = f"{plugin_name}.{func_name}"
            passive_registry.register(registry_key, registration_func)

            success_count += 1

        except Exception as e:
            _warn(t("plugin.warn_passive_register", variables={"ref": ref, "err": str(e)}))
    
    return success_count


def _load_step_providers(
    plugin_dir: Path,
    plugin_name: str,
    entries: Iterable[Any],
    step_catalog: StepCatalog | None,
) -> int:
    """Load step providers described in plugin manifest."""

    if step_catalog is None:
        if entries:
            _warn(t("plugin.warn_step_provider_no_catalog", variables={"plugin": plugin_name}))
        return 0

    loaded = 0
    for raw in entries:
        replace = False
        if isinstance(raw, str):
            ref = raw
        elif isinstance(raw, dict):
            ref = raw.get("ref")
            replace = bool(raw.get("replace", False))
            if not ref:
                _warn(t("plugin.warn_step_provider_format", variables={"plugin": plugin_name, "ref": str(raw)}))
                continue
        else:
            _warn(t("plugin.warn_step_provider_format", variables={"plugin": plugin_name, "ref": str(raw)}))
            continue

        if ":" not in ref:
            _warn(t("plugin.warn_step_provider_format", variables={"plugin": plugin_name, "ref": ref}))
            continue

        module_ref, func_name = ref.split(":", 1)
        try:
            factory = _import_func_from_plugin(plugin_dir, module_ref, func_name)
        except Exception as exc:
            _warn(t("plugin.warn_step_provider_import", variables={"plugin": plugin_name, "ref": ref, "err": str(exc)}))
            continue

        try:
            result = factory()
            providers: list[StepProvider]
            if isinstance(result, StepProvider):  # type: ignore[misc]
                providers = [result]
            elif isinstance(result, Iterable):
                providers = [prov for prov in result if isinstance(prov, StepProvider)]  # type: ignore[misc]
            else:
                _warn(t("plugin.warn_step_provider_return", variables={"plugin": plugin_name, "ref": ref}))
                continue

            if not providers:
                _warn(t("plugin.warn_step_provider_return", variables={"plugin": plugin_name, "ref": ref}))
                continue

            for provider in providers:
                try:
                    step_catalog.register(provider, replace=replace)
                    loaded += 1
                except Exception as exc:
                    _warn(
                        t(
                            "plugin.warn_step_provider_register",
                            variables={
                                "plugin": plugin_name,
                                "namespace": getattr(provider, "namespace", repr(provider)),
                                "err": str(exc),
                            },
                        )
                    )
        except Exception as exc:
            _warn(t("plugin.warn_step_provider_exec", variables={"plugin": plugin_name, "ref": ref, "err": str(exc)}))

    return loaded


def _load_effect_handlers(
    plugin_dir: Path,
    plugin_name: str,
    entries: Iterable[Any],
    env: PluginEnvironment,
) -> int:
    if env.effect_registry is None:
        if entries:
            _warn(t("plugin.warn_effect_handler_no_registry", variables={"plugin": plugin_name}))
        return 0

    loaded = 0
    for ref in entries:
        if not isinstance(ref, str) or ":" not in ref:
            _warn(t("plugin.warn_effect_handler_format", variables={"plugin": plugin_name, "ref": str(ref)}))
            continue
        module_ref, func_name = ref.split(":", 1)
        try:
            factory = _import_func_from_plugin(plugin_dir, module_ref, func_name)
        except Exception as exc:
            _warn(t("plugin.warn_effect_handler_import", variables={"plugin": plugin_name, "ref": ref, "err": str(exc)}))
            continue

        try:
            factory(env)
            loaded += 1
        except Exception as exc:
            _warn(t("plugin.warn_effect_handler_exec", variables={"plugin": plugin_name, "ref": ref, "err": str(exc)}))

    return loaded


def _load_single_plugin(
    plugin_dir: Path,
    bus: EventBus,
    tm: TurnManager,
    step_catalog: StepCatalog | None,
    effect_registry: EffectRegistry | None,
) -> tuple[bool, int]:
    """
    Load a single plugin from a directory.
    
    Args:
        plugin_dir: Directory containing the plugin
        bus: Event bus for passive registration
        tm: Turn manager for passive registration
        
    Returns:
        Tuple of (success: bool, passive_count: int)
    """
    manifest_path = plugin_dir / "manifest.json"
    if not manifest_path.exists():
        return False, 0

    try:
        manifest: dict[str, Any] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except Exception as e:
        _warn(t("plugin.error_manifest", variables={"plugin": plugin_dir.name, "err": str(e)}))
        return False, 0

    name = manifest.get("name", plugin_dir.name)
    version = manifest.get("version", "unknown")

    _info(t("plugin.loading", variables={"name": name, "version": version}))

    # Load types/elements
    if "types" in manifest:
        _load_types(plugin_dir, manifest["types"])

    # Load content
    _load_moves(plugin_dir, manifest.get("moves", "moves"))
    _load_creatures(plugin_dir, manifest.get("creatures", "creatures"))
    _load_statuses(plugin_dir, manifest.get("statuses", "statuses"))

    # Load passives
    passive_count = 0
    passive_paths = manifest.get("passives", [])
    if passive_paths:
        passive_count = _load_passives(plugin_dir, name, passive_paths, bus, tm)

    env = PluginEnvironment(
        bus=bus,
        tm=tm,
        step_catalog=step_catalog,
        effect_registry=effect_registry,
    )

    step_providers = manifest.get("step_providers", [])
    _load_step_providers(plugin_dir, name, step_providers, step_catalog)

    effect_handlers = manifest.get("effect_handlers", [])
    _load_effect_handlers(plugin_dir, name, effect_handlers, env)

    return True, passive_count


def load_plugins(
    plugin_dirs: list[str],
    bus: EventBus,
    tm: TurnManager,
    step_catalog: StepCatalog | None = None,
    effect_registry: EffectRegistry | None = None,
) -> None:
    """
    Scan plugin directories and load all valid plugins.
    
    Plugins must contain a manifest.json file with metadata and content references.
    
    Supported content types:
    - types: Element definitions with strengths/weaknesses
    - moves: Attack and skill definitions
    - creatures: Playable creature definitions
    - statuses: Status effect definitions
    - passives: Event-driven passive abilities (Python code)
    
    Args:
        plugin_dirs: List of directory paths to scan for plugins
        bus: Event bus for passive ability registration
        tm: Turn manager for passive ability registration
        
    Warning:
        Plugins are third-party content that may execute arbitrary code.
        Only load plugins from trusted sources.
    """
    _info(t("plugin.disclaimer_top"))
    
    loaded_plugins = 0
    loaded_passives = 0

    for root in plugin_dirs:
        root_path = Path(root)

        if not root_path.exists():
            _warn(t("plugin.warn_root_missing", variables={"root": root}))
            continue

        if not root_path.is_dir():
            _warn(t("plugin.warn_root_notdir", variables={"root": root}))
            continue

        # Scan for plugin subdirectories
        for sub in root_path.iterdir():
            if not sub.is_dir():
                continue
            
            success, passive_count = _load_single_plugin(
                sub,
                bus,
                tm,
                step_catalog,
                effect_registry,
            )
            if success:
                loaded_plugins += 1
                loaded_passives += passive_count

    # Count total registered passives
    try:
        total_passives = len(passive_registry)
    except TypeError:
        # Fallback if registry doesn't implement __len__
        total_passives = sum(1 for _ in passive_registry.items())

    _info(
        t(
            "plugin.loaded_summary",
            variables={
                "plugins": loaded_plugins,
                "passives": total_passives,
            },
        )
    )
    _info(t("plugin.disclaimer_bottom"))