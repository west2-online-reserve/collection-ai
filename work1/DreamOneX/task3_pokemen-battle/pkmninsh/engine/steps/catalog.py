"""Step catalog and provider protocols for action assembly.

This module is responsible for collecting step providers and resolving them
into an ordered list of executable steps for a given :class:`ActionContext`.

Providers register themselves with the catalog and are queried at runtime to
determine whether they should participate in the current action.  Resolution
never mutates the context; it only collects the callables that will later be
invoked by the action pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Protocol, runtime_checkable

from pkmninsh.engine.pipeline import ActionContext

__all__ = ["Step", "StepProvider", "StepCatalog"]


Step = Callable[[ActionContext], None]


@runtime_checkable
class StepProvider(Protocol):
    """Protocol describing how step providers participate in assembly."""

    namespace: str
    """Unique namespace identifying the provider.

    The namespace is used to prevent duplicate registrations across core and
    plugins.  Providers originating from different packages should choose
    distinct namespace strings (for example, ``"core.damage"`` or
    ``"myplugin.combo"``).
    """

    def match(self, ctx: ActionContext) -> bool:
        """Return ``True`` when the provider wishes to contribute steps.

        Implementations must treat ``ctx`` as read-only and may not perform
        side effects.  The provider is strictly responsible for orchestration
        decisions; all mutations belong to the steps executed later by the
        pipeline.
        """

    def steps(self, ctx: ActionContext) -> List[Step]:
        """Return the ordered list of steps to insert into the pipeline.

        The provider should only return callables and must not execute any
        behaviour during resolution.  Side effects occur solely when the
        returned steps are run by the pipeline.
        """

    def priority(self) -> int:
        """Return the ordering priority for this provider.

        Lower numbers are executed first.  When two providers share the same
        priority value they retain their registration order.
        """


@dataclass(slots=True)
class _RegisteredProvider:
    provider: StepProvider
    order: int


class StepCatalog:
    """Registry that resolves matching providers into executable steps."""

    def __init__(self) -> None:
        self._providers: list[_RegisteredProvider] = []
        self._namespace_index: dict[str, int] = {}

    def register(self, provider: StepProvider, *, replace: bool = False) -> None:
        """Register a provider with the catalog.

        Args:
            provider: Provider instance to register.
            replace: When ``True`` an existing provider with the same
                :attr:`StepProvider.namespace` is replaced while retaining the
                original registration order.
        """

        namespace = provider.namespace
        if namespace in self._namespace_index:
            if not replace:
                raise ValueError(f"Step provider '{namespace}' is already registered")
            index = self._namespace_index[namespace]
            self._providers[index] = _RegisteredProvider(provider=provider, order=index)
            self._namespace_index[namespace] = index
            return

        if replace:
            raise ValueError(
                f"Cannot replace step provider '{namespace}' because it is not registered"
            )

        index = len(self._providers)
        self._providers.append(_RegisteredProvider(provider=provider, order=index))
        self._namespace_index[namespace] = index

    def resolve(self, ctx: ActionContext) -> List[Step]:
        """Resolve providers for ``ctx`` and return the ordered steps.

        Resolution respects two ordering guarantees:

        1. Providers with lower :meth:`priority` values run earlier.
        2. Providers with equal priority retain their registration order.

        The catalog never deduplicates or executes steps.  It merely concatenates
        the results from all matching providers, leaving short-circuiting and
        side-effect management to the pipeline runtime.
        """

        matched: list[tuple[int, int, StepProvider]] = []
        for entry in self._providers:
            provider = entry.provider
            if provider.match(ctx):
                matched.append((provider.priority(), entry.order, provider))

        matched.sort(key=lambda item: (item[0], item[1]))

        steps: List[Step] = []
        for _, _, provider in matched:
            steps.extend(provider.steps(ctx))
        return steps
