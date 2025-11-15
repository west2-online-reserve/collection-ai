from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar

__all__ = [
    "Registry",
    "type_registry",
    "move_registry",
    "creatures_registry",
    "status_registry",
    "passive_registry",
]

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self) -> None:
        self._: dict[str, T] = {}

    def register(self, key: str, item: T) -> None:
        self._[key] = item

    def get(self, key: str) -> T:
        return self._[key]

    def has(self, key: str) -> bool:
        return key in self._

    def items(self) -> Iterable[tuple[str, T]]:
        return self._.items()

    def keys(self) -> Iterable[str]:
        return self._.keys()

    def values(self) -> Iterable[T]:
        return self._.values()

    def __len__(self) -> int:
        return len(self._)

    def __iter__(self) -> Iterator[tuple[str, T]]:
        return iter(self._.items())


type_registry      : Registry[dict]   = Registry()       # element_key -> element_spec
move_registry      : Registry[dict]   = Registry()       # move_key -> move_spec
creatures_registry : Registry[dict]   = Registry()       # creature_key -> creature_spec
status_registry    : Registry[dict]   = Registry()       # status_key -> status_spec
passive_registry   : Registry[object] = Registry()       # name -> callable (passive hook)
