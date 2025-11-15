from __future__ import annotations

import random
from typing import Any, Sequence, TypeVar

__all__ = ["RNG"]

T = TypeVar("T")


class RNG:
    """Singleton random number generator used across the engine."""

    _instance: "RNG" | None = None
    _initialized: bool = False

    def __new__(cls, seed: int | None = None) -> "RNG":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        elif seed is not None:
            cls._instance.seed(seed)
        return cls._instance

    def __init__(self, seed: int | None = None) -> None:
        if self.__class__._initialized:
            return
        self._rng = random.Random(seed)
        self.__class__._initialized = True

    def roll(self) -> float:
        """Return a float in [0.0, 1.0)."""

        return self._rng.random()

    def chance(self, probability: float) -> bool:
        """Return ``True`` with the given probability."""

        return self._rng.random() < probability

    def pick(self, seq: Sequence[T]) -> T:
        """Return a random element from ``seq``."""

        return self._rng.choice(seq)  # type: ignore[arg-type]

    def randint(self, a: int, b: int) -> int:
        """Return a random integer in [a, b]."""

        return self._rng.randint(a, b)

    def seed(self, value: int | None = None) -> None:
        """Reset the RNG with a new seed."""

        self._rng.seed(value)
