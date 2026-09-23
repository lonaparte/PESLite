"""Attribute containers for the ``state`` / ``inp`` / ``out`` records of a subsystem."""

from __future__ import annotations

__all__ = ["Bag", "Empty"]


class Bag:
    """Record whose ``__slots__`` fields default to ``0.0`` unless given as keywords."""

    __slots__ = ()

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs.get(name, 0.0))


class Empty(Bag):
    """Record with no fields."""

    __slots__ = ()
