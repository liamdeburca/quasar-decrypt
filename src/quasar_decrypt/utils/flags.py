__all__ = [
    "_Flag", 
    "_Flags",
]

from dataclasses import fields
from typing import Any, ClassVar, TypeVar

from pydantic.dataclasses import dataclass

T = TypeVar("T", bound="_Flag")

@dataclass(kw_only=True, repr=False)
class _Flag:
    """Base class for flags."""
    success: bool = False

    desc: ClassVar[str]

    def __str__(self) -> str:
        return sum(
            (f"\n- {name}: {value}" for name, value in self.fields().items()),
            start=f"{self.__class__.__name__}: {self.desc}",
        )

    def __repr__(self) -> str:
        return self.__str__()

    def fields(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(kw_only=True, repr=False)
class _Flags[T]:
    """Base class for flags."""
    desc: ClassVar[str]
    
    def __str__(self) -> str:
        s = f"{self.__class__.__name__}: {self.desc}"
        for flag in self.fields().values():
            s += f"\n- {flag.__class__.__name__}: {flag.desc}"
            for name, value in flag.fields().items():
                s += f"\n  - {name}: {value}"
        return s

    def __repr__(self) -> str:
        return self.__str__()

    def fields(self) -> dict[str, T]:
        return {field.name: getattr(self, field.name) for field in fields(self)}