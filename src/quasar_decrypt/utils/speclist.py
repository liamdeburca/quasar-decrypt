from abc import abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import TypeVar

from numpy import bool_, zeros_like
from quasar_typing.bounds import CoordBounds
from quasar_typing.numpy import BoolVector

from .specdata import SpecData

T = TypeVar("T", bound="SpecData")

logger = getLogger(__name__)


@dataclass(init=False)
class SpecList(SpecData, list[T]):
    def __str__(self, simple: bool = False) -> str:
        s = f"'{self.__class__.__name__}' class ["
        s += " | ".join("{:.1f} <-> {:.1f}".format(*window.x_bounds) for window in self)
        s += "]"

        if not simple:
            s += f" w/ {self.rejected_pixels.sum()}/{self.size} (rej.) "
            s += f"{self.absorbed_pixels.sum()}/{self.size} (abs.) "
            s += f"{self.valid_pixels.sum()}/{self.size} (val.) "
            s += f"{self.log_valid_pixels.sum()}/{self.size} (log-val.)"

        return s + "."

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @property
    def mask(self) -> BoolVector:
        mask = zeros_like(self._x, dtype=bool_)
        for window in self:
            mask |= window.mask
        return mask

    @property
    def window_bounds(self) -> list[CoordBounds]:
        return [window.x_bounds for window in self]

    @property
    def window_sizes(self) -> list[int]:
        return [window.size for window in self]

    @property
    @abstractmethod
    def sample(self):
        pass
