from logging import getLogger
from abc import abstractmethod
from typing import TypeVar
from numpy import zeros_like, bool_
from dataclasses import dataclass

from quasar_typing.numpy import BoolVector
from quasar_typing.bounds import CoordBounds

from .specdata import SpecData

T = TypeVar('T', bound='SpecData')

logger = getLogger(__name__)

@dataclass(init=False)
class SpecList(SpecData, list[T]):
    def __str__(self, simple: bool = False) -> str:

        s = f"'{self.__class__.__name__}' class [" \
            + ' | '.join([
                "{:.1f} <-> {:.1f}".format(*window.x_bounds) \
                for window in self
            ]) \
            + ']'
        
        if not simple:
            s += " w/ {}/{n} (rej.) {}/{n} (abs.) {}/{n} " \
                "(val.) {}/{n} (log-val.)" \
                    .format(
                        self.rejected_pixels.sum(), 
                        self.absorbed_pixels.sum(), 
                        self.valid_pixels.sum(), 
                        self.log_valid_pixels.sum(), 
                        n = self.size,
                    )

        return s + '.'
    
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