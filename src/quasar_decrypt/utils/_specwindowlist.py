from abc import abstractmethod
from typing import ClassVar, TypeVar

from numpy import bool_, zeros_like
from pydantic.dataclasses import dataclass
from quasar_typing.bounds import CoordBounds
from quasar_typing.misc import BackgroundFlux
from quasar_typing.numpy import BoolVector

from ._specwindow import _SpecWindow

T = TypeVar("T", bound="_SpecWindow")

@dataclass(init=False)
class _SpecWindowList[T](_SpecWindow, list[T]):
    """
    Class for storing lists of spectral windows. 

    Has no inherent attributes, but points to a '_Spectrum' object.
    """

    default_bg: ClassVar[BackgroundFlux]

    def __str__(self, simple: bool = False) -> str:
        s = f"<{self.__class__.__name__} object at {hex(id(self))}> "
        s += f"x_bounds={self.window_bounds}"
        if not simple:
            n = self.size
            s += " w/ "
            s += f"{self.n_rej}/{n} (rej.) "
            s += f"{self.n_abs}/{n} (abs.) "
            s += f"{self.n_val}/{n} (val.) "
            s += f"{self.n_logval}/{n} (log-val.)"
        return s

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