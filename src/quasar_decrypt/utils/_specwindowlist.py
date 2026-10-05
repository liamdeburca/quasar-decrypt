from abc import abstractmethod
from dataclasses import field
from typing import Any, ClassVar, Self, TypeVar

from numpy import bool_, zeros_like
from pydantic.dataclasses import dataclass
from quasar_typing.bounds import CoordBounds
from quasar_typing.misc import BackgroundFlux
from quasar_typing.numpy import BoolVector
from quasar_utils.decorators import validate_call

from ._spec import _Spec
from ._specwindow import _SpecWindow

T = TypeVar("T", bound="_SpecWindow")    


@dataclass
class _SpecWindowList(_SpecWindow, list[T]):
    """
    Class for storing lists of spectral windows. 

    Has no inherent attributes, but points to a '_Spec' object.
    """

    get_mask: None = field(default=None, init=False)
    get_mask_original: None = field(default=None, init=False)

    default_bg: ClassVar[BackgroundFlux]

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["windows"] = list(self)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.clear()
        self.extend(state["windows"])

    def _finalize_unpickling(self, spectrum: _Spec) -> None:
        """
        Set the `spectrum` for this object and all windows after unpickling.
        """
        super()._finalize_unpickling(spectrum)
        for window in self:
            window._finalize_unpickling(spectrum)

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

    @classmethod
    @validate_call
    def create(cls, *, spectrum: _Spec) -> Self:
        return cls(
            spectrum=spectrum,
            x_bounds=spectrum.x_bounds,
        )

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @property
    def mask(self) -> BoolVector:
        mask = zeros_like(self._x, dtype=bool_, order="C")
        for window in self:
            mask |= window.mask
        return mask

    @property
    def mask_original(self) -> BoolVector:
        mask = zeros_like(self._x_original, dtype=bool_, order="C")
        for window in self:
            mask |= window.mask_original
        return mask

    @property
    def window_bounds(self) -> list[CoordBounds]:
        return [window.x_bounds for window in self]

    @property
    def window_sizes(self) -> list[int]:
        return [window.size for window in self]

    @property
    def window_sizes_original(self) -> list[int]:
        return [window.size_original for window in self]

    @property
    @abstractmethod
    def sample(self):
        pass
