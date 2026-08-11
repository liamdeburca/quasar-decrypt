__all__ = [
    "ContiguousMaskedCoords",
    "MaskedCoords",
    "ReadOnlyMaskedCoords",
]

from collections.abc import Callable
from dataclasses import field
from functools import wraps
from typing import Self, TypeVar

from numpy import float64, log, zeros
from pydantic.dataclasses import dataclass
from quasar_typing.misc import BackgroundFlux
from quasar_typing.numpy import Array_, BoolVector, CoordsTuple, FloatVector

A = TypeVar("A", bound=Array_)
P = TypeVar("P", bound=Callable)


def formatted_property[P, A](p: Callable[[P], A]) -> A:
    @wraps(p)
    def formatted_property(self: "MaskedCoords"):
        key = p.__name__
        if key not in self._cache:
            self._cache[key] = self._format(p(self))
        return self._cache[key]

    formatted_property.__wrapped__ = p
    return property(formatted_property)


@dataclass(frozen=True)
class MaskedCoords:
    spec: object
    mask: BoolVector
    bg_flux: BackgroundFlux | None = None

    _cache: dict[str, Array_] = field(default_factory=dict, kw_only=True)

    def __post_init__(self):
        assert self.spec._x.size == self.mask.size

    def __getstate__(self) -> dict:
        return {
            "spec": self.spec,
            "mask": self.mask,
            "bg_flux": self.bg_flux,
        }

    def __tuple__(self) -> CoordsTuple:
        return CoordsTuple(self.x, self.y, self.dy)

    def copy(self) -> Self:
        return self.__class__(
            self.spec,
            self.mask.copy(),
            bg_flux=self.bg_flux,
        )

    def updateMask(
        self, 
        new_mask: BoolVector,
        inplace: bool = False,
    ) -> Self | None:
        obj = self if inplace else self.copy()
        obj.mask = new_mask
        return None if inplace else obj

    def ensureLogValid(
        self,
        inplace: bool = False,
    ) -> Self:
        """
        Updates the 'mask' attribute to ensure that all y-values are positive.
        """
        new_mask = self.mask.copy()
        new_mask[self.mask] &= (self.y > 0)
        return self.updateMask(new_mask, inplace=inplace)

    @classmethod
    def _format(cls, arr: A) -> A:
        return arr

    @property
    def size(self) -> int:
        return self.mask.sum()

    @formatted_property
    def x(self) -> FloatVector:
        return self.spec._x[self.mask]

    @property
    def _y_bg(self) -> FloatVector:
        if self.bg_flux is None:
            return zeros(self.mask.size, dtype=float64)

        return sum(getattr(self.spec, f"_y_{b}") for b in self.bg_flux)

    @formatted_property
    def y_bg(self) -> FloatVector:
        return self._y_bg[self.mask]

    @formatted_property
    def dx(self) -> FloatVector:
        return self.spec._dx[self.mask]

    @formatted_property
    def y(self) -> FloatVector:
        return (self.spec._y - self._y_bg)[self.mask]

    @formatted_property
    def dy(self) -> FloatVector:
        return self.spec._dy[self.mask]

    @formatted_property
    def y_smooth(self) -> FloatVector:
        return self.spec._y_smooth[self.mask]

    @formatted_property
    def y_pl(self) -> FloatVector:
        return self.spec._y_pl[self.mask]

    @formatted_property
    def y_fe(self) -> FloatVector:
        return self.spec._y_fe[self.mask]

    @formatted_property
    def y_ba(self) -> FloatVector:
        return self.spec._y_ba[self.mask]

    @formatted_property
    def y_hg(self) -> FloatVector:
        return self.spec._y_hg[self.mask]

    @formatted_property
    def y_em(self) -> FloatVector:
        return self.spec._y_em[self.mask]

    @formatted_property
    def rejected_pixels(self) -> BoolVector:
        return self.spec._rejected_pixels[self.mask]

    @formatted_property
    def absorbed_pixels(self) -> BoolVector:
        return self.spec._absorbed_pixels[self.mask]

    @formatted_property
    def valid_pixels(self) -> BoolVector:
        return self.spec._valid_pixels[self.mask]

    @formatted_property
    def log_valid_pixels(self) -> BoolVector:
        return self.spec._log_valid_pixels[self.mask]

    @formatted_property
    def p_absorbed(self) -> FloatVector:
        return self.spec._p_absorbed[self.mask]

    @formatted_property
    def x_log(self) -> FloatVector:
        return self.spec._x_log[self.mask]

    @formatted_property
    def y_log(self) -> FloatVector:
        return log(self.y / self.spec.y0)

    @formatted_property
    def dy_log(self) -> FloatVector:
        return self.dy / self.y


@dataclass(frozen=True)
class ReadOnlyMaskedCoords(MaskedCoords):
    """
    This class containts read-only arrays of a masked `SpecData` instance.
    """

    @classmethod
    def format_property(cls, arr: Array_) -> Array_:
        _arr = arr.copy()
        _arr.setflags(
            write=False,
        )
        return _arr


@dataclass(frozen=True)
class ContiguousMaskedCoords(MaskedCoords):
    """
    This class containts contiguous, read-only arrays of a masked `SpecData`
    instance.
    """

    @classmethod
    def format_property(cls, arr: Array_) -> Array_:
        _arr = arr.copy()
        _arr.setflags(
            write=False,
            align=True,
        )
        return _arr
