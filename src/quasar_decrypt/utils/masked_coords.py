__all__ = [
    "ContiguousMaskedCoords",
    "MaskedCoords",
    "OriginalContiguousMaskedCoords",
    "OriginalMaskedCoords",
    "OriginalReadOnlyMaskedCoords",
    "ReadOnlyMaskedCoords",
]
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import field
from functools import wraps
from typing import ClassVar, Literal, Self, TypeVar

import numpy as np
import pydantic
from quasar_typing.numpy import (
    Array,
    BoolVector,
    CoordsTuple,
    FloatMatrix,
    FloatVector,
)
from quasar_typing.scipy import CSRMatrix
from quasar_typing.string_selections import BackgroundFlux

A = TypeVar("A", bound=Array)
P = TypeVar("P", bound=Callable)


def formatted_property[P, A](p: Callable[[P], A]) -> A:
    @wraps(p)
    def formatted_property(self: "BaseMaskedCoords"):
        key = p.__name__
        if key not in self._cache:
            self._cache[key] = self._format(p(self))
        return self._cache[key]

    formatted_property.__wrapped__ = p
    return property(formatted_property)


@pydantic.dataclasses.dataclass
class BaseMaskedCoords(ABC):
    """
    This class contains arrays of a masked `SpecData` instance.
    """
    spec: object
    primary_mask: BoolVector
    secondary_mask: BoolVector
    bg_flux: BackgroundFlux | None = None

    _cache: dict[str, Array] = field(default_factory=dict, kw_only=True)
    mask: BoolVector = field(init=False)

    _original: ClassVar[bool] = False
    _suffix: ClassVar[str] = ""

    def __post_init__(self):
        # Calculate the combined mask
        self.mask = self.primary_mask & self.secondary_mask
        n = self._x.size
        assert n == self.mask.size

    @classmethod
    @abstractmethod
    def _format(cls, arr: Array | None) -> Array | None:
        pass

    ### Context

    def __enter__(self, mode: Literal["primary", "secondary"]) -> Self:
        if mode == "primary":
            return self.__class__(
                self.spec,
                self.primary_mask,
                np.ones_like(self._x, dtype=np.bool_),
                bg_flux=self.bg_flux,
            )
        else:
            return self.__class__(
                self.spec,
                np.ones_like(self._x, dtype=np.bool_),
                self.secondary_mask,
                bg_flux=self.bg_flux,
            )

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    ###

    def __getstate__(self) -> dict:
        return {
            "spec": self.spec,
            "primary_mask": self.primary_mask,
            "secondary_mask": self.secondary_mask,
            "bg_flux": self.bg_flux,
        }

    def __tuple__(self) -> CoordsTuple:
        return (self.x, self.y, self.dy)

    def copy(self) -> Self:
        return self.__class__(
            self.spec,
            self.primary_mask.copy(),
            self.secondary_mask.copy(),
            bg_flux=self.bg_flux,
        )

    def updatePrimaryMask(self, new_mask: BoolVector) -> Self:
        return self.__class__(
            self.spec,
            new_mask,
            self.secondary_mask.copy(),
            bg_flux=self.bg_flux,
        )

    def updateSecondaryMask(self, new_mask: BoolVector) -> Self:
        return self.__class__(
            self.spec,
            self.primary_mask.copy(),
            new_mask,
            bg_flux=self.bg_flux,
        )

    def ensureLogValid(self) -> Self:
        """
        Update the 'secondary_mask' to ensure that all y-values are positive.
        """
        new_mask = self.secondary_mask & (self._y > 0)
        return self.updateSecondaryMask(new_mask)

    def limitLoss(self, shrink: bool = True) -> Self:
        """
        Limit the flux density loss at the edges of the spectrum due to 
        instrumental resolution. 

        If `shrink` is True, the secondary mask is shrunk. Otherwise, the 
        primary mask is expanded.

        Notes
        -----
        This method is mostly useful for fitting with resolution correction. In 
        this context, "shrinking" corresponds to restricting your data-model 
        comparison to the central, well-resolved portion of the spectrum. Using 
        `shrink=False` corresponds to expanding the primary mask, i.e. where 
        the model is evaluated, to an extent where the edges of the secondary 
        mask are well-resolved. 
        """
        from quasar_utils.resolving import _limit_loss

        _mask = _limit_loss(self._kernels, self.primary_mask, shrink)
        if shrink:
            primary_mask = self.primary_mask.copy()
            secondary_mask = self.secondary_mask & _mask # Contraction
        else:
            primary_mask = self.primary_mask | _mask # Expansion
            secondary_mask = self.secondary_mask.copy()

        return self.__class__(
            self.spec,
            primary_mask,
            secondary_mask,
            bg_flux=self.bg_flux,
        )

    ###

    @property
    def size(self) -> int:
        return self.mask.sum()

    ###

    @property
    def _x(self) -> FloatVector:
        return getattr(self.spec, "_x" + self._suffix)

    @formatted_property
    def x(self) -> FloatVector:
        return self._x[self.mask]

    ###

    @property
    def _y_bg(self) -> FloatVector:
        if self.bg_flux is None:
            return np.zeros(self.mask.size, dtype=np.float64)

        return sum(
            getattr(self.spec, f"_y_{b}{self._suffix}") 
            for b in self.bg_flux
        )

    @formatted_property
    def y_bg(self) -> FloatVector:
        return self._y_bg[self.mask]

    ###

    @property
    def _dx(self) -> FloatVector:
        return getattr(self.spec, "_dx" + self._suffix)

    @formatted_property
    def dx(self) -> FloatVector:
        return self._dx[self.mask]

    ###

    @property
    def _y(self) -> FloatVector:
        return getattr(self.spec, "_y" + self._suffix) - self._y_bg

    @formatted_property
    def y(self) -> FloatVector:
        return self._y[self.mask]

    ###

    @property
    def _dy(self) -> FloatVector:
        return getattr(self.spec, "_dy" + self._suffix)

    @formatted_property
    def dy(self) -> FloatVector:
        return self._dy[self.mask]

    ###

    @property
    def _kernels(self) -> FloatMatrix | CSRMatrix:
        return getattr(self.spec, "_kernels" + self._suffix)

    @formatted_property
    def kernels(self) -> FloatMatrix | CSRMatrix:
        from quasar_utils.resolving import _mask_kernels
        return _mask_kernels(self._kernels, self.mask)

    ###

    @property
    def _y_smooth(self) -> FloatVector:
        return getattr(self.spec, "_y_smooth" + self._suffix)

    @formatted_property
    def y_smooth(self) -> FloatVector:
        return self._y_smooth[self.mask]

    ###

    @property
    def _y_pl(self) -> FloatVector:
        return getattr(self.spec, "_y_pl" + self._suffix)

    @formatted_property
    def y_pl(self) -> FloatVector:
        return self._y_pl[self.mask]

    ###

    @property
    def _y_fe(self) -> FloatVector:
        return getattr(self.spec, "_y_fe" + self._suffix)

    @formatted_property
    def y_fe(self) -> FloatVector:
        return self._y_fe[self.mask]

    ###

    @property
    def _y_ba(self) -> FloatVector:
        return getattr(self.spec, "_y_ba" + self._suffix)

    @formatted_property
    def y_ba(self) -> FloatVector:
        return self._y_ba[self.mask]

    ###

    @property
    def _y_hg(self) -> FloatVector:
        return getattr(self.spec, "_y_hg" + self._suffix)

    @formatted_property
    def y_hg(self) -> FloatVector:
        return self._y_hg[self.mask]

    ###

    @property
    def _y_em(self) -> FloatVector:
        return getattr(self.spec, "_y_em" + self._suffix)

    @formatted_property
    def y_em(self) -> FloatVector:
        return self._y_em[self.mask]

    ###

    @property
    def _rejected_pixels(self) -> BoolVector:
        return getattr(self.spec, "_rejected_pixels" + self._suffix)

    @formatted_property
    def rejected_pixels(self) -> BoolVector:
        return self._rejected_pixels[self.mask]

    ###

    @property
    def _absorbed_pixels(self) -> BoolVector:
        return getattr(self.spec, "_absorbed_pixels" + self._suffix)

    @formatted_property
    def absorbed_pixels(self) -> BoolVector:
        return self._absorbed_pixels[self.mask]

    ###

    @property
    def _valid_pixels(self) -> BoolVector:
        return getattr(self.spec, "_valid_pixels" + self._suffix)

    @formatted_property
    def valid_pixels(self) -> BoolVector:
        return self._valid_pixels[self.mask]

    ###

    @property
    def _log_valid_pixels(self) -> BoolVector:
        return getattr(self.spec, "_log_valid_pixels" + self._suffix)

    @formatted_property
    def log_valid_pixels(self) -> BoolVector:
        return self._log_valid_pixels[self.mask]

    ###

    @property
    def _p_absorbed(self) -> FloatVector:
        return getattr(self.spec, "_p_absorbed" + self._suffix)

    @formatted_property
    def p_absorbed(self) -> FloatVector:
        return self._p_absorbed[self.mask]

    ###

    @property
    def _x_log(self) -> FloatVector:
        return getattr(self.spec, "_x_log" + self._suffix)

    @formatted_property
    def x_log(self) -> FloatVector:
        return self._x_log[self.mask]

    ###

    @property
    def _y_log(self) -> FloatVector:
        return np.log(self._y / self.spec.y0)

    @formatted_property
    def y_log(self) -> FloatVector:
        assert self.y.min() > 0.0, self.y.min()
        return np.log(self.y / self.spec.y0)

    ###

    @property
    def _dy_log(self) -> FloatVector:
        return self._dy / self._y

    @formatted_property
    def dy_log(self) -> FloatVector:
        return self.dy / self.y

###

@pydantic.dataclasses.dataclass
class MaskedCoords(BaseMaskedCoords):
    """
    This class contains arrays of a masked `SpecData` instance.
    """
    _original: ClassVar[Literal[False]] = False
    _suffix: ClassVar[str] = ""

    @classmethod
    def _format(cls, arr: Array | None) -> Array | None:
        return arr    


@pydantic.dataclasses.dataclass
class ReadOnlyMaskedCoords(BaseMaskedCoords):
    """
    This class contains read-only arrays of a masked `SpecData` instance.
    """
    _original: ClassVar[Literal[False]] = False
    _suffix: ClassVar[str] = ""

    @staticmethod
    def _format(arr: Array | None) -> Array | None:
        if arr is None:
            return None
        _arr = arr.copy(order="C")
        _arr.setflags(write=False)
        return _arr


@pydantic.dataclasses.dataclass
class ContiguousMaskedCoords(BaseMaskedCoords):
    """
    This class contains contiguous, read-only arrays of a masked `SpecData`
    instance.
    """
    _original: ClassVar[Literal[False]] = False
    _suffix: ClassVar[str] = ""

    @staticmethod
    def _format(arr: Array | None) -> Array | None:
        if arr is None:
            return None
        _arr = arr.copy(order="C")
        return _arr

###

@pydantic.dataclasses.dataclass
class OriginalMaskedCoords(MaskedCoords):
    """
    This class contains read-only arrays of a masked `SpecData` instance.
    """
    _original: ClassVar[Literal[True]] = True
    _suffix: ClassVar[Literal["_original"]] = "_original"


@pydantic.dataclasses.dataclass
class OriginalReadOnlyMaskedCoords(ReadOnlyMaskedCoords):
    """
    This class contains read-only arrays of a masked `SpecData` instance.
    """
    _original: ClassVar[Literal[True]] = True
    _suffix: ClassVar[Literal["_original"]] = "_original"


@pydantic.dataclasses.dataclass
class OriginalContiguousMaskedCoords(ContiguousMaskedCoords):
    """
    This class contains contiguous, read-only arrays of a masked `SpecData`
    instance.
    """
    _original: ClassVar[Literal[True]] = True
    _suffix: ClassVar[Literal["_original"]] = "_original"
