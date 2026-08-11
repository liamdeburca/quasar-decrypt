__all__ = ["_BaseSpec"]

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import field
from logging import getLogger
from typing import Any, Literal, Optional, Self, Union

from numpy import invert, isfinite
from pydantic.dataclasses import dataclass
from quasar_models import (
    BalmerModel,
    GaussianModel,
    HostGalaxyModel,
    IronModel,
    PowerLawModel,
)
from quasar_models.modeling import PrepareModel
from quasar_typing.astropy import CompoundModel_
from quasar_typing.bounds import CoordBounds
from quasar_typing.misc import BackgroundFlux
from quasar_typing.numpy import BoolVector, FloatVector
from quasar_utils.decorators import validate_call

from .masked_coords import (
    ContiguousMaskedCoords,
    MaskedCoords,
    ReadOnlyMaskedCoords,
)

logger = getLogger(__name__)


@dataclass
class _BaseSpec:
    """
    Base class for '_Spec', '_SpecWindow', and '_SpecWindowList' classes.
    
    Stores very basic attributes and essential methods. 
    """
    x_bounds: CoordBounds = field(kw_only=True)
    get_mask: Callable[[float, float], BoolVector] | None = field(kw_only=True)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("get_mask", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __str__(self, simple: bool = False) -> str:
        s = f"<{self.__class__.__name__} object at {hex(id(self))}> "
        s += f"x_bounds={self.x_bounds}"
        if not simple:
            n = self.size
            s += " w/ "
            s += f"{self.n_rej}/{n} (rej.) "
            s += f"{self.n_abs}/{n} (abs.) "
            s += f"{self.n_val}/{n} (val.) "
            s += f"{self.n_logval}/{n} (log-val.)"
        return s

    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs) -> Self:
        pass

    @property
    def mask(self) -> BoolVector:
        return self.get_mask(*self.x_bounds)

    @property
    def size(self) -> int:
        return self.mask.astype(int).sum()

    @property
    def x(self) -> FloatVector:
        return self._x[self.mask]

    @property
    def y(self) -> FloatVector:
        return self._y[self.mask]

    @property
    def dy(self) -> FloatVector:
        return self._dy[self.mask]

    @property
    def dx(self) -> FloatVector:
        return self._dx[self.mask]

    @property
    def x_log(self) -> FloatVector:
        return self._x_log[self.mask]

    @property
    def y_log(self) -> FloatVector:
        return self._y_log[self.mask]

    @property
    def dy_log(self) -> FloatVector:
        return self._dy_log[self.mask]

    @property
    def y_smooth(self) -> FloatVector:
        return self._y_smooth[self.mask]

    @property
    def y_pl(self) -> FloatVector:
        return self._y_pl[self.mask]

    @property
    def y_fe(self) -> FloatVector:
        return self._y_fe[self.mask]

    @property
    def y_ba(self) -> FloatVector:
        return self._y_ba[self.mask]

    @property
    def y_hg(self) -> FloatVector:
        return self._y_hg[self.mask]

    @property
    def y_em(self) -> FloatVector:
        return self._y_em[self.mask]

    @property
    def rejected_pixels(self) -> BoolVector:
        return self._rejected_pixels[self.mask]

    @rejected_pixels.setter
    def rejected_pixels(self, value: BoolVector) -> None:
        self.applyRejections.__wrapped__(self, value, enforce=True)

    @rejected_pixels.deleter
    def rejected_pixels(self) -> None:
        self.resetRejections()

    @property
    def absorbed_pixels(self) -> BoolVector:
        return self._absorbed_pixels[self.mask]

    @absorbed_pixels.setter
    def absorbed_pixels(self, value: BoolVector) -> None:
        self.applyAbsorption.__wrapped__(self, value, enforce=True)

    @absorbed_pixels.deleter
    def absorbed_pixels(self) -> None:
        self.resetAbsorption()

    @property
    def valid_pixels(self) -> BoolVector:
        return self._valid_pixels[self.mask]

    @property
    def log_valid_pixels(self) -> BoolVector:
        return self._log_valid_pixels[self.mask]

    @property
    def p_absorbed(self) -> FloatVector:
        return self._p_absorbed[self.mask]

    @property
    def n_rej(self) -> int:
        return self.rejected_pixels.sum()

    @property
    def n_abs(self) -> int:
        return self.absorbed_pixels.sum()

    @property
    def n_val(self) -> int:
        return self.valid_pixels.sum()

    @property
    def n_logval(self) -> int:
        return self.log_valid_pixels.sum()

    @validate_call
    def getMask(
        self,
        *,
        covered: bool = True,
        without_rejections: bool = False,
        without_absorption: bool = False,
        valid: bool = False,
        log_valid: bool = False,
    ) -> BoolVector:
        mask = self.mask.copy(order="C")

        if not covered:
            mask[:] = True
        if without_rejections:
            mask &= invert(self._rejected_pixels)
        if without_absorption:
            mask &= invert(self._absorbed_pixels)

        if log_valid:
            mask &= self._log_valid_pixels
        elif valid:
            mask &= self._valid_pixels

        return mask

    @validate_call
    def getMaskedCoords(
        self,
        *,
        mode: Literal["r", "c"] | None = None,
        covered: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        valid: bool = False,
        log_valid: bool = False,
        bg_flux: BackgroundFlux | None = None,
    ) -> MaskedCoords:
        mask = self.getMask.__wrapped__(
            self,
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=valid,
            log_valid=log_valid,
        )
        match mode:
            case "r":
                return ReadOnlyMaskedCoords(self, mask, bg_flux=bg_flux)
            case "c":
                return ContiguousMaskedCoords(self, mask, bg_flux=bg_flux)
            case _:
                return MaskedCoords(self, mask, bg_flux=bg_flux)

    def resetRejections(self) -> None:
        s = self.__str__(simple=True).removesuffix(".")
        n = self.size
        r = self.n_rej
        logger.debug(f"Resetting rejection mask for {s}: {r}/{n} -> {0}/{n}.")
        self._rejected_pixels[:] = False

    def resetAbsorption(self) -> None:
        s = self.__str__(simple=True).removesuffix(".")
        n = self.size
        a = self.n_abs
        logger.debug(f"Resetting absorption mask for {s}: {a}/{n} -> {0}/{n}.")
        self._absorbed_pixels[:] = False

    @validate_call
    def applyRejections(
        self,
        rejected_pixels: BoolVector,
        enforce: bool = True,
    ) -> Self:
        s = self.__str__(simple=True).removesuffix(".")
        msg = f"Applying rejection mask to {s}: "
        m = len(rejected_pixels)
        _n = len(self._x)
        n = self.size

        if m not in [n, _n]:
            logger.error(
                msg + f"Improper mask size, {m}. Should be either {n} or {_n}."
            )
            return

        r1 = self.n_rej
        if len(rejected_pixels) == len(self._x):
            if enforce:
                self._rejected_pixels[:] = rejected_pixels
            else:
                self._rejected_pixels |= rejected_pixels

        elif len(rejected_pixels) == len(self.x):
            if enforce:
                self._rejected_pixels[self.mask] = rejected_pixels
            else:
                self._rejected_pixels[self.mask] |= rejected_pixels

        else:
            logger.error(
                f"Mask size should be '{self.size}' or '{len(self._x)}', but is '{len(rejected_pixels)}'!"
                "Doing nothing"
            )
            return

        r2 = self.n_rej
        logger.debug(msg + f"{r1}/{n} -> {r2}/{n} (rej.).")

        return self

    @validate_call
    def applyAbsorption(
        self,
        absorbed_pixels: BoolVector,
        y_smooth: FloatVector | None = None,
        enforce: bool = True,
    ) -> Self:
        s = self.__str__(simple=True).removesuffix(".")
        msg = f"Applying absorption mask to {s}: "
        n = self.size

        a1 = self.n_abs
        if len(absorbed_pixels) == len(self._x):
            if enforce:
                self._absorbed_pixels[:] = absorbed_pixels
            else:
                self._absorbed_pixels |= absorbed_pixels

            if y_smooth is not None:
                self._y_smooth[:] = y_smooth

        elif len(absorbed_pixels) == self.size:
            if enforce:
                self._absorbed_pixels[self.mask] = absorbed_pixels
            else:
                self._absorbed_pixels[self.mask] |= absorbed_pixels

            if y_smooth is not None:
                self._y_smooth[self.mask] = y_smooth

        else:
            logger.error(
                f"Mask size should be '{self.size}' or '{len(self._x)}', but is '{len(absorbed_pixels)}'!"
                "Doing nothing..."
            )
            return

        a2 = self.n_abs
        logger.debug(msg + f"{a1}/{n} -> {a2}/{n} (abs.).")

        return self

    @validate_call
    def updateContinuumEmission(
        self,
        model: Optional[PowerLawModel] = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self._y_pl[:] = 0
        if model is not None:
            mask = isfinite(self._x)
            self._y_pl[mask] = model(self._x[mask])
        return self

    @validate_call
    def updateIronEmission(
        self,
        model: Union[IronModel, CompoundModel_[IronModel], None] = None,
    ) -> Self:
        self._y_fe[:] = 0
        if model is not None:
            mask = isfinite(self._x)
            x = self._x[mask]
            with PrepareModel(x=x, model=model):
                self._y_fe[mask] = model(x)
        return self

    @validate_call
    def updateBalmerEmission(
        self,
        model: Optional[BalmerModel] = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self._y_ba[:] = 0
        if model is not None:
            mask = isfinite(self._x)
            x = self._x[mask]
            with PrepareModel(x=x, model=model):
                self._y_ba[mask] = model(x)
        return self

    @validate_call
    def updateHostGalaxyEmission(
        self,
        model: Optional[HostGalaxyModel] = None,
    ) -> Self:
        self._y_hg[:] = 0
        if model is not None:
            mask = isfinite(self._x)
            x = self._x[mask]
            with PrepareModel(x=x, model=model):
                self._y_hg[mask] = model(x)
        return self

    @validate_call
    def updateLinesEmission(
        self,
        model: Union[
            GaussianModel, CompoundModel_[GaussianModel], None
        ] = None,
    ) -> Self:
        self._y_em[:] = 0
        if model is not None:
            mask = isfinite(self._x)
            self._y_em[mask] = model(self._x[mask])
        return self
