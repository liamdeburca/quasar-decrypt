__all__ = ['_SpecData']

from logging import getLogger
from typing import Self, Callable
from dataclasses import dataclass
from numpy import zeros_like, invert, isfinite, ones_like, float64, bool_, ascontiguousarray, inf

from quasar_models.continuum import PowerLawModel
from quasar_models.iron import IronModel
from quasar_models.balmer import BalmerModel
from quasar_models.line import GaussianModel
from quasar_models.host import HostGalaxyModel

from quasar_utils.setup import Info

from pydantic import validate_call
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function

from quasar_typing.numpy import FloatVector, BoolVector, CoordsTuple
from quasar_typing.astropy import CompoundModel_
from quasar_typing.bounds import CoordBounds
from quasar_typing.misc import BackgroundFlux

from .utils import create_cached_get_mask, get_log

logger = getLogger(__name__)

@dataclass
class _SpecData:
    """
    Parent class used for inheriting properties and methods. Designed for 
    inputting arrays. 
    """
    _coords: CoordsTuple
    _x: FloatVector
    _dx: FloatVector
    _y: FloatVector
    _dy: FloatVector
    
    _y_smooth: FloatVector
    _y_pl: FloatVector
    _y_fe: FloatVector
    _y_ba: FloatVector
    _y_hg: FloatVector
    _y_em: FloatVector

    _rejected_pixels: BoolVector
    _absorbed_pixels: BoolVector
    _valid_pixels: BoolVector
    _log_valid_pixels: BoolVector
    
    _p_absorbed: FloatVector

    x0: float
    y0: float

    _x_log: FloatVector
    _y_log: FloatVector
    _dy_log: FloatVector

    x_bounds: CoordBounds
    info: Info
    get_mask: Callable[[float, float], BoolVector] | None

    @validate_call
    def __init__(
        self,
        coords: CoordsTuple,
        *,
        x_bounds: CoordBounds | None = None,
        info: Info = None,
        y_smooth: FloatVector | None = None,
        y_pl: FloatVector | None = None,
        y_fe: FloatVector | None = None,
        y_ba: FloatVector | None = None,
        y_hg: FloatVector | None = None,
        y_em: FloatVector | None = None,
        rejected_pixels: BoolVector | None = None,
        absorbed_pixels: BoolVector | None = None,
        valid_pixels: BoolVector | None = None,
        log_valid_pixels: BoolVector | None = None,
        p_absorbed: FloatVector | None = None,
        x0: float | None = None,
        y0: float | None = None,
        x_log: FloatVector | None = None,
        y_log: FloatVector | None = None,
        dy_log: FloatVector | None = None,
        get_mask: Callable[[float, float], BoolVector] = None,
    ):
        """
        ** PYDANTIC VALIDATED METHOD **

        This method assigns the basic properties of _SpecData-like classes, such
        as the coordinate arrays, the Info object, and the 'get_mask' method.

        Input arrays are converted to C-contiguous format if they aren't 
        already.
        """
        self.info = info
        
        # Ensures coordinate arrays are C-contiguous
        self._coords = tuple(ascontiguousarray(arr) for arr in coords)
        self._x = self._coords[0]
        self._dx = self._coords[0] * info.loading['sigma_res']
        self._y = self._coords[1]
        self._dy = self._coords[2]

        self._y_smooth = (
            self._y.copy(order='C')
            if y_smooth is None else 
            ascontiguousarray(y_smooth)
        )
        self._y_pl = (
            zeros_like(self._x, dtype=float64, order='C')
            if y_pl is None else 
            ascontiguousarray(y_pl)
        )
        self._y_fe = (
            zeros_like(self._x, dtype=float64, order='C')
            if y_fe is None else 
            ascontiguousarray(y_fe)
        )
        self._y_ba = (
            zeros_like(self._x, dtype=float64, order='C')
            if y_ba is None else 
            ascontiguousarray(y_ba)
        )
        self._y_hg = (
            zeros_like(self._x, dtype=float64, order='C')
            if y_hg is None else 
            ascontiguousarray(y_hg)
        )
        self._y_em = (
            zeros_like(self._x, dtype=float64, order='C')
            if y_em is None else 
            ascontiguousarray(y_em)
        )

        self._rejected_pixels = (
            zeros_like(self._x, dtype=bool_, order='C')
            if rejected_pixels is None else 
            ascontiguousarray(rejected_pixels)
        )
        self._absorbed_pixels = (
            zeros_like(self._x, dtype=bool_, order='C')
            if absorbed_pixels is None else 
            ascontiguousarray(absorbed_pixels)
        )
        self._valid_pixels = ascontiguousarray(
            isfinite(coords).all(axis=0) & (self._dy > 0)
            if valid_pixels is None else 
            valid_pixels
        )
        self._log_valid_pixels = ascontiguousarray(
            self._valid_pixels & (self._y > 0)
            if log_valid_pixels is None else 
            log_valid_pixels
        )
        
        self._p_absorbed = (
            ones_like(self._x, dtype=float64, order='C')
            if p_absorbed is None else 
            ascontiguousarray(p_absorbed)
        )

        if x_bounds is not None:
            self.x_bounds = x_bounds
        else:
            x = self._x[self._valid_pixels]
            dx = self._dx[self._valid_pixels]

            if x.size < 2:
                msg = "No. of valid pixels is less than 2! This is likely " \
                    "due to most pixels being invalid. Valid fraction: {}/{}" \
                    .format(x.size, self._x.size)
                logger.critical(msg)
                self.x_bounds = (0, inf)
            else:
                self.x_bounds = (x[0] - dx[0] / 2, x[-1] + dx[-1] / 2)

        self.x0 = x0 or info.continuum['x0']
        self.y0 = y0 or info.continuum['y0']

        self._x_log = ascontiguousarray(
            get_log(self._x, self.x0, self._log_valid_pixels)
            if x_log is None else 
            x_log
        )
        self._y_log = ascontiguousarray(
            get_log(self._y, self.y0, self._log_valid_pixels)
            if y_log is None else 
            y_log
        )
        self._dy_log = ascontiguousarray(
            get_log(self._dy, self._y, self._log_valid_pixels)
            if dy_log is None else 
            dy_log
        )

        self.get_mask = (
            create_cached_get_mask(self._x, maxsize=1)
            if get_mask is None else
            get_mask
        )

        if get_mask is None:
            self.get_mask = create_cached_get_mask(self._x, maxsize=1)
        else:
            self.get_mask = get_mask

        self.__post_init__()

    def __post_init__(self):
        """
        This does nothing.
        """
        pass

    def __str__(self, simple: bool = False) -> str:

        s = "'{}' class [{:.1f} <-> {:.1f}] <{}>".format(
            self.__class__.__name__, 
            *self.x_bounds,
            hex(id(self)),
        )
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
    
    @classmethod
    def _validate(cls, value: object) -> Self:
        if not isinstance(value, cls):
            msg = f"Expected a {cls.__name__} instance, \
                got {type(value).__name__}"
            raise PydanticCustomError('validation_error', msg)
        return value

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return no_info_plain_validator_function(cls._validate)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('get_mask', None)
        return state

    def __setstate__(self, state):
        state['get_mask'] = create_cached_get_mask(state['_x'], maxsize=1)
        self.__dict__.update(state)

    @property
    def mask(self) -> BoolVector:
        return self.get_mask(*self.x_bounds)
    
    @property
    def size(self) -> int:
        return self.mask.astype(int).sum()
    
    @property
    def coords(self) -> CoordsTuple:
        return self.x, self.y, self.dy
    
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
    
    @property
    def absorbed_pixels(self) -> BoolVector:
        return self._absorbed_pixels[self.mask]
    
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
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        mask = self.mask.copy(order='C')

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
        covered: bool = False,
        log: bool = False,
        without_rejections: bool = False, 
        without_absorption: bool = False,
        valid: bool = False,
        log_valid: bool = False,
        bg_flux: BackgroundFlux | None = None,
    ) -> CoordsTuple:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        x = self._x_log if log else self._x
        dy = self._dy_log if log else self._dy

        y = self._y.copy()
        if bg_flux is not None:
            for bg in bg_flux:
                y -= getattr(self, f"_y_{bg}")

        if log:
            y = get_log(y, self.y0, self._log_valid_pixels)

        mask = self.getMask.__wrapped__(
            self,
            covered = covered,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
            valid = valid,
            log_valid = log_valid,
        )
        return x[mask], y[mask], dy[mask]
    
    def resetRejections(self) -> None:
        s = self.__str__(simple=True).removesuffix('.')
        n = self.size
        r = self.n_rej
        logger.debug(
            f"Resetting rejection mask for {s}: {r}/{n} -> {0}/{n}."
        )
        self._rejected_pixels[:] = False

    def resetAbsorption(self) -> None:
        s = self.__str__(simple=True).removesuffix('.')
        n = self.size
        a = self.n_abs
        logger.debug(
            f"Resetting absorption mask for {s}: {a}/{n} -> {0}/{n}."
        )
        self._absorbed_pixels[:] = False

    @validate_call
    def applyRejections(
        self, 
        rejected_pixels: BoolVector,
        enforce: bool = True,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        s = self.__str__(simple=True).removesuffix('.')
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
                self._rejected_pixels = rejected_pixels
            else:
                self._rejected_pixels |= rejected_pixels

        elif len(rejected_pixels) == len(self.x):
            if enforce:
                self._rejected_pixels[self.mask] = rejected_pixels
            else:
                self._rejected_pixels[self.mask] |= rejected_pixels

        else:
            logger.error(
                "Mask size should be '{}' or '{}', but is '{}'!" \
                "Doing nothing" \
                .format(self.size, len(self._x), len(rejected_pixels))
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
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        s = self.__str__(simple=True).removesuffix('.')
        msg = f"Applying absorption mask to {s}: "
        n = self.size
        
        a1 = self.n_abs
        if len(absorbed_pixels) == len(self._x):
            if enforce:
                self._absorbed_pixels = absorbed_pixels
            else:
                self._absorbed_pixels |= absorbed_pixels

            if y_smooth is not None:
                self._y_smooth = y_smooth

        elif len(absorbed_pixels) == self.size:
            if enforce:
                self._absorbed_pixels[self.mask] = absorbed_pixels
            else:
                self._absorbed_pixels[self.mask] |= absorbed_pixels

            if y_smooth is not None:
                self._y_smooth[self.mask] = y_smooth
        
        else:
            logger.error(
                "Mask size should be '{}' or '{}', but is '{}'!" \
                    "Doing nothing..." \
                    .format(self.size, len(self._x), len(absorbed_pixels))
            )
            return

        a2 = self.n_abs
        logger.debug(msg + f"{a1}/{n} -> {a2}/{n} (abs.).")

        return self

    @validate_call
    def updateContinuumEmission(
        self,
        model: PowerLawModel | None = None,
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
        model: IronModel | CompoundModel_[IronModel] | None = None,
    ) -> Self:
        self._y_fe[:] = 0
        if model is not None:
            mask = isfinite(self._x)
            self._y_fe[mask] = model(self._x[mask])
        return self

    @validate_call
    def updateBalmerEmission(
        self,
        model: BalmerModel | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self._y_ba[:] = 0
        if model is not None:
            mask = isfinite(self._x)
            self._y_ba[mask] = model(self._x[mask])
        return self

    @validate_call
    def updateHostGalaxyEmission(
        self,
        model: HostGalaxyModel | None = None,
    ) -> Self:
        self._y_hg[:] = 0
        if model is not None:
            mask = isfinite(self._x)
            self._y_hg[mask] = model(self._x[mask])
        return self

    @validate_call
    def updateLinesEmission(
        self,
        model: GaussianModel | CompoundModel_[GaussianModel] | None = None,
    ) -> Self:
        self._y_em[:] = 0
        if model is not None:
            mask = isfinite(self._x)
            self._y_em[mask] = model(self._x[mask])
        return self