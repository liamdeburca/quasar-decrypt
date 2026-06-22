__all__ = ['_SpecData']

from logging import getLogger
from typing import Self, Callable, Literal, Optional, Union
from dataclasses import field
from pydantic.dataclasses import dataclass
from numpy import invert, zeros_like, isfinite, ones_like, float64, bool_, ascontiguousarray, inf

from quasar_models.continuum import PowerLawModel
from quasar_models.iron import IronModel
from quasar_models.balmer import BalmerModel
from quasar_models.line import GaussianModel
from quasar_models.host import HostGalaxyModel
from quasar_models.utils.prepare_model import PrepareModel

from quasar_utils.setup import Info
from quasar_utils.decorators import validate_call

from quasar_typing.numpy import FloatVector, BoolVector
from quasar_typing.astropy import CompoundModel_
from quasar_typing.bounds import CoordBounds
from quasar_typing.misc import BackgroundFlux

from .utils import create_cached_get_mask, get_log
from .masked_coords import (
    MaskedCoords,
    ReadOnlyMaskedCoords,
    ContiguousMaskedCoords,
)

logger = getLogger(__name__)

@dataclass
class _SpecData:
    """
    Parent class used for inheriting properties and methods. Designed for 
    inputting arrays. 
    """
    _x: FloatVector = field(kw_only=True)
    _y: FloatVector = field(kw_only=True)
    _dy: FloatVector = field(kw_only=True)
    
    _dx: FloatVector = field(kw_only=True)
    
    _y_smooth: FloatVector = field(kw_only=True)
    _y_pl: FloatVector = field(kw_only=True)
    _y_fe: FloatVector = field(kw_only=True)
    _y_ba: FloatVector = field(kw_only=True)
    _y_hg: FloatVector = field(kw_only=True)
    _y_em: FloatVector = field(kw_only=True)

    _rejected_pixels: BoolVector = field(kw_only=True)
    _absorbed_pixels: BoolVector = field(kw_only=True)
    _valid_pixels: BoolVector = field(kw_only=True)
    _log_valid_pixels: BoolVector = field(kw_only=True)
    
    _p_absorbed: FloatVector = field(kw_only=True)

    x0: float = field(kw_only=True)
    y0: float = field(kw_only=True)

    _x_log: FloatVector = field(kw_only=True)
    _y_log: FloatVector = field(kw_only=True)
    _dy_log: FloatVector = field(kw_only=True)

    x_bounds: CoordBounds = field(kw_only=True)
    info: Info = field(kw_only=True)
    get_mask: Callable[[float, float], BoolVector] | None = field(kw_only=True)

    @classmethod
    @validate_call
    def create(
        cls,
        *,
        x: FloatVector,
        y: FloatVector,
        dy: FloatVector,
        dx: FloatVector | None = None,

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
        
        info: Info = None,
        x_bounds: CoordBounds | None = None,
        get_mask: Callable[[float, float], BoolVector] | None = None,
    ) -> Self:
        kwargs = cls.get_kwargs.__wrapped__(
            x=x,
            y=y,
            dy=dy,
            dx=dx,
            x_bounds=x_bounds,
            info=info,
            y_smooth=y_smooth,
            y_pl=y_pl,
            y_fe=y_fe,
            y_ba=y_ba,
            y_hg=y_hg,
            y_em=y_em,
            rejected_pixels=rejected_pixels,
            absorbed_pixels=absorbed_pixels,
            valid_pixels=valid_pixels,
            log_valid_pixels=log_valid_pixels,
            p_absorbed=p_absorbed,
            x0=x0,
            y0=y0,
            x_log=x_log,
            y_log=y_log,
            dy_log=dy_log,
            get_mask=get_mask,
        )
        return cls(**kwargs)

    @classmethod
    @validate_call
    def get_kwargs(
        cls,
        *,
        x: FloatVector,
        y: FloatVector,
        dy: FloatVector,
        dx: FloatVector | None = None,

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
        
        info: Info = None,
        x_bounds: CoordBounds | None = None,
        get_mask: Callable[[float, float], BoolVector] | None = None,
    ) -> dict:
        kwargs = {
            '_x': x,
            '_y': y,
            '_dy': dy,
        }
        kwargs['_dx'] = ascontiguousarray(
            x * info.loading.sigma_res if dx is None else dx
        )

        kwargs['_y_smooth'] = (
            kwargs['_y'].copy(order='C')
            if y_smooth is None else 
            ascontiguousarray(y_smooth)
        )
        kwargs['_y_pl'] = (
            zeros_like(kwargs['_x'], dtype=float64, order='C')
            if y_pl is None else 
            ascontiguousarray(y_pl)
        )
        kwargs['_y_fe'] = (
            zeros_like(kwargs['_x'], dtype=float64, order='C')
            if y_fe is None else 
            ascontiguousarray(y_fe)
        )
        kwargs['_y_ba'] = (
            zeros_like(kwargs['_x'], dtype=float64, order='C')
            if y_ba is None else 
            ascontiguousarray(y_ba)
        )
        kwargs['_y_hg'] = (
            zeros_like(kwargs['_x'], dtype=float64, order='C')
            if y_hg is None else 
            ascontiguousarray(y_hg)
        )
        kwargs['_y_em'] = (
            zeros_like(kwargs['_x'], dtype=float64, order='C')
            if y_em is None else 
            ascontiguousarray(y_em)
        )

        kwargs['_rejected_pixels'] = (
            zeros_like(kwargs['_x'], dtype=bool_, order='C')
            if rejected_pixels is None else 
            ascontiguousarray(rejected_pixels)
        )
        kwargs['_absorbed_pixels'] = (
            zeros_like(kwargs['_x'], dtype=bool_, order='C')
            if absorbed_pixels is None else 
            ascontiguousarray(absorbed_pixels)
        )
        kwargs['_valid_pixels'] = ascontiguousarray(
            isfinite(kwargs['_x']) \
            & isfinite(kwargs['_y']) \
            & isfinite(kwargs['_dy']) \
            & (kwargs['_dy'] > 0)
            if valid_pixels is None else 
            valid_pixels
        )
        kwargs['_log_valid_pixels'] = ascontiguousarray(
            kwargs['_valid_pixels'] & (kwargs['_y'] > 0)
            if log_valid_pixels is None else 
            ascontiguousarray(log_valid_pixels)
        )

        kwargs['_p_absorbed'] = (
            ones_like(kwargs['_x'], dtype=float64, order='C')
            if p_absorbed is None else 
            ascontiguousarray(p_absorbed)
        )

        if x_bounds is None:      
            mask = kwargs['_valid_pixels']      
            n_valid = mask.sum()
            if n_valid < 2:
                msg = "No. of valid pixels is less than 2!"
                logger.critical(msg)
                x_bounds = (0, inf)
            else:
                x = kwargs['_x'][mask]
                x_bounds = (
                    x[ 0] * (1 - info.loading.sigma_res / 2),
                    x[-1] * (1 + info.loading.sigma_res / 2),
                )
                
        kwargs['x_bounds'] = x_bounds

        kwargs['x0'] = x0 or info.continuum.x0
        kwargs['y0'] = y0 or info.continuum.y0

        kwargs['_x_log'] = ascontiguousarray(
            get_log(kwargs['_x'], kwargs['x0'], kwargs['_log_valid_pixels'])
            if x_log is None else 
            ascontiguousarray(x_log)
        )
        kwargs['_y_log'] = ascontiguousarray(
            get_log(kwargs['_y'], kwargs['y0'], kwargs['_log_valid_pixels'])
            if y_log is None else 
            ascontiguousarray(y_log)
        )
        kwargs['_dy_log'] = ascontiguousarray(
            get_log(kwargs['_dy'], kwargs['_y'], kwargs['_log_valid_pixels'])
            if dy_log is None else 
            ascontiguousarray(dy_log)
        )

        kwargs['info'] = info

        kwargs['get_mask'] = (
            create_cached_get_mask(kwargs['_x'], maxsize=1)
            if get_mask is None else
            get_mask
        )

        return kwargs

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

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop('get_mask', None)
        return state

    def __setstate__(self, state: dict) -> None:
        state['get_mask'] = create_cached_get_mask(state['_x'], maxsize=1)
        self.__dict__.update(state)

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
        mode: Literal['r', 'c'] | None = None,
        covered: bool = False,
        without_rejections: bool = False, 
        without_absorption: bool = False,
        valid: bool = False,
        log_valid: bool = False,
        bg_flux: BackgroundFlux | None = None,
    ) -> MaskedCoords:
        mask = self.getMask.__wrapped__(
            self,
            covered = covered,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
            valid = valid,
            log_valid = log_valid,
        )
        match mode:
            case 'r':
                return ReadOnlyMaskedCoords(self, mask, bg_flux=bg_flux)
            case 'c':
                return ContiguousMaskedCoords(self, mask, bg_flux=bg_flux)
            case _:
                 return MaskedCoords(self, mask, bg_flux=bg_flux)
    
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
        model: Union[GaussianModel, CompoundModel_[GaussianModel], None] = None,
    ) -> Self:
        self._y_em[:] = 0
        if model is not None:
            mask = isfinite(self._x)
            self._y_em[mask] = model(self._x[mask])
        return self