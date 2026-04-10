__all__ = ['_SpecData']

from logging import getLogger
from typing import Self, Optional, Union, Callable
from dataclasses import dataclass
from numpy import (
    zeros_like, invert, full_like, nan, isfinite, ones_like, 
    where, quantile, maximum, minimum, arange, float64, bool_
)

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from quasar_models.continuum import PowerLawModel
from quasar_models.iron import IronModel
from quasar_models.balmer import BalmerModel
from quasar_models.line import GaussianModel
from quasar_models.host import HostGalaxyModel

from quasar_utils.setup import Info
from quasar_utils.wrappers import apply_info_to_method

from pydantic import validate_call
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function

from quasar_typing.numpy import FloatVector, BoolVector
from quasar_typing.astropy import CompoundModel_
from quasar_typing.bounds import CoordBounds
from quasar_typing.misc.literals import BGFlux
from quasar_typing.misc.coords_tuple import CoordsTuple

from .utils import create_cached_get_mask, get_log

logger = getLogger(__name__)
logger.disabled = not getLogger().hasHandlers()

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
    get_mask: Callable[[float, float], BoolVector]

    @validate_call(validate_return=False)
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
        get_mask: Callable[[float, float], BoolVector] | None = None,
    ):
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self.info = info
        self.get_mask = create_cached_get_mask(coords[0], maxsize=1)
        
        self._coords = coords
        self._x = coords[0]
        self._dx = coords[0] * info.loading['sigma_res']
        self._y = coords[1]
        self._dy = coords[2]

        self._y_smooth = (
            self._y.copy()
            if y_smooth is None else 
            y_smooth
        )
        self._y_pl = (
            zeros_like(self._x, dtype=float64)
            if y_pl is None else 
            y_pl
        )
        self._y_fe = (
            zeros_like(self._x, dtype=float64)
            if y_fe is None else 
            y_fe
        )
        self._y_ba = (
            zeros_like(self._x, dtype=float64)
            if y_ba is None else 
            y_ba
        )
        self._y_hg = (
            zeros_like(self._x, dtype=float64)
            if y_hg is None else 
            y_hg
        )
        self._y_em = (
            zeros_like(self._x, dtype=float64)
            if y_em is None else 
            y_em
        )

        self._rejected_pixels = (
            zeros_like(self._x, dtype=bool_)
            if rejected_pixels is None else 
            rejected_pixels
        )
        self._absorbed_pixels = (
            zeros_like(self._x, dtype=bool_)
            if absorbed_pixels is None else 
            absorbed_pixels
        )
        self._valid_pixels = (
            isfinite(coords).all(axis=0) & (self._dy > 0)
            if valid_pixels is None else 
            valid_pixels
        )
        self._log_valid_pixels = (
            self._valid_pixels & (self._y > 0)
            if log_valid_pixels is None else 
            log_valid_pixels
        )
        
        self._p_absorbed = (
            ones_like(self._x, dtype=float64)
            if p_absorbed is None else 
            p_absorbed
        )

        if x_bounds is None:
            x = self._x[self._valid_pixels]
            dx = self._dx[self._valid_pixels]
            self.x_bounds = (x[0] - dx[0] / 2, x[-1] + dx[-1] / 2)
        else:
            self.x_bounds = x_bounds

        self.x0 = x0 or info.continuum['x0']
        self.y0 = y0 or info.continuum['y0']

        self._x_log = (
            get_log(self._x, self.x0, self._log_valid_pixels)
            if x_log is None else 
            x_log
        )
        self._y_log = (
            get_log(self._y, self.y0, self._log_valid_pixels)
            if y_log is None else 
            y_log
        )
        self._dy_log = (
            get_log(self._dy, self._y, self._log_valid_pixels)
            if dy_log is None else 
            dy_log
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
    
    @validate_call(validate_return=False)
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
        mask = self.mask.copy()

        if not covered:
            mask[:] = True
        if without_rejections:
            mask &= invert(self._rejected_pixels)
        if without_absorption:
            mask &= invert(self._absorbed_pixels)

        if log_valid:
            mask &=  self._log_valid_pixels
        elif valid:
            mask &=  self._valid_pixels

        return mask
    
    @validate_call(validate_return=False)
    def getMaskedCoords(
        self, 
        *,
        covered: bool = False,
        log: bool = False,
        without_rejections: bool = False, 
        without_absorption: bool = False,
        valid: bool = False,
        log_valid: bool = False,
        bg_flux: BGFlux | None = None,
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

    @validate_call(validate_return=False)
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

    @validate_call(validate_return=False)
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

    @validate_call(validate_return=False)
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

    @validate_call(validate_return=False)
    def updateIronEmission(
        self,
        model: Union[IronModel, CompoundModel_[IronModel], None] = None,
    ) -> Self:
        self._y_fe[:] = 0
        if model is not None:
            mask = isfinite(self._x)
            self._y_fe[mask] = model(self._x[mask])
        return self

    @validate_call(validate_return=False)
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
            self._y_ba[mask] = model(self._x[mask])
        return self

    @validate_call(validate_return=False)
    def updateHostGalaxyEmission(
        self,
        model: Optional[HostGalaxyModel] = None,
    ) -> Self:
        self._y_hg[:] = 0
        if model is not None:
            mask = isfinite(self._x)
            self._y_hg[mask] = model(self._x[mask])
        return self

    @validate_call(validate_return=False)
    def updateLinesEmission(
        self,
        model: Union[GaussianModel, CompoundModel_[GaussianModel], None] = None,
    ) -> Self:
        self._y_em[:] = 0
        if model is not None:
            mask = isfinite(self._x)
            self._y_em[mask] = model(self._x[mask])
        return self

    ### Plotting routines

    def _quickplot(
        self,
        fig: tuple[Figure, Axes] | None = None,
        *,
        figsize: tuple[float, float] = (8, 6),
        dpi: int = 300,
        title: str | None = None,

        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,

        xlabel: str | None = None,
        ylabel: str | None = None,

        x_major: float = 500.0,
        x_minor: float = 20.0,
        y_major: float = 50.0,
        y_minor: float = 5.0,

        pl_color: str = 'dodgerblue',
        fe_color: str = 'firebrick',
        ba_color: str = 'orchid',
        hg_color: str = 'forestgreen',
        em_color: str = 'gold',
        
        sm_color: str = 'darkorange',
        ab_color: str = 'darkviolet',

        logx: bool = True,
        logy: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Basic plotting routine for '_SpecData' and inheriting classes. 
        """
        from matplotlib.pyplot import subplots
        from matplotlib.ticker import MultipleLocator, ScalarFormatter, \
            NullFormatter
        from astropy.units.format import LatexInline

        if fig is None:
            fig, ax = subplots(figsize=figsize, dpi=dpi)
        else:
            fig, ax = fig
    
        title = title or self.__str__(simple=True).removesuffix('.')
        ax.set_title(title, loc='left')

        ax.step(self._x, self._y, where='mid', color='k', zorder=1)
        ax.fill_between(
            self._x, 
            self._y - 2*self._dy, self._y + 2*self._dy,
            step='mid', color='grey', zorder=0,
        )
        ax.step(self._x, self._y_smooth, where='mid', color=sm_color, zorder=2)
        ax.step(
            where(self._absorbed_pixels, self._x, nan),
            where(self._absorbed_pixels, self._y, nan),
            where='mid', color=ab_color, zorder=1,
        )

        if not (self._y_pl == 0).all(): 
            ax.plot(
                self._x, 
                self._y_pl, 
                color=pl_color, zorder=6,
            )
        if not (self._y_fe == 0).all(): 
            ax.plot(
                self._x, 
                self._y_pl + self._y_fe, 
                color=fe_color, zorder=5,
            )
        if not (self._y_ba == 0).all(): 
            ax.plot(
                self._x, 
                self._y_pl + self._y_fe + self._y_ba, 
                color=ba_color, zorder=4,
            )
        if not (self._y_hg == 0).all():
            ax.plot(
                self._x,
                self._y_pl + self._y_fe + self._y_ba + self._y_hg,
                color=hg_color, zorder=3,
            )
        if not (self._y_em == 0).all(): 
            ax.plot(
                self._x, 
                self._y_pl + self._y_fe + self._y_ba + self._y_em, 
                color=em_color, zorder=2,
            )

        if ylim is None: 
            ylim = (
                quantile(self._y[self._valid_pixels], 0.01),
            )
        ax.set_ylim(*ylim)

        if xlim is None: 
            xlim = self.x_bounds
        ax.set_xlim(*xlim)

        if xlabel is None: 
            xlabel = "Rest-frame wavelength ({})".format(
                LatexInline.to_string(self.info.units['wavelength_unit']),
            )
        ax.set_xlabel(xlabel, loc='right')

        if ylabel is None: 
            ylabel = "Flux density ({})".format(
                LatexInline.to_string(self.info.units.getFluxUnit()),
            )
        ax.set_ylabel(ylabel, loc='top')

        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')

        ax.xaxis.set_major_locator(MultipleLocator(x_major))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        
        ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
        ax.xaxis.set_minor_formatter(NullFormatter())

        ax.yaxis.set_major_locator(MultipleLocator(y_major))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor))
        ax.yaxis.set_minor_formatter(NullFormatter())

        return fig, ax
    
    @apply_info_to_method('absorption', specific_kwargs={'z_crit', 'p_crit'})
    def _absorptionplot(
        self,
        fig: tuple[Figure, Axes] | None = None,
        *,
        figsize: tuple[float, float] = (8, 8),
        dpi: int = 300,
        title: str | None = None,
        height_ratio: float = 3.0,

        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        zlim: tuple[float, float] = (-5.0, 5.0),
        plim: tuple[float, float] = (1e-5, 1e0),

        xlabel: str | None = None,
        ylabel: str | None = None,

        x_major: float = 500.0,
        x_minor: float = 20.0,
        y_major: float = 50.0,
        y_minor: float = 5.0,
        z_major: float = 2.0,
        z_minor: float = 1.0,
        p_major: float = 10.0,
        p_minor: float = 1.0,

        pl_color: str = 'dodgerblue',
        fe_color: str = 'firebrick',
        ba_color: str = 'orchid',
        hg_color: str = 'forestgreen',
        em_color: str = 'gold',
        sm_color: str = 'darkorange',
        ab_color: str = 'darkviolet',

        cr_color: str = 'crimson',

        z_crit: float | None = None,
        p_crit: float | None = None,
    ) -> tuple[Figure, Axes]:
        """
        Basic absorption-removal plotting routine for '_SpecData' and inheriting 
        classes. 
        """
        from matplotlib.pyplot import subplots
        from matplotlib.ticker import MultipleLocator, ScalarFormatter, \
            NullFormatter, LogLocator, LogFormatterSciNotation

        if fig is None:
            fig, axes = subplots(
                3, 1,
                figsize = figsize, 
                dpi = dpi,
                sharex = True,
                height_ratios = [height_ratio, 1, 1],
            )
        else:
            fig, axes = fig

        _ = _SpecData._quickplot(
            self,
            (fig, axes[0]),
            title = title or self.__str__(simple=True).removesuffix('.'),
            xlim = xlim,
            ylim = ylim,
            xlabel = xlabel,
            ylabel = ylabel,
            x_major = x_major,
            x_minor = x_minor,
            y_major = y_major,
            y_minor = y_minor,
            pl_color = pl_color,
            fe_color = fe_color,
            ba_color = ba_color,
            hg_color = hg_color,
            em_color = em_color,
            ab_color = ab_color,
            sm_color = sm_color,
        )
        xlabel = axes[0].get_xlabel()
        axes[0].set_xlabel(None)

        ###

        v = self._valid_pixels
        _z = full_like(self._x, fill_value=nan)
        _z[v] = (self._y[v] - self._y_smooth[v]) / self._dy[v]
        _z = maximum(minimum(_z, zlim[1]), zlim[0])

        ax = axes[1]
        ax.step(self._x, _z, where='mid', color='k', zorder=0)
        ax.hlines(z_crit, *ax.set_xlim(), color=cr_color, zorder=1, ls='dotted')
        
        ax.set_ylim(*zlim)
        ax.set_ylabel('Residual')

        ###

        _p = maximum(minimum(self._p_absorbed, plim[1]), plim[0])

        ax = axes[2]
        ax.step(self._x, _p, where='mid', color='k', zorder=0)
        ax.hlines(p_crit, *ax.set_xlim(), color=cr_color, zorder=1, ls='dotted')

        ax.set_yscale('log')
        ax.set_ylim(*plim)
        ax.set_xlabel(xlabel, loc='right')
        ax.set_ylabel('Significance')

        ###

        axes[1].yaxis.set_major_locator(MultipleLocator(z_major))
        axes[1].yaxis.set_major_formatter(ScalarFormatter())

        axes[1].yaxis.set_minor_locator(MultipleLocator(z_minor))
        axes[1].yaxis.set_minor_formatter(NullFormatter())

        axes[2].yaxis.set_major_locator(LogLocator(base=p_major))
        axes[2].yaxis.set_major_formatter(LogFormatterSciNotation(base=p_major))

        axes[2].yaxis.set_minor_locator(
            LogLocator(base=p_major, subs=arange(p_minor, 1, p_minor), numticks=100)
        )
        axes[2].yaxis.set_minor_formatter(NullFormatter())

        return fig, axes
