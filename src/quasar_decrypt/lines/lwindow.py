__all__ = ['LWindow']

from logging import getLogger
from typing import Self, Iterable, ClassVar, Literal
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy import (
    inf, array, empty, invert, interp, exp, linspace, convolve, argmax, 
    isfinite, diff, unique
)
from scipy.ndimage import binary_dilation

from ..utils import _SpecData, SpecData, get_log
from ..utils.general import stopwatch, get_bounds_indices
from .utils import common_middle

from ..ml import Under, Over

from pydantic import validate_call
from pydantic_core import ValidationError

from quasar_errors.bootstrapping import BaseBootstrapper
from quasar_errors.error_result import ErrorResult
from quasar_errors.spectrum_utils.format_bootstrapping_kwargs_for_lwindow \
    import format_bootstrapping_kwargs_for_lwindow

from quasar_typing.numpy import BoolVector, FloatVector, RandomState_
from quasar_typing.bounds import AstropyBounds
from quasar_typing.pathlib import AbsoluteDirPath, RelativeFilePath
from quasar_typing.astropy import (
    FitInfo, FitterInstance, QTable_, CompoundModel_,
)
from quasar_typing.misc import (
    Scale, Variant, BootstrapType, VaryLines, FWHMStrategy, BackgroundFlux, 
    ModelTypes, OutLines, OutMeasures,
)
from quasar_typing.misc.pool import Pool_

from quasar_utils.decorators import validated_apply_info_to_method
from quasar_utils.continuum_fit_result import ContinuumFitResult

from quasar_models import PowerLawModel, GaussianModel, IronModel, BalmerModel
from quasar_models.line import _VProfileCopy, VProfileCopyDict
from quasar_models.utils.astropy import apply_bounds, order_submodels

from quasar_plotting import quickplot, absorptionplot, fitplot
from quasar_plotting.utils import get_coords
from quasar_plotting.colors import DEFAULT_COLORS

from quasar_errors.model_samples import GaussianSampleList

logger = getLogger(__name__)

type LineModel = GaussianModel | CompoundModel_[GaussianModel]

@dataclass(init=False)
class LWindow(SpecData):
    names: set[str] = field(default_factory=set, init=False)
    
    lines: dict[str, float] = field(default_factory=dict, init=False)
    n_maxs: dict[str, int] = field(default_factory=dict, init=False)
    needs_line: dict[str, str] = field(default_factory=dict, init=False)
    
    strength_bounds: dict[str, tuple] = field(default_factory=dict, init=False)
    sigma_v_bounds: dict[str, tuple] = field(default_factory=dict, init=False)
    v_off_bounds: dict[str, tuple] = field(default_factory=dict, init=False)
    
    is_copy_of: dict[str, str] = field(default_factory=dict, init=False)
    scale_init: dict[str, float] = field(default_factory=dict, init=False)
    scale_bounds: dict[str, tuple] = field(default_factory=dict, init=False)
    
    copies_to: dict[str, list[tuple[int, str]]] = field(default_factory=lambda: defaultdict(list), init=False)
    i_bounds: dict[str, tuple[float, float]] = field(default_factory=dict, init=False)
    blacklist: dict[str, bool] = field(default_factory=dict, init=False)
    _blacklist: dict[str, bool] = field(default_factory=dict, init=False)
    
    neighbours: tuple[_SpecData | None, _SpecData | None] = field(default=(None, None), init=False)
    prev_model: LineModel | None = field(default=None, init=False)
    model: LineModel | None = field(default=None, init=False)
    fit: LineModel | None = field(default=None, init=False)
    fit_info: FitInfo | None = field(default=None, init=False)

    fits: dict[int, LineModel] = field(default_factory=dict, init=False)
    fit_infos: dict[int, FitInfo] = field(default_factory=dict, init=False)

    cropped: set[str] = field(default_factory=set, init=False)

    bootstrapper: BaseBootstrapper | None = field(default=None, init=False)
    error_result: ErrorResult | None = field(default=None, init=False)
    
    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({'all', 'em'})

    def __post_init__(self):
        self.names = set()
        
        self.lines = {}
        self.n_maxs = {}
        self.needs_line = {}

        self.strength_bounds = {}
        self.sigma_v_bounds = {}
        self.v_off_bounds = {}

        self.is_copy_of = {}
        self.scale_init = {}
        self.scale_bounds = {}

        self.copies_to = defaultdict(list)
        self.i_bounds = {}
        self.blacklist = {}
        self._blacklist = {}

        self.fits = {}
        self.fit_infos = {}

        self.cropped = set()

    @property
    def sample(self) -> GaussianSampleList | None:
        if (model := self.getModel()) is None:
            return None
        return GaussianSampleList.fromGaussianModels(model)
    
    @validated_apply_info_to_method(subjects=('loading', 'lines', 'nonlinear'))
    def __call__(
        self,
        *,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        limited: bool = False,

        sigma_res: float | None = None,
        w: int | None = None,
        v_sep: float | None = None,
        min_fittable_total: int | None = None,
        min_fittable_ratio: float | None = None,
        evaluate_initial: float | int | None = None,
        aggressive: bool | None = None,
        crop: bool | None = None,
        measure: str | None = None,
        reverse: bool | None = None,
        fitter: FitterInstance | None = None,
    ) -> bool:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if bg_flux is None:
            bg_flux = self.default_bg
        
        assert len(self.lines) > 0

        logger.debug(f"Starting pipeline for {self.__str__(True)}")

        with stopwatch() as watch:
            success = self.prepareLines.__wrapped__(
                self,
                v_sep = v_sep,
                min_fittable_total = min_fittable_total,
                min_fittable_ratio = min_fittable_ratio,
            )
            if not success: 
                logger.warning(">>> Failed pipeline during 'prepareLines'!")
                return False
            
            if with_neighbours:
                success = self.prepareNeighbours.__wrapped__(
                    self,
                    sigma_res = sigma_res,
                )
                if not success: 
                    logger.warning(
                        ">>> Failed pipeline during 'prepareNeighbours'!"
                    )
                    return False

            success = self.instantiateModels.__wrapped__(
                self,
                bg_flux = bg_flux,
                without_rejections = without_rejections,
                without_absorption = without_absorption,
                with_neighbours = with_neighbours,
            )
            if not success: 
                logger.warning(
                    ">>> Failed pipeline during 'instantiateModels'!"
                )
                return False

            success = self.makeInitialFit.__wrapped__(
                self,
                bg_flux = bg_flux,
                without_rejections = without_rejections,
                without_absorption = without_absorption,
                with_neighbours = with_neighbours,
                evaluate_initial = evaluate_initial,
                fitter = fitter,
            )
            if not success:
                logger.warning(
                    ">>> Failed pipeline during 'makeInitialFit'!"
                )
                return False
            
            success = self.makeFinalFit.__wrapped__(
                self,
                bg_flux = bg_flux,
                without_rejections = without_rejections,
                without_absorption = without_absorption,
                with_neighbours = with_neighbours,
                limited = limited,
                w = w,
                aggressive = aggressive,
                crop = crop,
                measure = measure,
                reverse = reverse,
                evaluate_initial = evaluate_initial,
                v_sep = v_sep,
                fitter = fitter,
            )
            if not success:
                logger.warning(
                    ">>> Failed pipeline during 'makeFinalFit'!"
                )
                return False

        logger.debug(
            ">>> Finished entire pipeline in {:.1f} ms." \
            .format(1e3 * watch.elapsed)
        )
        return True

    @validated_apply_info_to_method(subjects=('lines',), specific_kwargs={'v_sep'})
    def getMask(
        self,
        *,
        covered: bool = True,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        valid: bool = False,
        log_valid: bool = False,
        line: str | float | None = None,
        limited: bool = True,
        v_sep: float | None = None,
    ) -> BoolVector:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        mask = super().getMask.__wrapped__(
            self,
            covered = covered,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
            valid = valid,
            log_valid = log_valid,
        )
        if with_neighbours and self.neighbours != (None, None):
            for cwindow in filter(lambda w: w is not None, self.neighbours):
                mask |= cwindow.__wrapped__.getMask(
                    cwindow,
                    covered = covered,
                    without_rejections = without_rejections,
                    without_absorption = without_absorption,
                    valid = valid,
                    log_valid = log_valid,
                )

        if line is not None:
            if isinstance(line, str):
                if limited:
                    bounds = self.i_bounds.get(line, (-inf, inf))
                else:
                    _line = self.lines[line]
                    bounds = (_line * (1 - v_sep), _line * (1 + v_sep))
            else:
                bounds = (line * (1 - v_sep), line * (1 + v_sep))

            idx_left, idx_right = get_bounds_indices(self._x, bounds)

            lmask = mask.copy()
            lmask[:] = False
            lmask[idx_left:idx_right+1] = True

            mask &= lmask

        return mask
    
    @validated_apply_info_to_method(subjects=('lines',), specific_kwargs={'v_sep'})
    def getMaskedCoords(
        self, 
        *,
        covered: bool = False,
        log: bool = False,
        without_rejections: bool = False, 
        without_absorption: bool = False,
        with_neighbours: bool = False,
        valid: bool = False,
        log_valid: bool = False,
        bg_flux: BackgroundFlux | None = None,

        line: float | str | None = None,
        limited: bool = True,
        v_sep: float | None = None,
    ) -> tuple[FloatVector, FloatVector, FloatVector, FloatVector]:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        x = self._x_log if log else self._x
        dy = self._dy_log if log else self._dy

        y = self._y.copy()
        y_smooth = self._y_smooth.copy()
        for bg in bg_flux:
            y        -= getattr(self, f"_y_{bg}")
            y_smooth -= getattr(self, f"_y_{bg}")

        if log: 
            y        = get_log(y, self.y0, self._log_valid_pixels)
            y_smooth = get_log(y_smooth, self.y0, self._log_valid_pixels)

        mask = self.getMask.__wrapped__(
            self,
            covered = covered,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
            with_neighbours = with_neighbours,
            valid = valid,
            log_valid = log_valid,
            line = line,
            limited = limited,
            v_sep = v_sep,
        )
        return x[mask], y[mask], dy[mask], y_smooth[mask]

    @validated_apply_info_to_method(subjects=('lines',))
    def add(
        self, 
        name: str,
        line: float, 
        n_max: int,
        needs_line: str | None = None,
        is_copy_of: str | None = None,
        *,
        strength_bounds: AstropyBounds | None = None,
        v_off_bounds: AstropyBounds | None = None,
        sigma_v_bounds: AstropyBounds | None = None,
        scale_init: float | None = None,
        scale_bounds: AstropyBounds | None = None,
        force_add: bool = False,
    ) -> bool:
        """
        ** PYDANTIC VALIDATED METHOD **

        Adds a given line to the SubSlice under the condition that it falls 
        within the covered wavelength range.

        Parameters
        ----------
        line : float
            Unitless rest wavelength of the emission line. 
        v_off : float
            Maximum absolute velocity offset of the emission line in units of 
            the speed of light.
        n_max : int
            Maximum allowed number of profile functions to model the emission 
            line.
        name : str
            Name of the emission line model(s). 
        """
        if     ((self.x_bounds[0] < line) and (line < self.x_bounds[1])) \
            or force_add:
            self.names.add(name)
            
            self.lines          [name] = line
            self.n_maxs         [name] = n_max

            self.strength_bounds[name] = strength_bounds
            self.sigma_v_bounds [name] = sigma_v_bounds
            self.v_off_bounds   [name] = v_off_bounds

            if bool(needs_line):
                self.needs_line  [name] = needs_line

            if bool(is_copy_of):
                self.is_copy_of  [name] = is_copy_of
                self.scale_init  [name] = scale_init
                self.scale_bounds[name] = scale_bounds  

            return True

        return False 

    @validated_apply_info_to_method(subjects=('lines',))
    def prepareLines(
        self, 
        *,
        v_sep: float | None = None,
        min_fittable_total: int | None = None,
        min_fittable_ratio: float | None = None,
    ) -> bool:
        """
        ** PYDANTIC VALIDATED METHOD **

        Performs the following steps:
        1.  Sorts the expected emission lines, and truncates the SubSlice if 
            necessary. 
        2.  Checks for heavy absorption. 

        Parameters
        ----------
        v_sep : float
            Velocity separation (units of the speed of light) used to define the
            spectral region relevant for a given emission line. Default is 'self.default_kwargs['v_sep']'.
        crit_abs_ratio : float, optional
            Critical absorption ratio (# of absorbed pixels / # of pixels). 
            Lines with absorption ratios exceeding this value are deemed as too
            absorbed, and are limited to a single profile function. If None, no
            emission lines are limited based on absorption. Default is 'self.default_kwargs['crit_abs_ratio'].
        """        
        # STEP 1

        self.i_bounds.clear()
        self.blacklist.clear()
        self._blacklist.clear()

        if len(self.lines) == 0:
            # No lines to consider.
            return False
        
        elif len(self.lines) == 1:
            # Only one line to consider.
            name = next(iter(self.names))
            
            line = self.lines[name]
            (lower, upper) = self.v_off_bounds[name]

            self.x_bounds = (
                max(self.x_bounds[0], line * (1 - v_sep)),
                min(self.x_bounds[1], line * (1 + v_sep)),
            )
            self.i_bounds[name] = self.x_bounds
            self.v_off_bounds[name] = (
                max(lower, self.x_bounds[0] / line - 1),
                min(upper, self.x_bounds[1] / line - 1)
            )

        else:
            # Sorted array of unique lines
            _lines = unique(list(self.lines.values()))

            # Multiple lines to consider. 
            # Adjust integration bounds
            _i_bounds = empty(_lines.size+1)
            _i_bounds[0] = max(self.x_bounds[0], _lines[0] * (1 - v_sep))
            _i_bounds[1:-1] = common_middle(
                _lines[:-1], _lines[1:],
                v_sep, v_sep,
            )
            _i_bounds[-1] = min(self.x_bounds[1], _lines[-1] * (1 + v_sep))

            _dict = dict(zip(_lines, zip(_i_bounds[:-1], _i_bounds[1:])))
            for name, line in self.lines.items():
                self.i_bounds[name] = _dict[line]

            self.x_bounds = (_i_bounds[0], _i_bounds[-1])

            # Update velocity offset bounds

            v_off_bounds: dict[float, tuple[float, float]] = {}
            for name, line in self.lines.items():
                b = self.v_off_bounds[name]
                if line in v_off_bounds:
                    # Use the largest possible bounds
                    v_off_bounds[line] = (
                        min(v_off_bounds[line][0], b[0]),
                        max(v_off_bounds[line][1], b[1]),
                    )
                else:
                    v_off_bounds[line] = b

            lower_bounds = array([v_off_bounds[line][0] for line in _lines])
            upper_bounds = array([v_off_bounds[line][1] for line in _lines])

            _x = empty(_lines.size+1)
            _x[0] = max(self.x_bounds[0], _lines[0] * (1 + lower_bounds[0]))
            
            _x[1:-1] = common_middle(
                _lines[:-1], _lines[1:],
                -upper_bounds[:-1], lower_bounds[1:],   #! Does this make sense?
            )
            _x[-1] = min(self.x_bounds[1], _lines[-1] * (1 + upper_bounds[-1]))

            x1_dict = dict(zip(_lines, _x[:-1]))
            x2_dict = dict(zip(_lines, _x[1:]))
            for name, line in self.lines.items():
                curr_bounds = self.v_off_bounds[name]
                self.v_off_bounds[name] = (
                    max(curr_bounds[0], x1_dict[line] / line - 1),
                    min(curr_bounds[1], x2_dict[line] / line - 1),
                )

        # STEP 2
        for name, n_max in self.n_maxs.items():
            self.blacklist [name] = (n_max == 1)
            self._blacklist[name] = (n_max == 1)

        return self.gradeLines.__wrapped__(
            self,
            with_neighbours = True,
            min_fittable_total = min_fittable_total,
            min_fittable_ratio = min_fittable_ratio,
            v_sep = v_sep,
        )

    @validated_apply_info_to_method(subjects=('loading',), specific_kwargs={'sigma_res'})
    def prepareNeighbours(
        self,
        *,
        sigma_res: float | int | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if self.spectrum is None:
            msg = "LWindow must have a `spectrum` attribute to prepare \
                neighbours!"
            logger.critical(msg)
            raise ValueError(msg)
        elif self.spectrum['pl'] is None:
            msg = "LWindow's `spectrum` must have a `continuum_windows` \
                attribute to prepare neighbours!"
            logger.critical(msg)
            raise ValueError(msg)

        if isinstance(sigma_res, float):
            iterations = int(sigma_res // self.info.loading['sigma_res'])
        else:
            iterations = sigma_res

        vicinity = binary_dilation(self.mask, iterations=iterations)

        left = None
        right = None
        for cwindow in self.spectrum.continuum_windows:
            if not (cwindow.mask & vicinity).any(): 
                continue

            if cwindow.x_bounds[1] < self.x_bounds[0]: 
                left = cwindow
            elif self.x_bounds[1] < cwindow.x_bounds[0]: 
                right = cwindow

        self.neighbours = (left, right)

        return self

    @validated_apply_info_to_method(subjects=('lines',))
    def gradeLines(
        self,
        with_neighbours: bool = False,
        *,
        min_fittable_total: int | None = None,
        min_fittable_ratio: float | None = None,
        v_sep: float | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        logger.debug(f"Grading lines in {self.__str__(simple=True)}:")

        lines_to_remove: set[str] = set()

        self.sortLines()
        for name, line in self.lines.items():
            mask = self.getMask.__wrapped__(
                self,
                with_neighbours = with_neighbours,
                covered = True,
                line = line,
                limited = True,
                v_sep = v_sep,
            )
            absorbed_pixels = self._absorbed_pixels[mask]
            valid_pixels = self._valid_pixels[mask]

            n = mask.sum()
            f = (valid_pixels & invert(absorbed_pixels)).sum()
            a = (valid_pixels & absorbed_pixels).sum()
            i = invert(valid_pixels).sum()

            msg = ">>> [{:.1f} <-> {:.1f}] w/ {}/{n} (fit.)," \
                "{}/{n} (abs.), {}/{n} (inv.): " \
                .format(*self.i_bounds[name], f, a, i, n=n)
            
            if f == 0:
                msg += "No valid data -> removing line '{}' at {:.1f}!" \
                    .format(name, line)
                lines_to_remove.add(name)

            elif (f / n < min_fittable_ratio) or (f < min_fittable_total):
                msg += "Not enough valid data -> " \
                    "blacklisting line '{}' at {:.1f}!" \
                    .format(name, line)
                self.blacklist[name] = True
            
            else:
                msg += "Enough valid data for line '{}' at {:.1f}!" \
                    .format(name, line)

            logger.debug(msg)

        for name in lines_to_remove: 
            self.removeLine(name)

        return True

    @validated_apply_info_to_method(subjects=('lines',), specific_kwargs={'v_sep'})
    def instantiateModels(
        self, 
        *,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        v_sep: float | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **

        Quickly instantiates each submodel using the data in its nearest
        vicinity. The amgorithm is described in my thesis (steps 1 and 2 prefer
        smoothed flux density values, step 3 does not.):

        1.  The mean is calculated using the mean rest wavelength value for 
            pixels whose flux density values exceed half the maximum value. The
            corresponding velocity offset is calculated using the theoretical
            rest wavelength. The velocity offset is adjusted to fit the accepted
            bounds. 

            !!! Method is from lmfit. 

        2.  The intrinsic velocity dispersion is inferred using the FWQM (full 
            width at a quarter maximum) of the data, after correcting for a
            non-zero velocity resolution. The velocity dispersion is adjusted to
            fit the accepted bounds. 

        3. The line strength is calculated as the discrete integral of the data.

        Lastly, the 'blacklist' (dict), 'fits' (dict) and 'fits_info' (dict)
        attributes are reset to their default values. 

        Note
        ----
        Steps 1 and 2 prefer smoothed data. Step 3 does not. 
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        logger.debug(
            "Instantiating models for {}:" \
            .format(self.__str__(True).removesuffix('.'))
        )

        self.sortLines()
        models = []
        for name, line in self.lines.items():
            if line in self.cropped:
                continue

            x, y, _, y_smooth = self.getMaskedCoords.__wrapped__(
                self,
                valid = True,
                bg_flux = bg_flux,
                without_rejections = without_rejections,
                without_absorption = without_absorption,
                with_neighbours = with_neighbours,
                line = name,
                limited = True,
                v_sep = v_sep,
            )
            try:
                model = GaussianModel.instantiate(
                    line,
                    x, y, y_smooth,
                    name = name,
                    strength_bounds = self.strength_bounds[name],
                    v_off_bounds = self.v_off_bounds[name],
                    sigma_v_bounds = self.sigma_v_bounds[name],
                    sigma_res = self.info.loading['sigma_res'],
                    logger = logger,
                )
                logger.debug(
                    ">>> Successfully instantiated line '{}' at {:.1f}." \
                    .format(name, line)
                )
            except Exception as e:
                logger.warning(
                    ">>> Failed instantiating line '{}' at {:.1f} due to:\n{}" \
                    .format(name, line, e)
                )
                continue

            models.append(model)

        if len(models) == 0: 
            return False

        self.model = sum(models[1:], start=models[0])
        # Remove current fit if existant
        self.fit = None

        # Reset blacklist and previous fits
        self.blacklist = self._blacklist.copy()
        self.fits.clear()
        self.fit_infos.clear()

        return self

    @validated_apply_info_to_method(subjects=('nonlinear',), specific_kwargs={'fitter'})
    def fitModel(
        self,
        *,
        update_flux: bool = False,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        fitter: FitterInstance | None = None,
    ) -> bool:
        """
        ** PYDANTIC VALIDATED METHOD **

        Fits the available model using the specified non-linear fitting 
        algorithm. The resultant fit is assigned to the 'fit' attribute, and 
        saved in the 'fits' (dict) attribute whereafter in can be accessed using 
        the corresponding fit complexity (number of profile functions). The fit 
        info is saved in the 'fit_infos' (dict) attribute. 
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        msg = "Fitting model for {}: " \
            .format(self.__str__(simple=True).removesuffix('.'))
        
        coords = self.getMaskedCoords.__wrapped__(
            self,
            covered = True,
            valid = True,

            without_rejections = without_rejections,
            without_absorption = without_absorption,
            with_neighbours = with_neighbours,
            bg_flux = bg_flux,
        )[:3]
        msg += f"no. of Gaussians: {self.model.n_submodels}, "
        msg += f"no. of data points: {len(coords[0])}. "
        
        try:
            with stopwatch() as watch:
                fit, fit_info = fitter(
                    self.model,
                    *coords, 
                    False,
                )
        except ValidationError as e:
            logger.warning(
                msg + f"Failed fitting due to validation error:\n{e}"
            )
            return False
        except ValueError as e:
            ms = [self.model] if self.model.n_submodels == 1 else self.model
            for m in ms:
                msg += f"Model: {m.name}, "
                msg += "strength: {:.1e}|{:.1e}|{:.1e}, ".format(m.strength.value, *m.strength.bounds)
                msg += "sigma_v: {:.1e}|{:.1e}|{:.1e}, ".format(m.sigma_v.value, *m.sigma_v.bounds)
                msg += "v_off: {:.1e}|{:.1e}|{:.1e}, ".format(m.v_off.value, *m.v_off.bounds)
                msg += "\n"
            raise ValueError(msg + f"Failed fitting due to value error:\n{e}")
        
        msg += "Finished fitting in {:.1f} ms." \
            .format(1e3 * watch.elapsed)
        logger.debug(msg)

        self.applyFit.__wrapped__(
            self, fit, fit_info,
            freeze = False,
        )
        if update_flux: 
            self.updateLinesEmission.__wrapped__(self, fit)

        return True
    
    @validated_apply_info_to_method(subjects=('lines', 'nonlinear'))
    def makeInitialFit(
        self, 
        *,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        evaluate_initial: float | int | None = None,
        fitter: FitterInstance | None = None,
    ) -> bool:
        """
        ** PYDANTIC VALIDATED METHOD **

        If necessary, instantiates the initial model (one profile function per 
        known emission line), and fits the model. 
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        if self.model is None:
            success = self.instantiateModels.__wrapped__(
                self,
                bg_flux = bg_flux,
                without_rejections = without_rejections,
                without_absorption = without_absorption,
                with_neighbours = with_neighbours,
            )
            if not success: 
                return False

        success = self.fitModel.__wrapped__(
            self,
            bg_flux = bg_flux,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
            with_neighbours = with_neighbours,
            fitter = fitter,
        )
        if not success: 
            return False

        if isinstance(evaluate_initial, (int, float)):
            x, _, dy = self.getMaskedCoords.__wrapped__(
                self,
                covered = True,
                valid = True,
                bg_flux = bg_flux,
                without_rejections = without_rejections,
                without_absorption = without_absorption,
                with_neighbours = with_neighbours,
            )[:3]

            for f in ([self.fit] if self.fit.n_submodels == 1 else self.fit):
                pure_name: str = f.pure_name
                
                if self.blacklist[pure_name]: 
                    continue

                # Interpolate noise level and compare with peak flux density
                crit_val = evaluate_initial * interp(f.mu, x, dy)
                self.blacklist[pure_name] = (f.peak < crit_val)

        return True

    @validated_apply_info_to_method(subjects=('lines',), specific_kwargs={'w'})
    def addLine(
        self,
        *,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        w: int | None = None,            #! Make 'lines'-specific window size?
    ) -> GaussianModel:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        n = self.fit.n_submodels
        x, y, dy = self.getMaskedCoords.__wrapped__(
            self,
            covered = True,
            valid = True,
            bg_flux = bg_flux,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
            with_neighbours = with_neighbours,
        )[:3]

        if (n == 1):
            f = self.fit
            data = (x, dy, (y - f(x)) / dy)
        else:
            h = w // 2

            valid_lines: list = [
                f
                for f in self.fit
                if not self.blacklist[f.pure_name]
            ]
            waves = [f.wave for f in valid_lines]

            z = abs((y - self.fit(x)) / dy)
            z_convolved = convolve(
                z, 
                exp(-linspace(-3, 3, w)**2), 
                mode = 'valid',
            )
            z_interp = interp(waves, x[h:-h], z_convolved)

            f = valid_lines[argmax(z_interp).flatten()[0]]
            (lb, ub) = self.i_bounds[f.pure_name]
            mask = (lb <= x) & (x < ub)
            data = (x[mask], dy[mask], z[mask])

        return f.makeCopy(*data)
    
    @validated_apply_info_to_method(subjects=('lines', 'nonlinear'))
    def makeFinalFit(
        self,
        *,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        limited: bool = False,

        w: int | None = None,
        aggressive: bool | None = None,
        crop: bool | None = None,
        measure: str | None = None,
        reverse: bool | None = None,
        evaluate_initial: float | int | None = None,
        v_sep: float | None = None,
        fitter: FitterInstance | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        !!! veto func?
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        # Instantiate if necessary
        if self.fit is None:
            self.makeInitialFit.__wrapped__(
                self,
                bg_flux = bg_flux,
                without_rejections = without_rejections,
                without_absorption = without_absorption,
                with_neighbours = with_neighbours,
                evaluate_initial = evaluate_initial,
                fitter = fitter,
            )

        continue_fitting: bool = True

        # Check if any lines aren't already blacklisted
        if all(self.blacklist.values()):
            # No additional models necessary
            continue_fitting = False
        
        elif self.fit.n_submodels <= 2:        
            # Check if under-fitted
            line = (
                self.fit.wave
                if self.fit.n_submodels == 1
                else None
            )
            continue_fitting = self.isUnderFitted.__wrapped__(
                self,
                self.fit.n_submodels,
                bg_flux = bg_flux,
                without_rejections = without_rejections,
                without_absorption = without_absorption,
                with_neighbours = with_neighbours,
                line = line,
                limited = limited,
                v_sep = v_sep,
            )

        current_config = self.getConfiguration()
        all_configs = [tuple(current_config.values())]

        while continue_fitting:
            self.updateBlacklistFromConfiguration()
            if all(self.blacklist.values()):
                break

            # Add a line component
            new_model = self.addLine.__wrapped__(
                self,
                bg_flux = bg_flux,
                without_rejections = without_rejections,
                without_absorption = without_absorption,
                with_neighbours = with_neighbours,
                w = w,
            )
            current_config[new_model.pure_name] += 1

            self.model = self.fit + new_model
            self.fitModel.__wrapped__(
                self,
                bg_flux = bg_flux,
                without_rejections = without_rejections,
                without_absorption = without_absorption,
                with_neighbours = with_neighbours,
                fitter = fitter,
            )

            # Check if current configuration has already been tried
            _config = tuple(current_config.values())
            if _config in all_configs:
                break
            
            all_configs.append(_config)

            # Check if model is over-fitted
            is_over, is_over_features = self.isOverFitted.__wrapped__(
                self,
                self.fit.n_submodels - 1,
                self.fit.n_submodels, 
                bg_flux = bg_flux,
                without_rejections = without_rejections,
                without_absorption = without_absorption,
                with_neighbours = with_neighbours,
                line = new_model.wave,
                limited = limited,
                v_sep = v_sep,
            )
            if is_over or False: #? 'False' is placeholder for VETO function
                # Check whether models are saturated (blacklisted) or touching 
                # their respective bounds. 
                fs = self.fit if (self.fit.n_submodels > 1) else [self.fit]
                is_redundant: list[bool] = [
                    f.isTouchingBounds() & self.blacklist[f.pure_name] \
                        if isinstance(f, GaussianModel) \
                        else False \
                    for f in fs
                ]

                if all(is_redundant):
                    continue_fitting = False

                elif any(is_redundant) and crop:
                    submodels = [
                        f \
                        for (idx, f) in enumerate(fs) if not is_redundant[idx]
                    ]

                    self.model = sum(submodels[1:], start=submodels[0])
                    self.fitModel.__wrapped__(
                        self,
                        update_flux = False,
                        bg_flux = bg_flux,
                        without_rejections = without_rejections,
                        without_absorption = without_absorption,
                        with_neighbours = with_neighbours,
                        fitter = fitter,
                    )

                else:
                    continue_fitting = False
                    self.blacklist[new_model.pure_name] = True

                    if not aggressive:
                        self.fit = self.fits[self.fit.n_submodels - 1]

        # Perform final model cropping
        if crop:
            self.cropFit.__wrapped__(
                self,
                bg_flux = bg_flux,
                without_rejections = without_rejections,
                without_absorption = without_absorption,
                with_neighbours = with_neighbours,
                limited = limited,
                v_sep = v_sep,
                measure = measure,
                reverse = reverse,
                fitter = fitter, 
            )
        else:
            self.updateLinesEmission.__wrapped__(self, self.fit)

        # Update submodel names and order
        self.reformatFit()
        
        return self

    @validated_apply_info_to_method(subjects=('lines', 'nonlinear'))
    def cropFit(
        self,
        *,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        limited: bool = False,
        v_sep: float | None = None,
        measure: str | None = None,
        reverse: bool | None = None,
        fitter: FitterInstance | None = None,
    ) -> bool:
        """
        ** PYDANTIC VALIDATED METHOD **

        Performs a single or multiple iterations of model cropping, i.e. identi-
        fies submodels with weak scores and evaluates whether those submodels 
        are necessary. The algorithm works as follows:

        1.  All lines represented by a single submodel are identified and 
            considered for cropping. 
        2.  For each submodel (line), a quality measure is calculated. The 
            quality measure serves to quantify the importance (or significance) 
            of a specific submodel. 
        3.  All submodels are ranked, and the weakest (worst quality) submodel 
            is identified. A second model, identical to the initial model but 
            excluding the weakest submodel, is created. The second model is then 
            fitted. 
        4.  If the first (advanced) model over-fits compared to the second 
            (cropped) model, the submodel is justifiably removed, and the second 
            model is accepted. 
        5.  Steps 1-4 are repeated (recursively) until no submodel cropping is 
            done. 

        Parameters
        ----------
        measure : str
            The quality measure to use when quantifying the quality of a 
            submodel. Options are 'getPeakSNR', 'getFluxSNR', 'getLineSNR', and
            'getWeightedAbsorption'. Default is {}. 
        reverse : str
            Whether to reverse the order when ranking submodel qualities. If 
            False, low-to-high quality score corresponds to bad-to-good. If 
            True, high-to-low quality score corresponds to bad-to-good. Default 
            is {}. 

        Notes
        -----
        It does not crop line blends (multiple submodels representing a single 
        emission line) as additional submodels were initially justified and 
        added. 
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        msg = "Cropping fit for {}: " \
            .format(self.__str__(simple=True).removesuffix('.'))
        
        if self.fit is None:
            logger.debug(msg + "no 'fit' attribute found -> skipping cropping!")
            return False
        
        if self.fit.n_submodels == 1:
            logger.debug(msg + "only a single line -> skipping cropping!")
            return True
        
        fit = self.fit
        fs = fit if fit.n_submodels > 1 else (fit,)
        n = fit.n_submodels
        fit_info = self.fit_infos[n]

        configuration = self.getConfiguration()

        single_lines: list[GaussianModel] = []
        multiple_lines: list[GaussianModel | _VProfileCopy] = []
        for f in fs:
            if configuration[f.pure_name] == 1 and isinstance(f, GaussianModel):
                single_lines.append(f)
            else:
                multiple_lines.append(f)

        msg += "no. of single lines: {}, no. of multiple lines: {}. " \
            .format(len(single_lines), n - len(single_lines))
        
        if len(single_lines) == 0:
            logger.debug(msg + "No single lines -> skipping cropping!")
            return True
        
        elif (len(single_lines) == 1) and (n == 1):
            logger.debug(msg + "Only a single line -> skipping cropping!")
            return True
                    
        else:
            def key(f): return getattr(f, measure)(self)
            single_lines = sorted(single_lines, key=key, reverse=reverse)

        removed_line = single_lines.pop(0)
        submodels = single_lines + multiple_lines
        self.model = sum(submodels[1:], start=submodels[0])

        success = self.fitModel.__wrapped__(
            self,
            update_flux = False,
            bg_flux = bg_flux,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
            with_neighbours = with_neighbours, 
            fitter = fitter, 
        )
        if not success:
            msg += "Fitting the cropped model failed -> skipping cropping!"
            return False
        
        is_over, is_over_features = self.isOverFitted.__wrapped__(
            self,
            self.fit.n_submodels,       #? Cropped fit
            self.fit.n_submodels + 1,   #? Previous fit
            bg_flux = bg_flux,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
            with_neighbours = with_neighbours,
            line = removed_line.wave,
            limited = limited,
            v_sep = v_sep,
        )
        if is_over or False: #? 'False' is placeholder for VETO function
            #* Removing the single line had no significant impact
            #* on fit quality, i.e. using the more advanced model constitutes
            #* over-fitting. 
            #* => Recursive call...
            self.cropped.add(removed_line.pure_name)
            return self.cropFit.__wrapped__(
                self,
                bg_flux = bg_flux,
                without_rejections = without_rejections,
                without_absorption = without_absorption,
                with_neighbours = with_neighbours,
                limited = limited,
                v_sep = v_sep,
                measure = measure,
                reverse = reverse,
                fitter = fitter,
            )
        else:
            #* Recursion ends -> update 'fit' and 'fit_info' attributes
            self.applyFit.__wrapped__(
                self, fit, fit_info,
                freeze = False,
            )
            self.updateLinesEmission.__wrapped__(
                self, fit,
            )
            return True

    @validate_call
    def getModel(self, thaw: bool = False) -> CompoundModel_ | None:
        """
        ** PYDANTIC VALIDATED METHOD **

        Retrieves this 'LWindow's current fit/model if available. 

        NOTES
        -----
        If 'thaw' is True, a copy of the model is retrieved. Fitting the 
        retrieved model inplace will therefore NOT update the 'LWindow's model, 
        and the fit will need to be applied using" 'ApplyFit'. 
        """
        fit = self.fit or self.model
        if fit is None:
            return None
        
        if thaw:
            fit = fit.copy()
            fs = (fit,) if fit.n_submodels == 1 else fit
            for f in filter(lambda f: isinstance(f, _VProfileCopy), fs):
                f._thaw_velocity_profile(inplace=True)
                f._remember_ties(inplace=True)

        return fit

    @validate_call
    def createVelocityProfileCopy(
        self,
        master: str,
        wave: float,
        mimic: str,
        freeze: bool = False,
        model_kwargs: dict = {},
    ) -> _VProfileCopy:
        """
        ** PYDANTIC VALIDATED METHOD **

        Creates a copy of an emission lines velocity profile, with the centre
        'wave' (float) and gives it the name 'name' (str). All submodels with
        a 'pure_name' equal to 'master' are used for the velocity profile. 
        """
        if master not in self.names:
            raise ValueError(
                f"Line '{master}' not found in "
                f"{self.__str__(simple=True).removesuffix('.')}!  Available "
                f"names are: {self.names}")

        model = self.getModel.__wrapped__(self, thaw=False)
        ms = (model,) if model.n_submodels == 1 else model
        names = set(m.pure_name for m in ms)

        if master not in names:
            raise ValueError(
                f"Line '{master}' not found in the current model of "
                f"{self.__str__(simple=True).removesuffix('.')}! Available "
                f"lines are: {names}"
            )

        submodels = [m for m in ms if m.pure_name == master]

        return VProfileCopyDict[len(submodels)].from_model(
            wave,
            mimic,
            sum(submodels[1:], start=submodels[0]),
            freeze = freeze,
            **model_kwargs,
        )
    
    @validated_apply_info_to_method(subjects=('lines',), specific_kwargs={'adapt_scale'})
    def applyMyself(
        self,
        line_windows: list[SpecData] | None = None,
        *,
        adapt_scale: bool | None = None,
    ) -> Self:
        """
        Applies this 'LWindow's velocity profile copies to other 'LWindow's.  

        NOTES
        -----
        When creating the velocity-profile copy, the keyword argument 'freeze'
        is set to True. 

        This method does not use Pydantic type validation due to the use of:
        'list[Self]' in the type annotation of 'line_windows'. 
        """
        if line_windows is None:
            assert self.spectrum is not None
            line_windows = self.spectrum.em

        for master, children in self.copies_to.items():
            for idx_d, mimic in children:
                lwindow = line_windows[idx_d]

                # Check if master line has been cropped
                if master in self.cropped:
                    # Crop corresponding mimic line
                    lwindow.cropped.add(mimic)
                    continue

                lwindow.applyVelocityProfileCopy.__wrapped__(
                    lwindow,
                    self.createVelocityProfileCopy(
                        master,
                        lwindow.lines[mimic],
                        mimic,
                        freeze = True,
                    ),
                    adapt_scale = adapt_scale,
                )

        return self
    
    @validated_apply_info_to_method(subjects=('lines',), specific_kwargs={'adapt_scale'})
    def applyVelocityProfileCopy(
        self,
        master_model: _VProfileCopy,
        *,
        adapt_scale: bool | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **

        Applies a velocity profile copy, 'master_model', to this 'LWindow'. This
        velocity profile copy replaces the existant submodels with same 
        'pure_name' attribute. 

        Successfully applying the velocity profile copy resets 'blacklist', 
        'fits' and 'fit_infos' attributes, and sets the default blacklist
        value of the velocity profile to 'True' and 'n_max' to 1, i.e. no 
        additional Gaussians will be placed on top of the velocity profile copy.
        """
        assert (mimic := master_model.pure_name) in self.names
        assert self.model is not None

        ms = self.model if (self.model.n_submodels > 1) else [self.model]
        replaced_ms = list(filter(
            lambda m: m.pure_name == mimic,
            ms,
        ))
        other_ms = list(filter(
            lambda m: m.pure_name != mimic,
            ms,
        ))
        val = apply_bounds(
            self.scale_init[mimic], 
            bounds := self.scale_bounds[mimic],
        )
        master_model.strength_scale.value  = val
        master_model.strength_scale.bounds = bounds
        master_model.strength_scale.fixed  = bool(val == bounds[0] == bounds[1])

        if adapt_scale and not master_model.strength_scale.fixed:
            master_model.adaptStrengthScale(replaced_ms)

        new_ms = sorted(
            other_ms + [master_model], 
            key = lambda m: m.sorting_key,
        )

        self.model = new_ms[0] \
            if (len(new_ms) == 1) \
            else sum(new_ms[1:], start=new_ms[0])
        
        # Remove current fit if existant
        self.fit = None

        # Reset blacklist and previous fits
        self._blacklist[master_model.pure_name] = True
        self.n_maxs    [master_model.pure_name] = 1

        self.blacklist = self._blacklist.copy()
        self.fits.clear()
        self.fit_infos.clear()

        return self

    @validate_call
    def applyFit(
        self,
        fit: GaussianModel | CompoundModel_[GaussianModel],
        fit_info: FitInfo,
        freeze: bool = False
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        
        Applies the 'fit' and 'fit_info' values to this 'LWindow' class. 

        NOTES
        -----
        If 'freeze' is True, a copy of the 'fit' value is applied.
        """
        if freeze:
            fit = fit.copy()
            fs = (fit,) if fit.n_submodels == 1 else fit
            for f in (f for f in fs if isinstance(f, _VProfileCopy)):
                f._freeze_velocity_profile(inplace=True)
                f._forget_ties(inplace=True)

        self.fit = fit
        self.fit_info = fit_info
        
        n = fit.n_submodels
        self.fits[n] = fit
        self.fit_infos[n] = fit_info

        return self
    
    @validate_call
    def adoptFit(
        self,
        fit: GaussianModel | CompoundModel_[GaussianModel],
        freeze: bool = False
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if freeze:
            fit = fit.copy()
            fs = (fit,) if fit.n_submodels == 1 else fit
            for f in (f for f in fs if isinstance(f, _VProfileCopy)):
                f._freeze_velocity_profile(inplace=True)
                f._forget_ties(inplace=True)

        n = fit.n_submodels
        self.fit = fit
        self.fit_info = None
        
        self.fits[n] = fit
        self.fit_infos[n] = None

        return self

    @validate_call
    def updateLinesEmission(
        self,
        model: GaussianModel | CompoundModel_[GaussianModel] | None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **

        Updates the '_y_em' attribute to account for a new emission line model,
        'model'. If no previous model is found, '_y_em' is updated directly. If 
        a previous model is found, this model's '_y_em' contribution is first 
        removed, and the new 'model's contribution is added. 

        If 'model' is not given, i.e. None, the previous model's contribution
        is removed. 
        """
        mask = isfinite(self._x)
        if self.prev_model is not None:
            self._y_em[mask] -= self.prev_model(self._x[mask])
            self.prev_model = None

        if model is not None:
            self._y_em[mask] += model(self._x[mask])
            self.prev_model = model.copy()

        return self

    def removeLine(
        self,
        name: str,
    ) -> Self:
        if name not in self.names:
            raise ValueError(f"line '{name}' not in 'self.names'!")
        
        self.names.remove(name)
        
        del self.lines [name]
        del self.n_maxs[name]
        
        del self.strength_bounds[name]
        del self.sigma_v_bounds [name]
        del self.v_off_bounds   [name]

        if name in self.i_bounds:
            del self.i_bounds[name]
        if name in self.blacklist:
            del self.blacklist[name]
        if name in self._blacklist:
            del self._blacklist[name]

        return self

    def sortLines(self) -> Self:

        if (diff(list(self.lines.values())) > 0).all(): 
            return self

        self.lines: dict[str, float] = dict(sorted(
            self.lines.items(),
            key=lambda item: item[1],
        ))
        self.names = set(self.lines.keys())

        # def _sorted_dict(d: dict, lines=self.lines) -> dict:
        #     return dict(sorted(
        #         d.items(), key=lambda item: lines[item[0]],
        #     ))

        # self.n_maxs = _sorted_dict(self.n_maxs)
        # self.strength_bounds = _sorted_dict(self.strength_bounds)
        # self.sigma_v_bounds = _sorted_dict(self.sigma_v_bounds)
        # self.v_off_bounds = _sorted_dict(self.v_off_bounds)

        # if hasattr(self, 'i_bounds'):
        #     self.i_bounds = _sorted_dict(self.i_bounds)
        # if hasattr(self, 'blacklist'):
        #     self.blacklist = _sorted_dict(self.blacklist)
        # if hasattr(self, '_blacklist'):
        #     self._blacklist = _sorted_dict(self._blacklist)

        return self

    def getConfiguration(self) -> dict[str, int]:

        if (model := self.getModel()) is None: 
            return {}
        ms = (model,) if model.n_submodels == 1 else model
        return Counter(m.pure_name for m in ms)
    
    def updateBlacklistFromConfiguration(self) -> Self:
        configuration = self.getConfiguration()

        for name in filter(lambda name: not self.blacklist[name], self.names):
            self.blacklist[name] = (self.n_maxs[name] == configuration[name])

        return self

    def reformatFit(self) -> Self:
        assert self.fit is not None

        if self.fit.n_submodels == 1:
            self.fit.name = self.fit.pure_name
        else:
            fs = order_submodels(self.fit, combine=False)
            current_count = defaultdict(lambda: 1)
            max_count = self.getConfiguration()

            for f in filter(lambda f: max_count[f.pure_name] > 1, fs):
                f.name = '#'.join([
                    f.pure_name,
                    str(current_count[f.pure_name]),
                ])
                current_count[f.pure_name] += 1

            self.fit = sum(fs[1:], start=fs[0])

        return self

    @validated_apply_info_to_method(subjects=('lines',), specific_kwargs={'v_sep'})
    def isUnderFitted(
        self,
        n: int,
        *,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        line: str | float | None = None,
        limited: bool = False,
        v_sep: float | None = None,
    ) -> tuple[bool, FloatVector]:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        coords = self.getMaskedCoords.__wrapped__(
            self,
            covered = True,
            valid = True,
            bg_flux = bg_flux,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
            with_neighbours = with_neighbours,
            line = line,
            limited = limited,
            v_sep = v_sep,
        )[:3]
        under = Under(
            *coords,
            fit = self.fits[n],
            snr = self.info.lines['snr'],       #! Changing snr has not effect on model!
        )
        return under(), under.getFeatures(as_dict=True)
    
    @validated_apply_info_to_method(subjects=('lines',), specific_kwargs={'v_sep'})
    def isOverFitted(
        self,
        n: int,
        m: int,
        *,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        line: str | float | None = None,
        limited: bool = False,
        v_sep: float | None = None,
    ) -> tuple[bool, FloatVector]:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        assert n < m

        coords = self.getMaskedCoords.__wrapped__(
            self,
            covered = True,
            valid = True,
            bg_flux = bg_flux,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
            with_neighbours = with_neighbours,
            line = line,
            limited = limited,
            v_sep = v_sep,
        )[:3]
        over = Over(
            *coords,
            fit_initial = self.fits[n],
            fit_final = self.fits[m],
            snr = self.info.lines['snr'],       #! Changing snr has not effect on model!
        )
        return over(), over.getFeatures(as_dict=True)

    ### Error estimation

    @validated_apply_info_to_method(subjects=('error', 'nonlinear'))
    def instantiateBootstrapper(
        self,
        *,
        pl: PowerLawModel | None = None,
        fe: IronModel | None = None,
        ba: BalmerModel | None = None,
        hg: None = None,

        model_types: ModelTypes | None = None,

        without_rejections: bool = False,
        without_absorption: bool = False,

        pool: Pool_ | None = None,

        scale: Scale | None = None,
        variant: Variant | None = None,
        bootstrap_type: BootstrapType | None = None,
        n_sigmas: float | None = None,
        vary_lines: VaryLines | None = None,

        cfit: ContinuumFitResult | None = None,
        iterations: int | None = None,
        random_state: RandomState_ | None = None,
        renew_rng: bool | None = None,
        replace_missing: bool | None = None,
        tqdm_disable: bool | None = None,
        tqdm_leave: bool | None = None,

        fitter: FitterInstance | None = None,
    ) -> BaseBootstrapper:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        assert scale == 'local' or scale == 'semilocal'

        if self.bootstrapper is not None:
            logger.debug("Existing 'bootstrapper' will be overwritten.")

        cls, args, kwargs = format_bootstrapping_kwargs_for_lwindow(
            self,
            fitter,
            pl=pl,
            fe=fe,
            ba=ba,
            hg=hg,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            with_neighbours=(scale == 'semilocal'),
            variant=variant,
            model_types=model_types,
            bootstrap_type=bootstrap_type,
            n_sigmas=n_sigmas,
            vary_lines=vary_lines,
            pool=pool,
            cfit=cfit,
            iterations=iterations,
            random_state=random_state,
            renew_rng=renew_rng,
            replace_missing=replace_missing,
            tqdm_disable=tqdm_disable,
            tqdm_leave=tqdm_leave,
            logger=logger,
        )
        self.bootstrapper = cls(*args, **kwargs)
        return self.bootstrapper
    
    @validate_call
    def runBootstrapper(self) -> ErrorResult:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if self.bootstrapper is None:
            msg = "No bootstrapper instance found! \
                Run 'instantiateBootstrapper()' method first!"
            logger.critical(msg)
            raise RuntimeError(msg)
        if self.error_result is not None:
            logger.debug("Existing 'error_result' will be overwritten.")
        
        out = self.bootstrapper.run()
        self.error_result = self.bootstrapper.toErrorResult(out)
        return self.error_result

    @validated_apply_info_to_method(subjects=('nonlinear', 'error'), start=6)
    def bootstrap(
        self,
        *,
        without_rejections: bool = False,
        without_absorption: bool = False,
        cfit: ContinuumFitResult | None = None,
        pool: Pool_ | None = None,
        pl_model: PowerLawModel | None = None,
        fe_model: IronModel | None = None,
        ba_model: BalmerModel | None = None,

        fitter: FitterInstance | None = None,
        scale: Scale | None = None,
        remodel: bool | None = None,
        variant: Variant | None = None,
        replace_missing: bool | None = None,
        bootstrap_type: BootstrapType | None = None,
        vary_lines: VaryLines | None = None,
        n_sigmas: float | None = None,
        iterations: int | None = None,
        random_state: RandomState_ | None = None,
        renew_rng: bool | None = None,
        tqdm_disable: bool | None = None,
        tqdm_leave: bool | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        # from ..errors.bootstrapping import bootstrap
        # from ...typing_.misc.error_result import ErrorResultLike

        # logger.debug(f"Bootstrapping: {self}")

        # if pl_model is None:
        #     try:
        #         pl_model = self._spectrum.continuum_windows.getModel()
        #         if pl_model is None: 
        #             raise ValueError
        #         logger.debug(
        #             "Retrieved 'pl_model' from parent 'Spectrum' class!"
        #         )
        #     except ValueError:
        #         logger.warning(
        #             "Retrieved an unfitted 'pl_model' using parent 'Spectrum' " \
        #             "class!"
        #         )
        #     except AttributeError:
        #         logger.warning(
        #             "Can not retrieve any 'pl_model' as there is " \
        #             "no possible access to parent 'Spectrum' class!"
        #         )

        # if fe_model is None:
        #     try:
        #         fe_model = self._spectrum.iron_windows.getModel()
        #         if fe_model is None: 
        #             raise ValueError
        #         logger.debug(
        #             "Retrieved 'fe_model' from parent 'Spectrum' class!"
        #         )
        #     except ValueError:
        #         logger.warning(
        #             "Retrieved an unfitted 'fe_model' using parent 'Spectrum' " \
        #             "class!"
        #         )
        #     except AttributeError:
        #         logger.warning(
        #             "Can not retrieve any 'fe_model' as there is no possible \
        #             access to parent 'Spectrum' class!"
        #         )

        # if ba_model is None:
        #     try:
        #         ba_model = self._spectrum.balmer_windows.getModel()
        #         if ba_model is None: 
        #             raise ValueError
        #         logger.debug(
        #             "Retrieved 'ba_model' from parent 'Spectrum' class!"
        #         )
        #     except ValueError:
        #         logger.warning(
        #             "Retrieved an unfitted 'ba_model' using parent 'Spectrum' "\
        #             "class!"
        #         )
        #     except AttributeError:
        #         logger.warning(
        #             "Can not retrieve any 'ba_model' as there is no possible \
        #             access to parent 'Spectrum' class!"
        #         )

        # with_neighbours = (scale == 'semilocal')
        # coords = self.getMaskedCoords.__wrapped__.raw(
        #     self,
        #     covered = True,
        #     with_neighbours = with_neighbours,
        # )[:3]
        # mask = self.getMask.__wrapped__.raw(
        #     self,
        #     covered = True,
        #     with_neighbours = with_neighbours,
        # )

        # usable_pixels = self._valid_pixels[mask].copy()
        # if without_rejections: 
        #     usable_pixels &= invert(self._rejected_pixels[mask])
        # if without_absorption:
        #     usable_pixels &= invert(self._absorbed_pixels[mask])

        # bootstrapper, samples, _stopwatch = bootstrap.__wrapped__(
        #     self.__str__(simple=True),
        #     coords,
        #     scale,
        #     pl_model = pl_model,
        #     fe_model = fe_model,
        #     ba_model = ba_model,
        #     em_model = self.getModel(),

        #     usable_pixels = usable_pixels,
        #     pool = pool,
        #     info = self.info,

        #     cfit = cfit,
        #     n_sigmas = n_sigmas,
        #     vary_lines = vary_lines,

        #     remodel = remodel,
        #     variant = variant,
        #     bootstrap_type = bootstrap_type,
        #     replace_missing = replace_missing,
        #     iterations = iterations,
        #     random_state = random_state,
        #     renew_rng = renew_rng,
        #     tqdm_disable = tqdm_disable,
        #     tqdm_leave = tqdm_leave,
        #     fitter = fitter,
        # )
        # logger.debug(
        #     ">>> Completed bootstrapping in {:.3f} (s) | {:.1f} (it/s)." \
        #     .format(
        #         self.__str__(simple=True), 
        #         _stopwatch.elapsed,
        #         bootstrapper.iterations / _stopwatch.elapsed,
        #     )
        # )
        # self.bootstrap_result: ErrorResultLike = \
        #     bootstrapper.toErrorResult(samples)

        # return self
    
    @validated_apply_info_to_method(subjects=('error',))
    def analyseBootstrapSamples(
        self,
        directory: AbsoluteDirPath | None = None,
        paths: Iterable[RelativeFilePath] | None = None,
        *,
        out_lines: OutLines | None = None,
        out_measures: OutMeasures | None = None,
        percentiles: set[int] | None = None,
        res: int | None = None,
        render_width: float | None = None,
        fwhm_strategy: FWHMStrategy | None = None,
        exact: bool | None = None,
        v_int: float | None = None,
        ipv_int: float | None = None,
        dx_int: float | None = None,
        tqdm_disable: bool | None = None,
        tqdm_leave: bool | None = None,
    ) -> dict[float, QTable_] | None:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        # from ..complexes.analysis import analyse_error_samples
        # return analyse_error_samples.__wrapped__(
        #     self.bootstrap_result,
        #     self.info,
        #     directory = directory,
        #     paths = paths,
        #     out_waves = out_waves,
        #     out_measures = out_measures,
        #     percentiles = percentiles,
        #     res = res,
        #     render_width = render_width,
        #     fwhm_strategy = fwhm_strategy,
        #     exact = exact,
        #     v_int = v_int,
        #     ipv_int = ipv_int,
        #     dx_int = dx_int,
        #     tqdm_disable = tqdm_disable,
        #     tqdm_leave = tqdm_leave,
        # )

    def quickplot(
        self,
        *,
        figure: tuple[Figure, Axes] | None = None,
        figsize: tuple[float, float] = (8, 6),
        dpi: int = 300,
        title: str | None = None,
        n_sigma: float = 2.0,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        pl_color: str | None = DEFAULT_COLORS['pl'],
        fe_color: str | None = DEFAULT_COLORS['fe'],
        ba_color: str | None = DEFAULT_COLORS['ba'],
        hg_color: str | None = DEFAULT_COLORS['hg'],
        em_color: str | None = DEFAULT_COLORS['em'],
        sm_color: str | None = DEFAULT_COLORS['sm'],
        ab_color: str | None = DEFAULT_COLORS['ab'],
        xticks: tuple[float, float] | None = None,
        yticks: tuple[float, float] | None = None,
        logx: bool = False,
        logy: bool = False,
    ) -> tuple[Figure, Axes]:
        """
        Basic plotting routing for 'Spectrum' classes.

        NOTES
        -----
        This method overwrites but still calls the inherited '_quickplot'
        method, setting 'title' equal to 'self.title' by default.
        """
        xlim = xlim or self.x_bounds
        return quickplot(
            get_coords(self, x_bounds=xlim, replace_with_nan=False),
            self.info,
            figure=figure,
            figsize=figsize,
            dpi=dpi,
            title=title or self.__str__(simple=True).removesuffix('.'),
            n_sigma=n_sigma,
            xlabel=xlabel or 'auto',
            ylabel=ylabel or 'auto',
            xlim=xlim,
            ylim=ylim or 'auto',
            pl_color=pl_color,
            fe_color=fe_color,
            ba_color=ba_color,
            hg_color=hg_color,
            em_color=em_color,
            sm_color=sm_color,
            ab_color=ab_color,
            xticks=xticks or 'auto',
            yticks=yticks or 'auto',
            logx=logx,
            logy=logy,
        )

    def absorptionplot(
        self,
        *,
        figure: tuple[Figure, list[Axes, Axes, Axes]] | None = None,
        figsize: tuple[float, float] = (8, 6),
        dpi: int = 300,
        title: str | None = None,
        height_ratio: float = 3.0,
        n_sigma: float = 2.0,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
        plabel: str | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        zlim: tuple[float, float] | None = (-5, 5),
        plim: tuple[float, float] | None = (-5, 0),
        pl_color: str | None = DEFAULT_COLORS['pl'],
        fe_color: str | None = DEFAULT_COLORS['fe'],
        ba_color: str | None = DEFAULT_COLORS['ba'],
        hg_color: str | None = DEFAULT_COLORS['hg'],
        em_color: str | None = DEFAULT_COLORS['em'],
        sm_color: str | None = DEFAULT_COLORS['sm'],
        ab_color: str | None = DEFAULT_COLORS['ab'],
        xticks: tuple[float, float] | None = None,
        yticks: tuple[float, float] | None = None,
        zticks: tuple[float, float] | None = None,
        pticks: tuple[float, float] | None = None,
        logx: bool = False,
        logy: bool = False,
        logp: bool = True,
    ) -> tuple[Figure, list[Axes]]:
        xlim = xlim or self.x_bounds
        return absorptionplot(
            get_coords(self, x_bounds=xlim, replace_with_nan=False),
            self.info,
            figure=figure,
            figsize=figsize,
            dpi=dpi,
            title=title or self.__str__(simple=True).removesuffix('.'),
            height_ratio=height_ratio,
            n_sigma=n_sigma,
            z_crit=self.info.absorption.z_crit,
            p_crit=self.info.absorption.p_crit,
            xlabel=xlabel or 'auto',
            ylabel=ylabel or 'auto',
            zlabel=zlabel or 'auto',
            plabel=plabel or 'auto',
            xlim=xlim or 'auto',
            ylim=ylim or 'auto',
            zlim=zlim or 'auto',
            plim=plim or 'auto',
            pl_color=pl_color,
            fe_color=fe_color,
            ba_color=ba_color,
            hg_color=hg_color,
            em_color=em_color,
            sm_color=sm_color,
            ab_color=ab_color,
            xticks=xticks or 'auto',
            yticks=yticks or 'auto',
            zticks=zticks or 'auto',
            pticks=pticks or 'auto',
            logx=logx,
            logy=logy,
            logp=logp,
        )

    def fitplot(
        self,
        *,
        plot_components: bool = False,
        plot_type: Literal['difference', 'residual'] = 'difference',
        figure: tuple[Figure, list[Axes, Axes]] | None = None,
        figsize: tuple[float, float] = (8, 6),
        dpi: int = 300,
        title: str | None = None,
        height_ratio: float = 3.0,
        n_sigma: float = 2.0,
        xlabel: str | None = None,
        ylabel: str | None = None,
        dlabel: str | None = None,
        zlabel: str | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        dlim: tuple[float, float] | None = None,
        zlim: tuple[float, float] | None = (-5, 5),
        pl_color: str | None = DEFAULT_COLORS['pl'],
        fe_color: str | None = DEFAULT_COLORS['fe'],
        ba_color: str | None = DEFAULT_COLORS['ba'],
        hg_color: str | None = DEFAULT_COLORS['hg'],
        em_color: str | None = DEFAULT_COLORS['em'],
        sm_color: str | None = DEFAULT_COLORS['sm'],
        ab_color: str | None = DEFAULT_COLORS['ab'],
        xticks: tuple[float, float] | None = None,
        yticks: tuple[float, float] | None = None,
        dticks: tuple[float, float] | None = None,
        zticks: tuple[float, float] | None = None,
        logx: bool = False,
        logy: bool = False,
        cmap_name: str = 'tab20',
        distinguish_narrow: bool = True,
    ) -> tuple[Figure, list[Axes, Axes]]:
        xlim = xlim or self.x_bounds
        model = (self.spectrum or self).getModel() if plot_components else None
        if model is not None:
            ms = (model,) if model.n_submodels == 1 else model
            submodels = []
            for m in ms:
                if m.model_type in ('pl', 'fe', 'ba', 'hg') or m.pure_name in self.names:
                    submodels.append(m)

            model = sum(submodels[1:], start=submodels[0]) if submodels else None

        return fitplot(
            get_coords(self, x_bounds=xlim, replace_with_nan=False),
            self.info,
            model=model,
            plot_type=plot_type,
            figure=figure,
            figsize=figsize,
            dpi=dpi,
            title=title or self.__str__(simple=True).removesuffix('.'),
            height_ratio=height_ratio,
            n_sigma=n_sigma,
            xlabel=xlabel or 'auto',
            ylabel=ylabel or 'auto',
            dlabel=dlabel or 'auto',
            zlabel=zlabel or 'auto',
            xlim=xlim or 'auto',
            ylim=ylim or 'auto',
            dlim=dlim or 'auto',
            zlim=zlim or 'auto',
            pl_color=pl_color,
            fe_color=fe_color,
            ba_color=ba_color,
            hg_color=hg_color,
            em_color=em_color,
            sm_color=sm_color,
            ab_color=ab_color,
            xticks=xticks or 'auto',
            yticks=yticks or 'auto',
            dticks=dticks or 'auto',
            zticks=zticks or 'auto',
            logx=logx,
            logy=logy,
            cmap_name=cmap_name,
            distinguish_narrow=distinguish_narrow,
        )