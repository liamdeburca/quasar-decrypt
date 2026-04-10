from logging import getLogger
from typing import Optional, Iterable, Literal, Self
from dataclasses import dataclass, field
from numpy import isfinite, invert, argwhere, nanargmax, zeros_like, array_equal
from scipy.optimize import OptimizeResult

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from pydantic import validate_call
from pydantic_core import ValidationError

from quasar_errors.bootstrapping import BaseBootstrapper
from quasar_errors.spectrum_utils import format_bootstrapping_kwargs_for_spectrum
from quasar_errors.error_result import ErrorResult

from quasar_utils.setup import Info
from quasar_utils.wrappers import apply_info_to_method
from quasar_utils.absorption import remove_absorption, smooth_spectrum
from quasar_utils.continuum_fit_result import ContinuumFitResult

from quasar_typing.numpy import FloatVector, BoolVector, RandomState_
from quasar_typing.bounds import CoordBounds, AstropyBounds
from quasar_typing.pandas import LineList
from quasar_typing.pathlib import (
    AbsoluteFilePath, AbsoluteDirPath, RelativeFilePath, AnyFITSPath, AnyFilePath
)
from quasar_typing.astropy import FitterInstance, Model_, FitInfo, QTable_
from quasar_typing.misc.pool import Pool_
from quasar_typing.misc.coords_tuple import CoordsTuple
from quasar_typing.misc.literals import (
    BGFlux, Scale, Variant, BootstrapType, VaryLines,
)

from quasar_models.utils.astropy import get_model_parts, get_free_params

from .utils import _SpecData, SpecList, get_log
from .continuum import ContinuumWindows
from .iron import IronWindows
from .balmer import BalmerWindows
from .lines import LineWindows

logger = getLogger(__name__)
logger.disabled = not getLogger().hasHandlers()

@dataclass
class Spectrum(_SpecData):
    path: AbsoluteFilePath
    title: str

    _is_preprocessed: bool = field(default=False, init=False)

    continuum_windows: ContinuumWindows | None = field(default=None, init=False)
    iron_windows: IronWindows | None = field(default=None, init=False)
    balmer_windows: BalmerWindows | None = field(default=None, init=False)
    host_windows: None = field(default=None, init=False)
    line_windows: LineWindows | None = field(default=None, init=False)

    bootstrapper: BaseBootstrapper | None = field(default=None, init=False)
    error_result: ErrorResult | None = field(default=None, init=False)

    @property
    def pl(self) -> ContinuumWindows | None: return self.continuum_windows

    @property
    def fe(self) -> IronWindows | None: return self.iron_windows

    @property
    def ba(self) -> BalmerWindows | None: return self.balmer_windows

    @property
    def hg(self) -> None: return self.host_windows

    @property
    def em(self) -> LineWindows | None: return self.line_windows
    

    @validate_call(validate_return=False)
    def __init__(
        self,
        path: AbsoluteFilePath,
        title: str,
        coords: CoordsTuple,
        info: Info = None,
    ):
        self.path = path
        self.title = title
        super().__init__(coords, info=info)

    def __post_init__(self):
        self.preprocess()

    def __str__(self, simple: bool = False):

        s = "'Spectrum' class [{:.1f} <-> {:.1f}]".format(*self.x_bounds)

        if not simple:
            i = int((~self.valid_pixels).sum())
            n = len(self.x)

            s += " w/ {}/{} (invalid).".format(i, n)
        
        return s
        
    @validate_call(validate_return=False)
    @apply_info_to_method('absorption', 'lines')
    def __call__(
        self,
        linelist: AbsoluteFilePath | LineList,
        *,
        refine: bool = False,
        x_limit: float | None = None,
    ) -> None:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        logger.debug("Running 'Spectrum' pipeline...")
        
        self.cropLymanAlphaForest.__wrapped__.raw(
            self, x_limit=x_limit,
        )
        self.fitContinuum.__wrapped__.raw(self)

        self.fitLines(linelist)
        self.finaliseFit()
        self.reformatFit()

        if refine:
            logger.debug("Refitting.")
            self.fitLines(linelist)
            self.finaliseFit()
            self.reformatFit()

        logger.debug("... finished running 'Spectrum' pipeline!")

    def __getitem__(
        self,
        key: Literal['pl', 'fe', 'ba', 'hg', 'em'],
    ) -> SpecList | None:
        match key:
            case 'pl':
                return self.continuum_windows
            case 'fe':
                return self.iron_windows
            case 'ba':
                return self.balmer_windows
            case 'hg':
                return self.host_windows
            case 'em':
                return self.line_windows

    def preprocess(self) -> None:
        if not self._is_preprocessed:
            logger.debug(f"Preprocessing: {self}")
            self.findAnomalies()
            self.truncateSpectrum()
            logger.debug(f"Preprocessed: {self}")
            
            self._is_preprocessed = True

    def findAnomalies(self) -> None:
        """
        Identifies anomalous pixels, and replaces them with NaN. Anomalous 
        pixels have: non-positive flux densities OR non-positive flux density
        uncertainties.
        """
        n = self._x.size
        cond_1 = invert(isfinite(self._x))
        cond_2 = invert(isfinite(self._y))
        cond_3 = invert(isfinite(self._dy)) | (self._dy <= 0)

        mask = cond_1 | cond_2 | cond_3

        msg = ">>> Found {}/{} anomalous pixels: ".format(mask.sum(), n) \
            + "{}/{} (invalid wave), ".format(cond_1.sum(), n) \
            + "{}/{} (invalid flux), ".format(cond_2.sum(), n) \
            + "{}/{} (invalid flux unc.).".format(cond_3.sum(), n)
        logger.debug(msg)

        self._valid_pixels = invert(mask)

    def truncateSpectrum(self) -> None:
        """
        Truncates the spectrum if edges contain NaN values.
        """
        mask = self._valid_pixels
        msg = ">>> "

        if mask[0] and mask[-1]:
            msg += "No invalid pixels at either edge. "
        else:
            valid_indices = argwhere(mask).flatten()
            idx_start = valid_indices[0]
            idx_end = valid_indices[-1]
            sel = slice(valid_indices[0], valid_indices[-1]+1)

            n = len(mask)
            n_tot = (n_blue := idx_start) + (n_red := n - 1 - idx_end)

            self.__init__.__wrapped__(
                self,
                self.path,
                self.title,
                (self._x[sel], self._y[sel], self._dy[sel]),
                info = self.info,
            )

            msg += "Cutting {}/{} pixel(s): ".format(n_tot, n) \
                + "{}/{} (blue edge), ".format(n_blue, n) \
                + "{}/{} (red edge).".format(n_red, n)
            
        logger.debug(msg)
            
    @validate_call(validate_return=False)
    @apply_info_to_method('lines')
    def cropLymanAlphaForest(
        self, 
        *,
        x_limit: float | None = None,
    ) -> None:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        logger.debug(f"Cropping Lyman-alpha forest: {self}")

        indices = argwhere(self._x < x_limit).flatten()
        if (len(indices) != 0) and ((edge_idx := indices[-1]) >= 1):
            _y = self._y[:edge_idx+1]
            idx_max = nanargmax(_y).flatten()[0]

            x_max = self._x[idx_max]
            y_max = self._y[idx_max]

            logger.debug(
                ">>> Peak found at ({:.1f}, {:.1f}) ".format(x_max, y_max) \
                + "under the condition: x < {:.1f}.".format(self._x[edge_idx])
            )

            self.x_bounds = (
                self._x[idx_max] - self._dx[idx_max] / 2,
                self.x_bounds[1]
            )

            logger.debug(f">>> Cropping {idx_max} pixel(s).")

        logger.debug(f"Cropped Lyman-alpha forest: {self}")

    @validate_call(validate_return=False)
    @apply_info_to_method('absorption')
    def smoothSpectrum(
        self,
        *,
        w: int | None = None,
        p: int | None = None,
        logspace: int | None = None,
    ) -> None:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self._y_smooth[:] = smooth_spectrum.__wrapped__(
            *self._coords,
            self._valid_pixels,
            w, 
            p,
            logspace,
        )

    @validate_call(validate_return=False)
    @apply_info_to_method('absorption')
    def removeAbsorption(
        self,
        *,
        bg_flux: BGFlux | None = None,
        w: int | None = None,
        p: int | None = None,
        p_crit: float | None = None,
        z_crit: float | None = None,
        join: int | None = None,
        refine: bool | None = None,
        logspace: bool | None = None,
    ) -> None:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if array_equal(self._y, self._y_smooth):
            self.smoothSpectrum.__wrapped__.raw(
                self, w=w, p=p, logspace=logspace,
            )
        if bg_flux is None: 
            bg_flux = {'pl', 'fe', 'ba', 'hg', 'em'}

        result = remove_absorption.__wrapped__(
            *self._coords,
            self._y_smooth,
            sum(getattr(self, f"_y_{bg}") for bg in bg_flux),
            self._valid_pixels,
            w,
            p_crit,
            z_crit,
            join,
            refine,
        )
        self._p_absorbed[:] = result[0]
        self._absorbed_pixels[:] = result[1]
        self._y_smooth[:] = result[2]
        
    @validate_call(validate_return=False)
    @apply_info_to_method('continuum')
    def instantiateContinuum(
        self,
        *,
        windows: list[CoordBounds] | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **

        Instantiates the Spectrum instance's 'continuum_windows' attribute.
        """
        self.continuum_windows = ContinuumWindows(self, windows=windows)
        return self

    @validate_call(validate_return=False)
    @apply_info_to_method('iron', specific_kwargs={'windows'})
    def instantiateIron(
        self,
        *,
        windows: list[CoordBounds] | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self.iron_windows = IronWindows(self, windows=windows)
        return self
    
    @validate_call(validate_return=False)
    @apply_info_to_method('balmer', specific_kwargs={'windows'})
    def instantiateBalmer(
        self,
        *,
        windows: list[CoordBounds] | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self.balmer_windows = BalmerWindows(self, windows=windows)
        return self

    @validate_call(validate_return=False)
    def instantiateLines(self) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self.line_windows = LineWindows(self)
        return self

    @validate_call(validate_return=False)
    @apply_info_to_method('continuum', 'nonlinear')
    def fitContinuum(
        self,
        *,
        windows: list[CoordBounds] | None = None,
        sigmas: list[float] | None = None,
        flux_bounds: AstropyBounds | None = None,
        alpha_bounds: AstropyBounds | None = None,
        fitter: FitterInstance | None = None,
    ) -> None:
        """
        ** PYDANTIC VALIDATED METHOD **

        Fits the continuum and creates a 'CSSpectrum' class. 
        Takes the following steps:

        1.  Creates a 'ContinuumWindows' class, makes a preliminary power law 
            fit (using linear methods), and refines it using sigma-clipping and 
            non-linear optimisation. 
        2.  Using the continuum fit, creates a 'CSSpectrum'
            class. 

        Parameters
        ----------
        sigmas : list[float], None
            List of critical residual values used for sigma-clipping. If None, 
            defaults to the values given by the 'ContinuumInfo' class.
        fitter : Callable, None
            Function which performs non-linear optimisation through 'astropy'. 
            If None, defaults to the function given by the 'NonLinearInfo' class
            via the 'ContinuumInfo' class.  

        Notes
        -----
        The instantiated classes are accessed using (self is the 'Spectrum' 
        class):
        
            ContinuumWindows ->             self.continuum_windows
            CSSpectrum ->  self.cs_spectrum
        """
        self.instantiateContinuum.__wrapped__.raw(
            self,
            windows=windows,
        )
        self.continuum_windows.__call__.__wrapped__.raw(
            self.continuum_windows,
            sigmas=sigmas,
            flux_bounds=flux_bounds,
            alpha_bounds=alpha_bounds,
            fitter=fitter,
        )

    @validate_call(validate_return=False)
    @apply_info_to_method('iron', 'nonlinear')
    def fitIron(
        self,
        without_rejections: bool = False,
        without_absorption: bool = True,
        *,
        windows: list[CoordBounds] | None = None,
        template_files: list[AnyFITSPath] | None = None, 
        resample: bool | None = None,
        flux_bounds: AstropyBounds | None = None,
        fwhm_bounds: AstropyBounds | None = None,
        split: FloatVector | None = None,
        fwhm: FloatVector | None = None,
        bias: list[str] | None = None,
        ratio: FloatVector | None = None,
        scale: float | None = None,
        allow_interp_fitting: bool | None = None,
        raster: bool | None = None,
        fine_tune: bool | None = None,
        fitter: FitterInstance | None = None,
    ) -> None:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self.instantiateIron.__wrapped__.raw(
            self, 
            windows=windows,
        )
        self.iron_windows.__call__.__wrapped__.raw(
            self.iron_windows,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
            template_files = template_files,
            resample = resample,
            flux_bounds = flux_bounds,
            fwhm_bounds = fwhm_bounds,
            split = split,
            fwhm = fwhm,
            bias = bias,
            ratio = ratio,
            scale = scale,
            allow_interp_fitting = allow_interp_fitting,
            raster = raster,
            fine_tune = fine_tune,
            fitter = fitter,
        )
        
    @validate_call(validate_return=False)
    @apply_info_to_method('balmer', 'nonlinear')
    def fitBalmer(
        self,
        without_rejections: bool = False,
        without_absorption: bool = True,
        *,
        windows: list[CoordBounds] | None = None,
        template_files: list[AnyFilePath] | None = None, 
        resample: bool | None = None,
        flux_bounds: AstropyBounds | None = None,
        split: FloatVector | None = None,
        fwhm: FloatVector | None = None,
        bias: list[str] | None = None,
        ratio: FloatVector | None = None,
        scale: float | None = None,
        fine_tune: bool | None = None,
        fitter: FitterInstance | None = None,
    ) -> None:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self.instantiateBalmer.__wrapped__.raw(
            self, 
            windows=windows,
        )
        self.balmer_windows.__call__.__wrapped__.raw(
            self.balmer_windows,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
            template_files = template_files,
            resample = resample,
            flux_bounds = flux_bounds,
            split = split,
            fwhm = fwhm,
            bias = bias,
            ratio = ratio,
            scale = scale,
            fine_tune = fine_tune,
            fitter = fitter,
        )

    @validate_call(validate_return=False)
    @apply_info_to_method('loading', 'lines', 'nonlinear')
    def fitLines(
        self,
        linelist: AbsoluteFilePath | LineList,
        *,
        bg_flux: BGFlux = {'pl', 'fe', 'ba', 'hg'},
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        limited: bool = False,

        sigma_res: float | None = None,
        v_sep: float | None = None,
        forced_splits: FloatVector | None = None,
        w: int | None = None,
        min_fittable_total: int | None = None,
        min_fittable_ratio: float | None = None,
        evaluate_initial: float | int | None = None,
        aggressive: bool | None = None,
        crop: bool | None = None,
        measure: str | None = None,
        reverse: bool | None = None,
        make_copies: bool | None = None,

        fitter: FitterInstance | None = None,
    ) -> None:
        """
        ** PYDANTIC VALIDATED METHOD **

        Calls the 'CSSpectrum' class, performing the following:

        1.  Smooths the flux density array, and identifies and removes pixels
            possibly affected by narrow absorption lines. 
        2.  Creates 'SpecSlice' classes between 'Window' classes (used for 
            continuum fitting). 
        3.  Creates 'SubSlice' classes from 'SpecSlice' classes using the given 
            list of emission lines. If a 'SpecSlice' is empty (i.e. doesn't
            contain any emission lines, it is discarded). 
        4.  Fits emission lines within each 'SubSlice' instance, increasing
            model complexity accordingly.  

        Parameters
        ----------
        See '__call__()' method of the 'CSSpectrum' class.
        """
        self.instantiateLines()
        self.line_windows.__call__.__wrapped__.raw(
            self.line_windows,
            linelist,
            bg_flux = bg_flux,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
            with_neighbours = with_neighbours,
            limited = limited,

            v_sep = v_sep,
            forced_splits = forced_splits,

            w = w,
            sigma_res = sigma_res,
            min_fittable_total = min_fittable_total,
            min_fittable_ratio = min_fittable_ratio,
            evaluate_initial = evaluate_initial,
            aggressive = aggressive,
            crop = crop,
            measure = measure,
            reverse = reverse,
            make_copies = make_copies,
            
            fitter = fitter,
        )

    @validate_call(validate_return=False)
    @apply_info_to_method('nonlinear')
    def finaliseFit(
        self,
        model_types: BGFlux | None = None,
        data_types: BGFlux | None = None,
        bg_flux: BGFlux | None = None,
        without_rejections: bool = False, 
        without_absorption: bool = False,
        *,
        fitter: FitterInstance | None = None,
    ) -> bool:
        from .utils.general import stopwatch

        msg = "Finalising fit for {}: ".format(
            self.__str__(simple=True).removesuffix('.'),
        )
        if (model_types is None) and (data_types is None):
            model_types = {'pl', 'fe', 'ba', 'hg', 'em'}
            data_types = {'pl', 'fe', 'ba', 'hg', 'em'}

        elif (model_types is None):
            model_types = data_types.copy()

        elif (data_types is None):
            data_types = model_types.copy()
        
        if bg_flux is None:
            bg_flux = {'pl', 'fe', 'ba', 'hg', 'em'}

        for d_type in data_types.copy():
            if self[d_type] is not None:
                continue

            data_types.remove(d_type)

        if not bool(data_types):
            logger.warning(msg + "No data to fit: cancelling!")
            return False
        
        bg_flux.difference_update(model_types)
        msg += f"{model_types=}, {data_types=}, {bg_flux=}"

        coords = self.getMaskedCoords.__wrapped__(
            self,
            data_types = data_types,
            bg_flux = bg_flux,
            covered = True,
            log = False,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
            valid = True,
        )
        n_data = len(coords[0])
        msg += f"{n_data=}, "

        model = self.getModel.__wrapped__(
            self,
            model_types = model_types,
        )
        n_submodels = 0 if (model is None) else model.n_submodels
        msg += f"{n_submodels=}. "

        if model is None:
            logger.warning(msg + "No models to fit: cancelling!")
            return False

        try:
            with stopwatch() as watch:
                fit, fit_info = fitter(
                    model,
                    *coords,
                    inplace = False,
                )
        except ValidationError as e: 
            logger.warning(
                msg + f"Failed fitting due to validation error: {e}",
            )
            return False
        
        logger.debug(
            msg + f"Finished fitting in {watch.elapsed:.1f} ms."
        )
        self.applyFit(fit, fit_info)

        return True

    def reformatFit(self):
        """
        Updates each 'Subslice' class' submodel names, such that they are in 
        order (sorted by observed wavelength, in ascending order). For emission
        lines with multiple assigned submodels, the submodels are numbered in 
        the appropriate order (#1, #2, #3, etc.). 
        """
        self.fit = self.fit[0]
        for lwindow in self.line_windows:
            lwindow.reformatFit()
            self.fit += lwindow.fit

    @validate_call(validate_return=False)
    def getMask(
        self,
        *,
        data_types: BGFlux | None = None,
        covered: bool = True,
        without_rejections: bool = False,
        without_absorption: bool = False,
        valid: bool = False,
        log_valid: bool = False,
    ) -> BoolVector:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        mask = zeros_like(self._x, dtype=bool)
        if (data_types is None) or (not covered):
            mask[:] = True
        else:
            for d_type in data_types:
                windows = self[d_type]
                if windows is None:
                    continue

                mask |= windows.mask
            
        if without_rejections: 
            mask &= invert(self._rejected_pixels)
        if without_absorption:
            mask &= invert(self._absorbed_pixels)

        if log_valid:
            mask &= self._log_valid_pixels
        elif valid:
            mask &= self._valid_pixels

        return mask
    
    @validate_call(validate_return=False)
    def getMaskedCoords(
        self,
        *,
        data_types: BGFlux | None = None,
        bg_flux: BGFlux = {},
        covered: bool = True,
        log: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        valid: bool = False,
        log_valid: bool = False,
    ) -> CoordsTuple:
        """
        ** PYDANTIC VALIDATED METHOD **
        """        
        x = self._x_log if log else self._x
        dy = self._dy_log if log else self._dy

        y = self._y.copy()
        for bg in bg_flux:
            y -= getattr(self, f"_y_{bg}")

        if log: 
            y = get_log(y, self.y0, self._log_valid_pixels)

        mask = self.getMask.__wrapped__(
            self,
            data_types = data_types,
            covered = covered,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
            valid = valid,
            log_valid = log_valid,
        )
        return x[mask], y[mask], dy[mask]
    
    @validate_call(validate_return=False)
    def getModel(
        self,
        model_types: BGFlux = {'pl', 'fe', 'ba', 'hg', 'em'},
    ) -> Optional[Model_]:
        """
        ** PYDANTIC VALIDATED METHOD **
        """        
        models: list[Model_] = []
        for m_type in model_types:
            windows = self[m_type]
            if windows is None:
                continue

            model = windows.getModel()
            if model is not None:
                models.append(model)

        return None if len(models) == 0 else sum(models[1:], start=models[0])
    
    @validate_call(validate_return=False)
    def applyFit(
        self,
        fit: Model_,
        fit_info: FitInfo,
    ) -> None:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        count = 0
        parts = get_model_parts(fit)
        for key in ['pl', 'fe', 'ba', 'hg', 'em']:
            model = parts[key]
            if model is None:
                continue

            n_free = sum(get_free_params(model).values())
            sel = slice(count, count+n_free)
            finfo = OptimizeResult(
                message =    fit_info.message,
                success =    fit_info.success,
                status =     fit_info.status,
                fun =        fit_info.fun,
                x =          fit_info.x[sel],
                cost =       fit_info.cost,
                jac =        fit_info.jac[sel,sel],
                grad =       fit_info.grad[sel],
                optimality = fit_info.optimality,
                nfev =       fit_info.nfev,
                njev =       fit_info.njev,
                param_cov =  fit_info.param_cov[sel,sel]
            )
            match key:
                case 'pl':
                    method = 'updateContinuumEmission'
                case 'fe':
                    method = 'updateIronEmission'
                case 'ba':
                    method = 'updateBalmerEmission'
                case 'hg':
                    method = 'updateHostEmission'
                case 'em': 
                    method = 'updateLinesEmission'

            windows = self[key]
            windows.applyFit.__wrapped__(windows, model, finfo)
            getattr(self, method).__wrapped__(self, model)

            count += n_free
    
    def summariseContinuumFit(self) -> ContinuumFitResult:
        try:
            return self.continuum_windows.summariseContinuumFit()
        except Exception as _:
            return None
    
    @validate_call(validate_return=False)
    @apply_info_to_method('error', 'nonlinear')
    def instantiateBootstrapper(
        self,
        *,
        covered: bool = True,
        data_types: BGFlux | None = None,
        model_types: BGFlux | None = None,

        without_rejections: bool = False,
        without_absorption: bool = False,

        pool: Pool_ | None = None,

        scale: Scale | None = None,
        variant: Variant | None = None,
        bootstrap_type: BootstrapType | None = None,
        n_sigmas: float | None = None,
        vary_lines: VaryLines | None = None,

        iterations: int | None = None,
        random_state: RandomState_ | None = None,
        renew_rng: bool | None = None,
        replace_missing: bool | None = None,
        tqdm_disable: bool | None = None,

        fitter: FitterInstance | None = None,
    ) -> BaseBootstrapper:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if scale == 'semilocal' or scale == 'local':
            if self['em'] is None:
                msg = f"Cannot perform bootstrapping on '{scale}' scale \
                    without emission line windows!"
                logger.warning(msg)
                raise ValueError(msg)
            
            for lwindow in self.line_windows:
                lwindow.instantiateBootstrapper.__wrapped__.raw(
                    lwindow,
                    pl=self['pl'].getModel(),
                    fe=self['fe'].getModel(),
                    ba=self['ba'].getModel(),
                    model_type=model_types,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                    pool=pool,
                    variant=variant,
                    bootstrap_type=bootstrap_type,
                    n_sigmas=n_sigmas,
                    vary_lines=vary_lines,
                    cfit=self['pl'].summariseContinuumFit(),
                    iterations=iterations,
                    random_state=random_state,
                    renew_rng=renew_rng,
                    replace_missing=replace_missing,
                    tqdm_disable=tqdm_disable,
                    fitter=fitter,
                    logger=logger,
                )

        if self.bootstrapper is not None:
            logger.debug("Existing 'bootstrapper' will be overwritten.")

        cls, args, kwargs = format_bootstrapping_kwargs_for_spectrum(
            self,
            fitter,
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            variant=variant,
            data_types=data_types,
            model_types=model_types,
            bootstrap_type=bootstrap_type,
            n_sigmas=n_sigmas,
            vary_lines=vary_lines,
            pool=pool,
            iterations=iterations,
            random_state=random_state,
            renew_rng=renew_rng,
            replace_missing=replace_missing,
            tqdm_disable=tqdm_disable,
            logger=logger,
        )
        self.bootstrapper = cls(*args, **kwargs)
        return self.bootstrapper

    @validate_call(validate_return=False)
    @apply_info_to_method('error', specific_kwargs={'scale'})
    def runBootstrapper(
        self,
        *,
        scale: Scale | None = None,
    ) -> ErrorResult:
        if scale == 'semilocal' or scale == 'local':
            pass

        if self.bootstrapper is None:
            msg = "No bootstrapper instance found! \
                Run 'instantiateBootstrapper()' method first."
            logger.critical(msg)
            raise AttributeError(msg)
        if self.error_result is not None:
            logger.debug("Existing 'error_result' will be overwritten.")

        out = self.bootstrapper.run()
        self.error_result = self.bootstrapper.toErrorResult(out)
        return self.error_result
    
    @validate_call(validate_return=False)
    @apply_info_to_method('error')
    def analyseBootstrapSamples(
        self,
        directory: AbsoluteDirPath | None = None,
        paths: Iterable[RelativeFilePath] | None = None,
        *,
        out_waves: set[str | float] | None = None,
        out_measures: set[str] | None = None,
        percentiles: set[int] | None = None,
        res: int | None = None,
        render_width: float | None = None,
        fwhm_strategy: str | None = None,
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
        raise NotImplementedError
        # from .complexes.analysis import analyse_error_samples


        # kwargs = dict(
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
        # if not hasattr(self, 'bootstrap_result'):
        #     out: dict[float, QTable] = {}
        #     for lwindow in self.line_windows:
        #         result = lwindow.analyseBootstrapSamples.__wrapped__.raw(
        #             lwindow, **kwargs,
        #         )
        #         if isinstance(result, dict):
        #             out.update(result)

        #     return None if (len(out) == 0) else out

        # return analyse_error_samples.__wrapped__(
        #     self.bootstrap_result,
        #     self.info,
        #     **kwargs,
        # )
    
    # @validate_call
    # def _get_all_coords(self) -> tuple[
    #     FloatVector, FloatVector, FloatVector, FloatVector,
    #     BoolVector, BoolVector, BoolVector,
    # ]:
    #     from numpy import zeros_like
    #     x = self.x
    #     y = self.y
    #     dy = self.dy
    #     valid_pixels = self.valid_pixels

    #     y_smooth = self.y \
    #         if not hasattr(self, 'line_windows') \
    #         else (
    #         self.cs_spectrum.y_smooth \
    #         + self.cs_spectrum.continuum_fit(x)
    #     )
    #     rejected_pixels = zeros_like(x, dtype=bool) \
    #         if not hasattr(self, 'continuum_windows') \
    #         else self.continuum_windows._rejected_pixels
        
    #     absorbed_pixels = zeros_like(x, dtype=bool) \
    #         if not hasattr(self, 'cs_spectrum') \
    #         else self.cs_spectrum.absorbed_pixels

    #     return (
    #         x, y, dy, y_smooth, \
    #         rejected_pixels, absorbed_pixels, valid_pixels,
    #     )
    
    # def _get_masks(self) -> tuple[BoolVector, BoolVector]:
    #     from numpy import zeros_like

    #     in_windows = zeros_like(self.x, dtype=bool)
    #     in_subslices = in_windows.copy()

    #     if hasattr(self, 'continuum_windows'):
    #         in_windows = self.continuum_windows.mask

    #     if hasattr(self, 'cs_spectrum'):
    #         for subslice in self.cs_spectrum.getSubslices():
    #             lb, ub = subslice.x_bounds
    #             in_subslices[(lb <= self.x) & (self.x <= ub)] = True

    #     return in_windows, in_subslices
    
    @validate_call
    @apply_info_to_method('nonlinear', 'lines')
    def _refresh(
        self,
        refresh_absorption: bool = True,
        refresh_fit: bool = True,
        *,
        w: int | None = None,
        p: int | None = None,
        p_crit: float | None = None,
        z_crit: float | None = None,
        join: int | None = None,
        refine: bool | None = None,
        logspace: bool | None = None,
        fitter: FitterInstance | None = None,
    ) -> None:
        return 
        _x = self.x
        _y = self.y - self.fit[0](self.x)
        _dy = self.dy

        self.continuum_windows.fit = self.fit[0]
        self.cs_spectrum._refresh.__wrapped__(
            self.cs_spectrum,
            x = _x,
            y = _y,
            dy = _dy,
            refresh_absorption = refresh_absorption,
            refresh_fit = refresh_fit,
            w = w,
            p = p,
            p_crit = p_crit,
            z_crit = z_crit,
            join = join,
            refine = refine,
            logspace = logspace,
            fitter = fitter,
        )

    def _quickplot(
        self,
        fig: tuple[Figure, Axes] | None = None,
        *,
        figsize: tuple[float, float] = (8, 6),
        dpi: int = 300,

        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,

        x_major: int = 500,
        x_minor: int = 20,
        y_major: int = 50,
        y_minor: int = 5,

        pl_color: str = 'dodgerblue',
        fe_color: str = 'firebrick',
        ba_color: str = 'orchid',
        hg_color: str = 'darkgreen',
        em_color: str = 'gold',
        
        sm_color: str = 'darkorange',
        ab_color: str = 'darkviolet',

        logx: bool = True,
        logy: bool = True,
    ) -> None:
        """
        Basic plotting routing for 'Spectrum' classes.

        NOTES
        -----
        This method overwrites but still calls the inherited '_quickplot'
        method, setting 'title' equal to 'self.title' by default.
        """
        return super()._quickplot(
            fig,
            figsize = figsize,
            dpi = dpi,
            title = self.title,
            xlim = xlim,
            ylim = ylim,
            x_major = x_major,
            x_minor = x_minor,
            y_major = y_major,
            y_minor = y_minor,
            pl_color = pl_color,
            fe_color = fe_color,
            ba_color = ba_color,
            hg_color = hg_color,
            em_color = em_color,
            sm_color = sm_color,
            ab_color = ab_color,
            logx = logx,
            logy = logy,
        )