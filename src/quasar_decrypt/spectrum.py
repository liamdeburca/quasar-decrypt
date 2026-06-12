from logging import getLogger
from typing import Iterable, Literal, Self, Callable
from pydantic.dataclasses import dataclass
from dataclasses import field
from numpy import isfinite, invert, argwhere, nanargmax, zeros_like, array_equal
from scipy.optimize import OptimizeResult

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from pydantic_core import ValidationError

from quasar_errors.bootstrapping import BaseBootstrapper
from quasar_errors.nested_sampling import BaseNestedSampler, UniformNestedSampler
from quasar_errors.spectrum_utils import (
    format_bootstrapping_kwargs_for_spectrum,
)
from quasar_errors.error_result import ErrorResult
from quasar_errors.model_samples import (
    PowerLawSample, IronSampleList, BalmerSample, GaussianSampleList, Sample,
)

from quasar_utils.setup import Info
from quasar_utils.fitting import FitterInstance
from quasar_utils.decorators import validate_call, validated_apply_info_to_method
from quasar_utils.absorption import remove_absorption, smooth_spectrum
from quasar_utils.continuum_fit_result import ContinuumFitResult
from quasar_utils.pipeline.linelist import LineList

from quasar_typing.numpy import FloatVector, BoolVector, RandomState_
from quasar_typing.bounds import CoordBounds, AstropyBounds

from quasar_typing.pathlib import (
    AbsoluteFilePath, AbsoluteDirPath, RelativeFilePath, AbsoluteFITSPath,
)
from quasar_typing.astropy import Model_, FitInfo, QTable_, CompoundModel_
from quasar_typing.misc import (
    Pool_, 
    BackgroundFlux, ModelTypes, DataTypes,
    Scale, Variant, BootstrapType, VaryLines,
)

from quasar_models import (
    PowerLawModel, 
    IronModel,
    BalmerModel,
    # HostGalaxyModel,
    GaussianModel,
)
from quasar_models.utils.astropy import get_model_parts, get_free_params

from quasar_plotting import quickplot, absorptionplot, fitplot
from quasar_plotting.utils import get_coords
from quasar_plotting.colors import DEFAULT_COLORS

from .utils.masked_coords import MaskedCoords, ReadOnlyMaskedCoords, ContiguousMaskedCoords

from .utils import _SpecData, SpecList
from .continuum import ContinuumWindows
from .iron import IronWindows
from .balmer import BalmerWindows
from .lines import LineWindows

logger = getLogger(__name__)

@dataclass
class Spectrum(_SpecData):
    path: AbsoluteFilePath = field(kw_only=True)
    title: str = field(kw_only=True)

    _is_preprocessed: bool = field(default=False, init=False)

    continuum_windows: ContinuumWindows | None = field(default=None, init=False)
    iron_windows: IronWindows | None = field(default=None, init=False)
    balmer_windows: BalmerWindows | None = field(default=None, init=False)
    host_windows: None = field(default=None, init=False)
    line_windows: LineWindows | None = field(default=None, init=False)

    bootstrapper: BaseBootstrapper | None = field(default=None, init=False)
    nested_sampler: BaseNestedSampler | None = field(default=None, init=False)
    error_result: ErrorResult | None = field(default=None, init=False)

    @classmethod
    @validate_call
    def create(
        cls,
        *,
        path: AbsoluteFilePath,
        title: str,
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

        x_bounds: CoordBounds | None = None,
        info: Info = None,
        get_mask: Callable[[float, float], BoolVector] | None = None,
    ) -> Self:
        kwargs = cls.get_kwargs.__wrapped__(
            cls,
            path=path,
            title=title,
            x=x,
            y=y,
            dy=dy,
            dx=dx,
            
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

            x_bounds=x_bounds,
            info=info,
            get_mask=get_mask,
        )
        return cls(**kwargs)

    @classmethod
    @validate_call
    def get_kwargs(
        cls,
        *,
        path: AbsoluteFilePath,
        title: str,
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

        x_bounds: CoordBounds | None = None,
        info: Info = None,
        get_mask: Callable[[float, float], BoolVector] | None = None,
    ) -> dict:
        kwargs = {'path': path, 'title': title}
        kwargs.update(super().get_kwargs.__wrapped__(
            cls,
            x=x,
            y=y,
            dy=dy,
            dx=dx,

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

            x_bounds=x_bounds,
            info=info,
            get_mask=get_mask,
        ))
        return kwargs
    
    def __post_init__(self) -> None:
        self.preprocess()

    @property
    def pl(self) -> ContinuumWindows | None: return self.continuum_windows

    @property
    def pl_sample(self) -> PowerLawSample | None: 
        return None if self.pl is None else self.pl.sample

    @property
    def fe(self) -> IronWindows | None: return self.iron_windows

    @property
    def fe_sample(self) -> IronSampleList | None:
        return None if self.fe is None else self.fe.sample
    
    @property
    def ba(self) -> BalmerWindows | None: return self.balmer_windows

    @property
    def ba_sample(self) -> BalmerSample | None:
        return None if self.ba is None else self.ba.sample

    @property
    def hg(self) -> None: return self.host_windows

    @property
    def hg_sample(self) -> None:
        return None

    @property
    def em(self) -> LineWindows | None: return self.line_windows

    @property
    def em_sample(self) -> GaussianSampleList | None:
        return None if self.em is None else self.em.sample
    
    @property
    def basic_sample(self) -> Sample:
        if (model := self.getModel()) is None:
            return Sample.fromSpectrumSample(self.x, self.y, self.dy)
        return Sample.fromModelSample(self.x, self.y, self.dy, model)
    
    @property
    def basic_error_result(self) -> ErrorResult:
        return ErrorResult((self.basic_sample,), (1.0,))

    def __str__(self, simple: bool = False):
        s = "'Spectrum' class [{:.1f} <-> {:.1f}]".format(*self.x_bounds)

        if not simple:
            i = int((~self.valid_pixels).sum())
            n = len(self.x)

            s += " w/ {}/{} (invalid).".format(i, n)
        
        return s
    
    @validated_apply_info_to_method(subjects=('absorption', 'lines'))
    def __call__(
        self,
        linelist: AbsoluteFilePath | LineList,
        *,
        refine: bool = False,
        x_limit: float | None = None,
    ) -> None:
        logger.debug("Running 'Spectrum' pipeline...")
        
        self.cropLymanAlphaForest.__wrapped__(
            self, x_limit=x_limit,
        )
        self.fitContinuum.__wrapped__(self)

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

    def _truncate_range(self, sel: slice) -> None:
        if sel.step not in (None, 1):
            raise ValueError("Only contiguous truncation is supported.")

        def f(arr): return arr[sel]

        for key, value in super().get_kwargs.__wrapped__(
            self.__class__,
            x=f(self._x),
            y=f(self._y),
            dy=f(self._dy),
            dx=f(self._dx),
            x_bounds=None,
            info=self.info,
            y_smooth=f(self._y_smooth),
            y_pl=f(self._y_pl),
            y_fe=f(self._y_fe),
            y_ba=f(self._y_ba),
            y_hg=f(self._y_hg),
            y_em=f(self._y_em),
            rejected_pixels=f(self._rejected_pixels),
            absorbed_pixels=f(self._absorbed_pixels),
            valid_pixels=f(self._valid_pixels),
            log_valid_pixels=f(self._log_valid_pixels),
            p_absorbed=f(self._p_absorbed),
            x0=self.x0,
            y0=self.y0,
            x_log=f(self._x_log),
            y_log=f(self._y_log),
            dy_log=f(self._dy_log),
            get_mask=None,
        ).items():
            setattr(self, key, value)

    def truncateSpectrum(self) -> None:
        """
        Truncates the spectrum if edges contain NaN values.
        """
        mask = self._valid_pixels

        n_valid = self._valid_pixels.sum()
        if n_valid == 0:
            logger.warning("All pixels are invalid! Cannot truncate spectrum.")
        elif n_valid == 1:
            logger.critical("Only one valid pixel! Cannot truncate spectrum.")
        elif self._valid_pixels[0] and self._valid_pixels[-1]:
            logger.debug("No invalid pixels at either edge")
        else:
            valid_indices = argwhere(mask).flatten()
            idx_start = valid_indices[0]
            idx_end = valid_indices[-1]
            sel = slice(valid_indices[0], valid_indices[-1]+1)

            n = len(mask)
            n_blue = idx_start
            n_red = n - 1 - idx_end
            n_tot = n_blue + n_red

            self._truncate_range(sel)

            logger.debug(
                "Cutting %d/%d pixel(s): %d/%d (blue edge), %d/%d (red edge)",
                n_tot, n, n_blue, n, n_red, n,
            )
                        
    @validated_apply_info_to_method(subjects=('lines',))
    def cropLymanAlphaForest(
        self, 
        *,
        x_limit: float | None = None,
    ) -> None:
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

            self._valid_pixels[:idx_max] = False
            self.truncateSpectrum()

            # self.x_bounds = (
            #     self._x[idx_max] - self._dx[idx_max] / 2,
            #     self.x_bounds[1]
            # )

            logger.debug(f">>> Cropping {idx_max} pixel(s).")

        logger.debug(f"Cropped Lyman-alpha forest: {self}")

    @validated_apply_info_to_method(subjects=('absorption',))
    def smoothSpectrum(
        self,
        *,
        w: int | None = None,
        p: int | None = None,
        logspace: int | None = None,
    ) -> None:
        """
        ** PYDANTIC VALIDATED AND INFO-APPLIED METHOD **
        """
        self._y_smooth[:] = smooth_spectrum.__wrapped__(
            *self._coords,
            self._valid_pixels,
            w, 
            p,
            logspace,
        )

    @validated_apply_info_to_method(subjects=('absorption',))
    def removeAbsorption(
        self,
        *,
        bg_flux: BackgroundFlux = BackgroundFlux({'all'}),
        w: int | None = None,
        p: int | None = None,
        p_crit: float | None = None,
        z_crit: float | None = None,
        join: int | None = None,
        refine: bool | None = None,
        logspace: bool | None = None,
    ) -> None:
        if array_equal(self._y, self._y_smooth):
            self.smoothSpectrum.__wrapped__(
                self, w=w, p=p, logspace=logspace,
            )

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
        
    @validated_apply_info_to_method(subjects=('continuum',))
    def instantiateContinuum(
        self,
        *,
        windows: list[CoordBounds] | None = None,
    ) -> Self:
        """
        Instantiates the Spectrum instance's 'continuum_windows' attribute.
        """
        self.continuum_windows = ContinuumWindows.create.__wrapped__(
            ContinuumWindows,
            spectrum=self,
        )
        self.continuum_windows.populate.__wrapped__(
            self.continuum_windows,
            windows=windows,
        )
        return self

    @validated_apply_info_to_method(subjects=('iron',), specific_kwargs={'windows'})
    def instantiateIron(
        self,
        *,
        windows: list[CoordBounds] | None = None,
    ) -> Self:
        self.iron_windows = IronWindows.create.__wrapped__(
            IronWindows,
            spectrum=self,
        )
        self.iron_windows.populate.__wrapped__(
            self.iron_windows, 
            windows=windows,
        )
        return self
    
    @validated_apply_info_to_method(subjects=('balmer',), specific_kwargs={'windows'})
    def instantiateBalmer(
        self,
        *,
        windows: list[CoordBounds] | None = None,
    ) -> Self:
        self.balmer_windows = BalmerWindows.create.__wrapped__(
            BalmerWindows,
            spectrum=self,
        )
        self.balmer_windows.populate.__wrapped__(
            self.balmer_windows,
            windows=windows,
        )
        return self

    def instantiateLines(self) -> Self:
        self.line_windows = LineWindows.create.__wrapped__(
            LineWindows,
            spectrum=self,
        )
        return self

    @validated_apply_info_to_method(
        subjects=('lines',),
        specific_kwargs={
            'sigma_res', 'v_sep', 'forced_splits', 'min_fittable_total', 
            'min_fittable_ratio',
        },
    )
    def instantiateFromTemplateModel(
        self,
        template_model: Model_,
        *,
        linelist: AbsoluteFilePath | LineList,

        continuum_windows: list[CoordBounds] | None = None,
        iron_windows: list[CoordBounds] | None = None,
        balmer_windows: list[CoordBounds] | None = None,
        host_windows: list[CoordBounds] | None = None,

        sigma_res: float | None = None,
        v_sep: float | None = None,
        forced_splits: FloatVector | None = None,
        min_fittable_total: int | None = None,
        min_fittable_ratio: float | None = None,
    ) -> Self:
        fs = (template_model,) if template_model.n_submodels == 1 else template_model
        model_components = set(f.model_type for f in fs)

        if 'pl' in model_components:
            self.instantiateContinuum.__unvalidated__(
                self,
                windows=continuum_windows,
            )
        if 'fe' in model_components:
            self.instantiateIron.__unvalidated__(
                self,
                windows=iron_windows,
            )
        if 'ba' in model_components:
            self.instantiateBalmer.__unvalidated__(
                self,
                windows=balmer_windows,
            )
        if 'hg' in model_components:
            self.instantiateHost.__unvalidated__(
                self,
                windows=host_windows,
            )
        if 'em' in model_components:
            self.instantiateLines()
            self.line_windows.applyLineList.__unvalidated__(
                self.line_windows,
                linelist,
                sigma_res=sigma_res,
                v_sep=v_sep,
                forced_splits=forced_splits,
                min_fittable_total=min_fittable_total,
                min_fittable_ratio=min_fittable_ratio,
            )

        self.adoptFit.__wrapped__(
            self,
            template_model,
            update_emission=True,
        )
        return self
        
    @validated_apply_info_to_method(subjects=('continuum', 'nonlinear'))
    def fitContinuum(
        self,
        *,
        template_model: PowerLawModel | None = None,
        bg_flux: BackgroundFlux | None = None,

        windows: list[CoordBounds] | None = None,
        sigmas: list[float] | None = None,
        flux_bounds: AstropyBounds | None = None,
        alpha_bounds: AstropyBounds | None = None,
        fitter: FitterInstance | None = None,
    ) -> None:
        """
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
        self.instantiateContinuum.__wrapped__(
            self,
            windows=windows,
        )
        self.continuum_windows.__call__.__wrapped__(
            self.continuum_windows,

            template_model=template_model,
            bg_flux=bg_flux,

            sigmas=sigmas,
            flux_bounds=flux_bounds,
            alpha_bounds=alpha_bounds,
            fitter=fitter,
        )

    @validated_apply_info_to_method(subjects=('iron', 'nonlinear'))
    def fitIron(
        self,
        *,
        template_model: IronModel | CompoundModel_[IronModel] | None = None,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,

        windows: list[CoordBounds] | None = None,
        template_files: list[AbsoluteFITSPath] | None = None, 
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
        self.instantiateIron.__wrapped__(
            self, 
            windows=windows,
        )
        self.iron_windows.__call__.__wrapped__(
            self.iron_windows,

            template_model=template_model,
            bg_flux=bg_flux,
            without_rejections=without_rejections,
            without_absorption=without_absorption,

            template_files=template_files,
            resample=resample,
            flux_bounds=flux_bounds,
            fwhm_bounds=fwhm_bounds,
            split=split,
            fwhm=fwhm,
            bias=bias,
            ratio=ratio,
            scale=scale,
            allow_interp_fitting=allow_interp_fitting,
            raster=raster,
            fine_tune=fine_tune,
            fitter=fitter,
        )
        
    @validated_apply_info_to_method(subjects=('balmer', 'nonlinear'))
    def fitBalmer(
        self,
        *,
        template_model: BalmerModel | None = None,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,

        windows: list[CoordBounds] | None = None,
        template_files: list[AbsoluteFITSPath] | None = None, 
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
        self.instantiateBalmer.__wrapped__(
            self, 
            windows=windows,
        )
        self.balmer_windows.__call__.__wrapped__(
            self.balmer_windows,

            template_model=template_model,
            bg_flux=bg_flux,
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

    @validated_apply_info_to_method(subjects=('loading', 'lines', 'nonlinear'))
    def fitLines(
        self,
        linelist: AbsoluteFilePath | LineList,
        *,
        template_model: GaussianModel | CompoundModel_[GaussianModel] | None = None,

        bg_flux: BackgroundFlux = BackgroundFlux({'pl', 'fe', 'ba', 'hg'}),
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
        self.line_windows.__call__.__wrapped__(
            self.line_windows,
            linelist,

            template_model = template_model,
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

    @validated_apply_info_to_method(subjects=('nonlinear',))
    def finaliseFit(
        self,
        model_types: ModelTypes | None = None,
        data_types: DataTypes | None = None,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False, 
        without_absorption: bool = False,
        *,
        fitter: FitterInstance | None = None,
    ) -> bool:
        """
        Notes
        -----
        If 'model_types' and 'data_types' are both None, they default to all 
        types. If only one of them is None, it default to the other. 

        If 'bg_flux' is None, is default to all types not specified in 
        'model_types'.

        """
        from .utils.general import stopwatch

        s = self.__str__(simple=True).removesuffix('.')
        msg = f"Finalising fit for {s}: "

        if model_types is None:
            model_types = (
                ModelTypes({'all'}) 
                if data_types is None 
                else data_types.copy()
            )

        if data_types is not None:
            if not any(self[dt] is not None for dt in data_types):
                msg += f"no data types found for {data_types}. "
                logger.warning(msg)
                return False
        
        if bg_flux is None:
            _bg_flux = set(model_types)
            _bg_flux.add('all')
            bg_flux = BackgroundFlux(_bg_flux)

        msg += f"{model_types=}, {data_types=}, {bg_flux=} -> "

        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            mode='c', # c: contiguous
            data_types=data_types,
            bg_flux=bg_flux,
            covered=True,
            log_valid=False,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=True,
        )

        model = self.getModel.__wrapped__(self, model_types=model_types)
        if model is None:
            msg += "no models found!"
            logger.warning(msg)
            return False

        n_pix = masked_coords.size
        n_free_params = sum(get_free_params(model).values())

        if n_pix <= n_free_params:
            msg += "cancelling final fitting due to insufficient no. of data " \
                f"points (n_pix={n_pix} <= n_free_params={n_free_params})!"
            logger.critical(msg)
            return False
        
        msg += "n_pixels={}, n_submodels={} (n_free_params={}), " \
            .format(n_pix, model.n_submodels, n_free_params)
        
        try:
            with stopwatch() as watch:
                fit, fit_info = fitter(
                    model,
                    masked_coords.x,
                    masked_coords.y,
                    masked_coords.dy,
                    inplace = False,
                )
            msg += "successfully finalised fit in {:.1f} ms." \
                .format(watch.elapsed)
        except ValidationError as e: 
            msg += f"failed fitting due to validation error: {e}"
            logger.warning(msg)
            return False
        except Exception as e:
            msg += f"failed fitting due to an unexpected error: {e}"
            logger.warning(msg)
            return False
        
        logger.debug(msg)    
        self.applyFit.__wrapped__(
            self, 
            fit, 
            fit_info=fit_info,
            update_emission=True,
        )

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

    @validate_call
    def getMask(
        self,
        *,
        data_types: DataTypes | None = None,
        covered: bool = True,
        without_rejections: bool = False,
        without_absorption: bool = False,
        valid: bool = False,
        log_valid: bool = False,
    ) -> BoolVector:
        """
        ** PYDANTIC VALIDATED METHOD **

        Notes
        -----
        If 'data_types' is None, the entire spectrum is considered, i.e. 
        window-like objects' masks are not applied. 
        """
        mask = zeros_like(self._x, dtype=bool)
        if (data_types is None) or (not covered):
            mask[:] = True
        else:
            for d_type in filter(lambda dt: self[dt] is not None, data_types):
                mask |= self[d_type].mask
            
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
        data_types: DataTypes | None = None,
        bg_flux: BackgroundFlux | None = None,
        covered: bool = True,
        without_rejections: bool = False,
        without_absorption: bool = False,
        valid: bool = False,
        log_valid: bool = False,
    ) -> MaskedCoords:
        """
        Notes
        -----
        If 'data_types' is None, the entire spectrum is considered, i.e.
        window-like objects' masks are not applied. 
        
        If 'bg_flux' is None, no background flux is subtracted.
        """
        mask = self.getMask.__wrapped__(
            self,
            data_types=data_types,
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=valid,
            log_valid=log_valid,
        )
        match mode:
            case 'r': 
                return ReadOnlyMaskedCoords(self, mask, bg_flux=bg_flux)
            case 'c': 
                return ContiguousMaskedCoords(self, mask, bg_flux=bg_flux)
            case None:
                return MaskedCoords(self, mask, bg_flux=bg_flux)
    
    @validate_call
    def getModel(
        self,
        model_types: ModelTypes = ModelTypes({'all'}),
    ) -> Model_ | None:
        """
        ** PYDANTIC VALIDATED METHOD **
        """        
        models: list[Model_] = []
        for m_type in filter(lambda mt: self[mt] is not None, model_types):
            windows = self[m_type]
            model = windows.getModel()
            if model is not None:
                models.append(model)

        return None if len(models) == 0 else sum(models[1:], start=models[0])
    
    @validate_call
    def adoptFit(
        self,
        fit: Model_,
        *,
        fit_info: FitInfo | None = None,
        update_emission: bool = False,
    ) -> None:
        count = 0
        parts = get_model_parts(fit)
        for key in ('pl', 'fe', 'ba', 'hg', 'em'):
            model = parts[key]
            if model is None:
                continue

            if fit_info is None:
                finfo = None
            else:
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
                count += n_free

            windows = self[key]
            windows.adoptFit.__wrapped__(
                windows, 
                model, 
                fit_info=finfo,
                update_emission=update_emission,
            )
    
    @validate_call
    def applyFit(
        self,
        fit: Model_,
        *,
        fit_info: FitInfo | None = None,
        update_emission: bool = False,
    ) -> None:
        count = 0
        parts = get_model_parts(fit)
        for key in ('pl', 'fe', 'ba', 'hg', 'em'):
            model = parts[key]
            if model is None:
                continue

            if fit_info is None:
                finfo = None
            else:
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
                count += n_free

            windows = self[key]
            windows.applyFit.__wrapped__(
                windows, 
                model, 
                fit_info=finfo,
                update_emission=update_emission,
            )
    
    def summariseContinuumFit(self) -> ContinuumFitResult:
        try:
            return self.continuum_windows.summariseContinuumFit()
        except Exception as _:
            return None
    
    @validated_apply_info_to_method(subjects=('error', 'nonlinear'))
    def instantiateBootstrapper(
        self,
        *,
        covered: bool = True,
        data_types: DataTypes = DataTypes({'all'}),
        model_types: ModelTypes = ModelTypes({'all'}),

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
        tqdm_leave: bool | None = None,

        fitter: FitterInstance | None = None,
    ) -> BaseBootstrapper:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if scale == 'semilocal' or scale == 'local':
            if self.em is None:
                msg = f"Cannot perform bootstrapping on '{scale}' scale \
                    without emission line windows!"
                logger.warning(msg)
                raise ValueError(msg)
            
            for lwindow in self.line_windows:
                lwindow.instantiateBootstrapper.__wrapped__(
                    lwindow,
                    pl=self.pl.getModel(),
                    fe=self.fe.getModel(),
                    ba=self.ba.getModel(),
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
                    tqdm_leave=tqdm_leave,
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
            tqdm_leave=tqdm_leave,
            logger=logger,
        )
        self.bootstrapper = cls(*args, **kwargs)
        return self.bootstrapper

    @validated_apply_info_to_method(subjects=('error',), specific_kwargs={'scale'})
    def runBootstrapper(self, *, scale: Scale | None = None) -> ErrorResult:
        """
        ** PYDANTIC VALIDATED METHOD // INFO APPLIED TO METHOD **
        """
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
    
    @validated_apply_info_to_method(subjects=('error',))
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
        ** PYDANTIC VALIDATED METHOD // INFO APPLIED TO METHOD **
        """
        raise NotImplementedError

    @validated_apply_info_to_method(subjects=('error',))
    def instantiateNestedSampler(
        self,
        *,
        covered: bool = True,
        data_types: DataTypes = DataTypes({'all'}),
        model_types: ModelTypes = ModelTypes({'all'}),

        without_rejections: bool = False,
        without_absorption: bool = False,

        pool: Pool_ | None = None,

        iterations: int | None = None,
        random_state: RandomState_ | None = None,
        renew_rng: bool | None = None,
        replace_missing: bool | None = None,
        tqdm_disable: bool | None = None,
        tqdm_leave: bool | None = None,
    ) -> BaseNestedSampler:
        """
        ...
        """
        self.nested_sampler = UniformNestedSampler.from_spectrum(
            spectrum=self,
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            data_types=data_types,
            model_types=model_types,
            pool=pool,
            iterations=iterations,
            random_state=random_state,
            renew_rng=renew_rng,
            replace_missing=replace_missing,
            tqdm_disable=tqdm_disable,
            tqdm_leave=tqdm_leave,
            logger=logger,
        )
        self.nested_sampler.instantiateSampler()
        return self.nested_sampler

    def runNestedSampler(
        self, 
        *, 
        maxiter: int,
        dlogz: float | None = None,
        print_progress: bool = True,
        scale: Scale | None = None) -> ErrorResult:
        """
        ...
        """
        if scale == 'semilocal' or scale == 'local':
            pass

        if self.nested_sampler is None:
            msg = "No nested sampler instance found! "\
                "Run 'instantiateNestedSampler()' method first."
            logger.critical(msg)
            raise AttributeError(msg)
        
        if self.error_result is not None:
            logger.debug("Existing 'error_result' will be overwritten.")

        out = self.nested_sampler.run(
            maxiter=maxiter,
            dlogz=dlogz,
            print_progress=print_progress,
        )

        self.error_result = self.nested_sampler.toErrorResult(out)
        return self.error_result

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
            title=title or self.title,
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
            title=title or self.title,
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
        model = self.getModel() if plot_components else None
        return fitplot(
            get_coords(self, x_bounds=xlim, replace_with_nan=False),
            self.info,
            model=model,
            plot_type=plot_type,
            figure=figure,
            figsize=figsize,
            dpi=dpi,
            title=title or self.title,
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