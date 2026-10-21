import pickle
from dataclasses import field
from logging import getLogger
from typing import Any, Literal, Optional, Self, Union, overload

import numpy as np
import pydantic
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic_core import ValidationError
from quasar_models import (
    BalmerModel,
    GaussianModel,
    HostGalaxyModel,
    IronModel,
    PowerLawModel,
)
from quasar_models.modeling import BaseModel
from quasar_models.utils.astropy import get_free_params, get_model_parts
from quasar_typing.astropy import CompoundModelType, ModelType
from quasar_typing.bounds import AstropyBounds, CoordBounds
from quasar_typing.literals import BootstrapType, Variant
from quasar_typing.multiprocessing import PoolType
from quasar_typing.numpy import BoolVector, FloatVector, RandomStateType
from quasar_typing.pathlib import FilePath, FITSPath
from quasar_typing.scipy import OptimizeResultType
from quasar_typing.string_selections import (
    BackgroundFlux,
    DataTypes,
    ModelTypes,
    VaryLines,
)
from quasar_utils.decorators import validated_apply_info_to_method
from quasar_utils.linelist import LineList
from quasar_utils.setup import FitterKwargs

from quasar_decrypt.utils import MaskedCoords, _Spec, _SpecWindowList

from .balmer import BalmerWindows
from .continuum import ContinuumWindows
from .host import HostWindows
from .iron import IronWindows
from .lines import LineWindows

logger = getLogger(__name__)


@pydantic.dataclasses.dataclass
class Spectrum(_Spec):
    path: FilePath = field(kw_only=True)
    title: str = field(kw_only=True)

    continuum_windows: ContinuumWindows | None = field(
        default=None, init=False
    )
    iron_windows: IronWindows | None = field(default=None, init=False)
    balmer_windows: BalmerWindows | None = field(default=None, init=False)
    host_windows: HostWindows | None = field(default=None, init=False)
    line_windows: LineWindows | None = field(default=None, init=False)

    emcee_sampler: object | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.findAnomalies()

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["path"] = self.path
        state["title"] = self.title
        state["continuum_windows"] = self.continuum_windows
        state["iron_windows"] = self.iron_windows
        state["balmer_windows"] = self.balmer_windows
        state["host_windows"] = self.host_windows
        state["line_windows"] = self.line_windows
        state["emcee_sampler"] = self.emcee_sampler
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.path = state["path"]
        self.title = state["title"]

        self.continuum_windows = state.get("continuum_windows", None)
        self.iron_windows = state.get("iron_windows", None)
        self.balmer_windows = state.get("balmer_windows", None)
        self.host_windows = state.get("host_windows", None)
        self.line_windows = state.get("line_windows", None)

        self.emcee_sampler = state.get("emcee_sampler", None)

    def _finalize_unpickling(self) -> None:
        """
        Set the `spectrum` for this object and all window(list) objects.
        """
        super()._finalize_unpickling()
        if self.pl is not None:
            self.pl._finalize_unpickling(self)
        if self.fe is not None:
            self.fe._finalize_unpickling(self)
        if self.ba is not None:
            self.ba._finalize_unpickling(self)
        if self.hg is not None:
            self.hg._finalize_unpickling(self)
        if self.em is not None:
            self.em._finalize_unpickling(self)

    def dump(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> Self:
        with open(path, "rb") as f:
            obj: Self = pickle.load(f)
        obj._finalize_unpickling()
        return obj

    @property
    def pl(self) -> ContinuumWindows | None:
        return self.continuum_windows

    @property
    def pl_sample(self):
        return None if self.pl is None else self.pl.sample

    @property
    def fe(self) -> IronWindows | None:
        return self.iron_windows

    @property
    def fe_sample(self):
        return None if self.fe is None else self.fe.sample

    @property
    def ba(self) -> BalmerWindows | None:
        return self.balmer_windows

    @property
    def ba_sample(self):
        return None if self.ba is None else self.ba.sample

    @property
    def hg(self) -> HostWindows | None:
        return self.host_windows

    @property
    def hg_sample(self):
        return None if self.hg is None else self.hg.sample

    @property
    def em(self) -> LineWindows:
        return self.line_windows

    @property
    def em_sample(self):
        return None if self.em is None else self.em.sample

    @property
    def basic_sample(self):
        from quasar_errors.model_samples import Sample
        if (model := self.getModel()) is None:
            return Sample.fromSpectrumSample(self.x, self.y, self.dy)
        return Sample.fromModelSample(self.x, self.y, self.dy, model)

    @property
    def basic_error_result(self):
        from quasar_errors.error_result import ErrorResult
        return ErrorResult((self.basic_sample,), (1.0,))

    @validated_apply_info_to_method(subjects=("absorption", "lines"))
    def __call__(
        self,
        linelist: FilePath | LineList,
        *,
        refine: bool = False,
        x_limit: float | None = None,
    ) -> None:
        logger.debug("Running 'Spectrum' pipeline...")

        self.cropLymanAlphaForest.__wrapped__(
            self,
            x_limit=x_limit,
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

    @overload
    def __getitem__(self, key: Literal["pl"]) -> ContinuumWindows | None: ...

    @overload
    def __getitem__(self, key: Literal["fe"]) -> IronWindows | None: ...

    @overload
    def __getitem__(self, key: Literal["ba"]) -> BalmerWindows | None: ...

    @overload
    def __getitem__(self, key: Literal["hg"]) -> HostWindows | None: ...

    @overload
    def __getitem__(self, key: Literal["em"]) -> LineWindows | None: ...

    def __getitem__(
        self,
        key: Literal["pl", "fe", "ba", "hg", "em"],
    ) -> _SpecWindowList | None:
        match key:
            case "pl":
                return self.continuum_windows
            case "fe":
                return self.iron_windows
            case "ba":
                return self.balmer_windows
            case "hg":
                return self.host_windows
            case "em":
                return self.line_windows
            
    def findAnomalies(self) -> None:
        """
        Identifies anomalous pixels, and replaces them with NaN. Anomalous
        pixels have: non-positive flux densities OR non-positive flux density
        uncertainties.
        """
        n = self._x.size
        cond_1 = ~(np.isfinite(self._x))
        cond_2 = ~(np.isfinite(self._y))
        cond_3 = ~(np.isfinite(self._dy)) | (self._dy <= 0)

        mask = cond_1 | cond_2 | cond_3

        msg = f">>> Found {mask.sum()}/{n} anomalous pixels: "
        msg += f"{cond_1.sum()}/{n} (invalid wave), "
        msg += f"{cond_2.sum()}/{n} (invalid flux), "
        msg += f"{cond_3.sum()}/{n} (invalid flux unc.)."
        logger.debug(msg)

        self._valid_pixels[:] = ~mask

    def _truncate_range(self, sel: slice) -> Self:
        """
        Creates a new 'Spectrum' instance with the same, but truncated, data 
        arrays.
        """
        if sel.step not in (None, 1):
            raise ValueError("Only contiguous truncation is supported.")

        def f(arr):
            return arr[sel]

        kwargs = self.get_kwargs.__wrapped__(
            self.__class__,
            path=self.path,
            title=self.title,
            x=f(self._x),
            y=f(self._y),
            dy=f(self._dy),
            dx=f(self._dx),
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
            info=self.info,
            x_bounds=None,
        )
        return self.__class__(**kwargs)

    def truncateSpectrum(self) -> Self:
        """
        Truncates the spectrum if edges contain NaN values.

        Returns the same 'Spectrum' instance if no truncation is necessary, 
        otherwise returns a new 'Spectrum' instance with the same, but 
        truncated, data arrays (see: `_truncate_range`).
        """
        logger.debug("Truncating spectrum:")

        mask = self._valid_pixels
        n_valid = self._valid_pixels.sum()
        logger.debug(f">>> Found {n_valid}/{mask.size} valid pixels.")

        if n_valid == 0:
            msg = ">>> All pixels are invalid! Cannot truncate spectrum."
            logger.critical(msg)
            raise ValueError(msg)
        elif n_valid == 1:
            msg = ">>> Only one valid pixel! Cannot truncate spectrum."
            logger.critical(msg)
            raise ValueError(msg)
        elif mask[0] and mask[-1]:
            logger.debug(">>> No invalid pixels at either edge")
            return self

        valid_indices = np.argwhere(mask).flatten()
        idx_start = valid_indices[0]
        idx_end = valid_indices[-1]

        lb, ub = self.x_bounds
        new_lb = self._x[idx_start]
        new_ub = self._x[idx_end]
        if lb < new_lb:
            logger.debug(f">>> Increasing lower bound: {lb:.1f} -> {new_lb:.1f}")
            lb = new_lb
        if new_ub < ub:
            logger.debug(f">>> Decreasing upper bound: {ub:.1f} -> {new_ub:.1f}")
            ub = new_ub

        self.x_bounds = (new_lb, new_ub)
        return self

    @validated_apply_info_to_method(subjects=("lines",))
    def cropLymanAlphaForest(
        self,
        *,
        x_limit: float | None = None,
    ) -> Self:
        logger.debug("Cropping Lyman-alpha forest:")

        lb = self.x_bounds[0]

        indices = np.argwhere((lb < self._x) & (self._x < x_limit)).flatten()
        logger.debug(
            f">>> Found {indices.size}/{self._x.size} pixels "
            f"above {lb=:.1f} and below {x_limit=:.1f}"
        )

        if not (indices.size and ((edge_idx := indices[-1]) >= 1)):
            logger.debug(
                ">>> No pixels found above the lower bound and below `x_limit`. "
                "Returning early!")
            return self

        _y = self._y[:edge_idx+1]
        if not np.isfinite(_y).any():
            logger.debug(
                ">>> No finite flux densities blue-wards of `x_limit`. "
                "Returning early!"
            )
            return self

        idx_max = np.nanargmax(_y).flatten()[0]
        x_max = self._x[idx_max]
        y_max = self._y[idx_max]
        logger.debug(
            f">>> Found peak at index {idx_max}: "
            f"({x_max=:.1f}, {y_max=:.1f})"
        )
        if x_max < lb:
            logger.debug(
                ">>> Peak is blue-wards of the existing lower bound! "
                "Returning early!"
            )
            return self

        self.x_bounds = (x_max, self.x_bounds[1])
        return self

    @validated_apply_info_to_method(subjects=("absorption",))
    def smoothSpectrum(
        self,
        *,
        w: int | None = None,
        p: int | None = None,
        logspace: int | None = None,
    ) -> None:
        from quasar_utils.absorption import _smooth_spectrum
        self._y_smooth[:] = _smooth_spectrum(
            self._x, self._y, self._dy,
            self._valid_pixels,
            w,
            p,
            logspace,
        )

    @validated_apply_info_to_method(subjects=("absorption",))
    def removeAbsorption(
        self,
        *,
        bg_flux: BackgroundFlux | None = None,
        w: int | None = None,
        p: int | None = None,
        p_crit: float | None = None,
        z_crit: float | None = None,
        join: int | None = None,
        refine: bool | None = None,
        logspace: bool | None = None,
    ) -> None:
        from quasar_utils.absorption import _remove_absorption

        if bg_flux is None:
            bg_flux = BackgroundFlux({"all"})

        if np.isnan(self._y_smooth).all():
            self.smoothSpectrum.__wrapped__(
                self,
                w=w,
                p=p,
                logspace=logspace,
            )

        result = _remove_absorption(
            self._x, self._y, self._dy,
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

    @validated_apply_info_to_method(subjects=("continuum",))
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

    @validated_apply_info_to_method(
        subjects=("iron",), specific_kwargs={"windows"}
    )
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

    @validated_apply_info_to_method(
        subjects=("balmer",), specific_kwargs={"windows"}
    )
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

    @validated_apply_info_to_method(
        subjects=("host",), specific_kwargs={"windows"}
    )
    def instantiateHost(
        self,
        *,
        windows: list[CoordBounds] | None = None,
    ) -> Self:
        self.host_windows = HostWindows.create.__wrapped__(
            HostWindows,
            spectrum=self,
        )
        self.host_windows.populate.__wrapped__(
            self.host_windows,
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
        subjects=("lines",),
        specific_kwargs={
            "sigma_res",
            "v_sep",
            "forced_splits",
            "min_fittable_total",
            "min_fittable_ratio",
        },
    )
    def instantiateFromTemplateModel(
        self,
        template_model: ModelType,
        *,
        linelist: FilePath | LineList,
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
        fs = (
            (template_model,)
            if template_model.n_submodels == 1
            else template_model
        )
        model_components = {f.model_type for f in fs}

        if "pl" in model_components:
            self.instantiateContinuum.__unvalidated__(
                self,
                windows=continuum_windows,
            )
        if "fe" in model_components:
            self.instantiateIron.__unvalidated__(
                self,
                windows=iron_windows,
            )
        if "ba" in model_components:
            self.instantiateBalmer.__unvalidated__(
                self,
                windows=balmer_windows,
            )
        if "hg" in model_components:
            self.instantiateHost.__unvalidated__(
                self,
                windows=host_windows,
            )
        if "em" in model_components:
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

    @validated_apply_info_to_method(subjects=("continuum", "nonlinear"))
    def fitContinuum(
        self,
        *,
        template_model: Optional[PowerLawModel] = None,
        original: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        bg_flux: BackgroundFlux | None = None,
        shrink: float | None = None,
        windows: list[CoordBounds] | None = None,
        sigmas: list[float] | None = None,
        flux_bounds: AstropyBounds | None = None,
        alpha_bounds: AstropyBounds | None = None,
        min_fittable_total: int | None = None,
        fitter_kwargs: FitterKwargs | None = None,
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
            original=original,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            template_model=template_model,
            bg_flux=bg_flux,
            shrink=shrink,
            sigmas=sigmas,
            flux_bounds=flux_bounds,
            alpha_bounds=alpha_bounds,
            min_fittable_total=min_fittable_total,
            fitter_kwargs=fitter_kwargs,
        )

    @validated_apply_info_to_method(subjects=("iron", "nonlinear"))
    def fitIron(
        self,
        *,
        template_model: Union[
            IronModel, 
            CompoundModelType[IronModel], 
            None,
        ] = None,
        original: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        bg_flux: BackgroundFlux | None = None,
        shrink: float | None = None,
        windows: list[CoordBounds] | None = None,
        template_files: list[FITSPath] | None = None,
        resample: bool | None = None,
        flux_bounds: AstropyBounds | None = None,
        fwhm_bounds: AstropyBounds | None = None,
        split: FloatVector | None = None,
        fwhm: FloatVector | None = None,
        bias: list[str] | None = None,
        ratio: FloatVector | None = None,
        scale: float | None = None,
        tie_similar: bool | None = None,
        allow_interp_fitting: bool | None = None,
        raster: bool | None = None,
        fine_tune: bool | None = None,
        fitter_kwargs: FitterKwargs | None = None,
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
            original=original,
            bg_flux=bg_flux,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            shrink=shrink,
            template_files=template_files,
            resample=resample,
            flux_bounds=flux_bounds,
            fwhm_bounds=fwhm_bounds,
            split=split,
            fwhm=fwhm,
            bias=bias,
            ratio=ratio,
            scale=scale,
            tie_similar=tie_similar,
            allow_interp_fitting=allow_interp_fitting,
            raster=raster,
            fine_tune=fine_tune,
            fitter_kwargs=fitter_kwargs,
        )

    @validated_apply_info_to_method(subjects=("balmer", "nonlinear"))
    def fitBalmer(
        self,
        *,
        template_model: Optional[BalmerModel] = None,
        original: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        bg_flux: BackgroundFlux | None = None,
        shrink: float | None = None,
        windows: list[CoordBounds] | None = None,
        flux: float | None = None,
        fwhm: float | None = None,
        ratio: float | None = None,
        source: Literal["sh1995"] | None = None,
        temp: float | None = None,
        dens: float | None = None,
        n_u_min: int | None = None,
        n_u_max: int | None = None,
        tau: float | None = None,
        scale: float | None = None,
        allow_interp_fitting: bool | None = None,
        fixed: dict[str, bool] | None = None,
        flux_bounds: AstropyBounds | None = None,
        fwhm_bounds: AstropyBounds | None = None,
        ratio_bounds: AstropyBounds | None = None,
        min_fittable_ratio: float | None = None,
        min_fittable_total: int | None = None,
        raster: bool | None = None,
        fine_tune: bool | None = None,
        fitter_kwargs: FitterKwargs | None = None,
    ) -> None:
        self.instantiateBalmer.__wrapped__(
            self,
            windows=windows,
        )
        self.balmer_windows.__call__.__wrapped__(
            self.balmer_windows,
            template_model=template_model,
            original=original,
            bg_flux=bg_flux,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            covered=not self.ba.is_empty,
            shrink=shrink,
            flux=flux,
            fwhm=fwhm,
            ratio=ratio,
            source=source,
            temp=temp,
            dens=dens,
            n_u_min=n_u_min,
            n_u_max=n_u_max,
            tau=tau,
            scale=scale,
            allow_interp_fitting=allow_interp_fitting,
            fixed=fixed,
            flux_bounds=flux_bounds,
            fwhm_bounds=fwhm_bounds,
            ratio_bounds=ratio_bounds,
            min_fittable_ratio=min_fittable_ratio,
            min_fittable_total=min_fittable_total,
            raster=raster,
            fine_tune=fine_tune,
            fitter_kwargs=fitter_kwargs,
        )

    @validated_apply_info_to_method(subjects=("host", "nonlinear"))
    def fitHost(
        self,
        *,
        template_model: Optional[HostGalaxyModel] = None,
        original: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        bg_flux: BackgroundFlux | None = None,
        shrink: bool | None = None,
        windows: list[CoordBounds] | None = None,
        x_min: float | None = None,
        template_files: list[FITSPath] | None = None,
        sources: list[Literal["bc2003"]] | None = None,
        ages: list[int] | None = None,
        default_source: Literal["bc2003"] | None = None,
        default_age: int | None = None,
        flux: float | None = None,
        fwhm: float | None = None,
        flux_bounds: AstropyBounds | None = None,
        fwhm_bounds: AstropyBounds | None = None,
        allow_interp_fitting: bool | None = None,
        fixed: dict[str, bool] | None = None,
        raster: bool = True,
        fine_tune: bool = True,
        fitter_kwargs: FitterKwargs | None = None,
    ) -> None:
        self.instantiateHost.__wrapped__(
            self,
            windows=windows,
        )
        self.host_windows.__call__.__wrapped__(
            self.host_windows,
            template_model=template_model,
            original=original,
            bg_flux=bg_flux,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            covered=not self.hg.is_empty,
            shrink=shrink,
            x_min=x_min,
            template_files=template_files,
            sources=sources,
            ages=ages,
            default_source=default_source,
            default_age=default_age,
            flux=flux,
            fwhm=fwhm,
            flux_bounds=flux_bounds,
            fwhm_bounds=fwhm_bounds,
            allow_interp_fitting=allow_interp_fitting,
            fixed=fixed,
            raster=raster,
            fine_tune=fine_tune,
            fitter_kwargs=fitter_kwargs,
        )

    @validated_apply_info_to_method(subjects=("loading", "lines", "nonlinear"))
    def fitLines(
        self,
        linelist: FilePath | LineList,
        *,
        template_model: Optional[
            Union[GaussianModel, CompoundModelType[GaussianModel]]
        ] = None,
        original: bool | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        bg_flux: BackgroundFlux | None = None,
        shrink: bool | None = None,
        limited: bool = False,
        sigma_res: float | None = None,
        v_sep: float | None = None,
        forced_splits: FloatVector | None = None,
        w: int | None = None,
        min_fittable_total: int | None = None,
        min_fittable_ratio: float | None = None,
        evaluate_initial: float | None = None,
        aggressive: bool | None = None,
        crop: bool | None = None,
        measure: str | None = None,
        reverse: bool | None = None,
        fitter_kwargs: FitterKwargs | None = None,
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
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        self.instantiateLines()
        self.line_windows.__call__.__wrapped__(
            self.line_windows,
            linelist,
            template_model=template_model,
            original=original,
            bg_flux=bg_flux,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            with_neighbours=with_neighbours,
            shrink=shrink,
            limited=limited,
            v_sep=v_sep,
            forced_splits=forced_splits,
            w=w,
            sigma_res=sigma_res,
            min_fittable_total=min_fittable_total,
            min_fittable_ratio=min_fittable_ratio,
            evaluate_initial=evaluate_initial,
            aggressive=aggressive,
            crop=crop,
            measure=measure,
            reverse=reverse,
            fitter_kwargs=fitter_kwargs,
        )

    @pydantic.validate_call
    def finaliseFit(
        self,
        *,
        original: bool = False,
        model_types: ModelTypes | None = None,
        data_types: DataTypes | None = None,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        shrink: bool | None = None,
        **fitter_kwargs,
    ) -> bool:
        """
        Notes
        -----
        If 'model_types' and 'data_types' are both None, they default to all
        types. If only one of them is None, it default to the other.

        If 'bg_flux' is None, is default to all types not specified in
        'model_types'.

        """
        from quasar_models.modeling import Fitter, PrepareModel

        s = self.__str__(simple=True).removesuffix(".")
        msg = f"Finalising fit for {s}: "

        masked_coords, model = self._final_data_and_model.__wrapped__(
            self,
            original=original,
            model_types=model_types,
            data_types=data_types,
            bg_flux=bg_flux,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            shrink=shrink,
        )
        fitter_kwargs = self.info.nonlinear.fitter_kwargs | fitter_kwargs

        preparer = PrepareModel.fromCoords(masked_coords, model)
        data = preparer.createDataDict(masked_coords)
        try:
            with preparer as (seq_model, where):
                fitter = Fitter()
                fit = fitter(
                    seq_model,
                    data["x"],
                    data["y"],
                    dy=data["dy"],
                    where=where,
                    get_model=True,
                    inplace=False,
                    **fitter_kwargs,
                )
                fit_info = fitter.fit_info

            msg += "successfully!"
        except ValidationError as e:
            msg += f"failed fitting due to validation error: {e}"
            logger.info(msg)
            return False

        logger.debug(msg)
        self.applyFit.__wrapped__(
            self,
            fit,
            fit_info=fit_info,
            update_emission=True,
        )
        return True

    @validated_apply_info_to_method(subjects=("random",))
    def finaliseFitWithMCMC(
        self,
        *,
        without_rejections: bool = False,
        without_absorption: bool = False,
        model_types: ModelTypes | None = None,
        data_types: DataTypes | None = None,
        bg_flux: BackgroundFlux | None = None,
        progress: bool = True,
        pool: Any | None = None,
        random_state: RandomStateType | None = None,
        **mcmc_kwargs,
    ):
        from quasar_models.modeling import MCMCFitter, PrepareModel

        s = self.__str__(simple=True).removesuffix(".")
        msg = f"Finalising fit with MCMC for {s}: "

        masked_coords, model = self._final_data_and_model.__wrapped__(
            self,
            model_types=model_types,
            data_types=data_types,
            bg_flux=bg_flux,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
        )
        mcmc_kwargs = self.info.mcmc.mcmc_kwargs | mcmc_kwargs

        try:
            with PrepareModel(x=masked_coords.x, model=model, is_logbinned=True) as seq_model:
                fitter = MCMCFitter()
                fit = fitter(
                    seq_model,
                    masked_coords.x,
                    masked_coords.y,
                    dy=masked_coords.dy,
                    get_model=True,
                    inplace=False,
                    progress=progress,
                    vectorize=pool is None,
                    random_state=random_state,
                    **mcmc_kwargs,
                )
                fit_info = fitter.fit_info

                self.emcee_sampler = fitter.sampler

            msg += "successfully!"
        except ValidationError as e:
            msg += f"failed fitting due to validation error: {e}"
            logger.info(msg)
            return False

        logger.debug(msg)
        self.applyFit.__wrapped__(
            self,
            fit,
            fit_info=fit_info,
            update_emission=True,
        )
        return fitter

    @validated_apply_info_to_method(
        subjects=("nonlinear",), 
        specific_kwargs=("fitter_kwargs",),
    )
    def chooseHostGalaxyTemplate(
        self,
        *,
        original: bool = False,
        model_types: ModelTypes | None = None,
        data_types: DataTypes | None = None,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        **fitter_kwargs,
    ) -> Optional[HostGalaxyModel]:
        from quasar_models.modeling import Fitter, PrepareModel

        msg = "Choosing host galaxy template: "
        if self.hg is None:
            msg += "no `HostWindows` instance available!"
            logger.info(msg)
            return None

        n = len(self.hg.models)
        if n == 0:
            msg += "no host galaxy templates available!"
            logger.info(msg)
            return None
        elif n == 1:
            msg += "only one host galaxy template available."
            logger.debug(msg)
            return self.hg.getModel()

        if model_types is not None:
            model_types.add("hg")

        if (bg_flux is not None) and ("hg" in bg_flux):
            logger.info(
                "Host galaxy emission included in background flux "
                "-> Removing!"
            )
            bg_flux.remove("hg")
    
        masked_coords, _model = self._final_data_and_model.__wrapped__(
            self,
            original=original,
            model_types=model_types,
            data_types=data_types,
            bg_flux=bg_flux,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
        )
        ms: list[BaseModel] = [
            m
            for m in ([_model] if _model.n_submodels == 1 else list(_model))
            if m.model_type != "hg"
        ]
        bg_model = sum(ms[1:], start=ms[0])

        fits: dict[str, Union[BaseModel, CompoundModelType[BaseModel]]] = {}
        fit_infos: dict[str, OptimizeResultType] = {}
        for template_name, host_model in self.hg.models.items():
            model = bg_model + host_model
            try:
                with PrepareModel(x=masked_coords.x, model=model) as seq_model:
                    fitter = Fitter()
                    fits[template_name] = fitter(
                        seq_model,
                        masked_coords.x,
                        masked_coords.y,
                        dy=masked_coords.dy,
                        get_model=True,
                        inplace=False,
                        **fitter_kwargs,
                    )
                    fit_infos[template_name] = fitter.fit_info
            except ValidationError as e:
                msg += f"failed fitting due to validation error: {e}"
                logger.info(msg)
                return None

        # Update host galaxy models
        for template_name, host_model in self.hg.models.items():
            self.hg.models[template_name] = fits[template_name][host_model.name]

        chi2s: dict[str, float] = {
            template_name: (fit_info.fun ** 2).mean()
            for template_name, fit_info in fit_infos.items()
        }
        best_template: str = min(chi2s, key=chi2s.get)

        for template_name, chi2 in chi2s.items():
            msg += f"({template_name}) {chi2:.1f} "
        msg = msg.removesuffix(" ")
        msg += f"-> Best template: {best_template}"
        logger.debug(msg)

        self.hg.chooseModel.__wrapped__(self.hg, best_template)
        self.applyFit.__wrapped__(
            self,
            fits[best_template],
            fit_info=fit_infos[best_template],
            update_emission=True,
        )
        return fits[best_template]

    @pydantic.validate_call
    def _final_data_and_model(
        self,
        original: bool = False,
        model_types: ModelTypes | None = None,
        data_types: DataTypes | None = None,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        shrink: bool | None = None,
    ) -> tuple[MaskedCoords, Union[BaseModel, CompoundModelType[BaseModel]]]:
        """
        Parameters
        ----------
        original : bool
            If True, the model will be fit to the original data arrays if these
            are not identical to the currently used data arrays.
        model_types : ModelTypes | None
            The types of models to include in the final model. If None, all
            available model types are included. 
        data_types : DataTypes | None
            The types of data to include in the final dataset. If None, the 
            entire spectrum is included (not just the data from all spectral 
            windows).
        bg_flux : BackgroundFlux | None
            The flux density contributions to be subtracted from the spectrum 
            before fitting. If None, defaults to all model types not included in 
            'model_types'.
        without_rejections : bool
            If True, pixels masked during sigma clipping are excluded.
        without_absorption : bool
            If True, pixels masked due to absorption are excluded.
        """
        logger.debug("Getting final data and model:")
        original = self._update_original_flag(
            original=original,
            log=True,
            prefix=">>> ",
        )
        if model_types is None:
            # Include all model types whose corresponding spectral windows exist
            model_types = ModelTypes({
                t
                for t in ModelTypes._string_options
                if getattr(self, t) is not None
            })

        if bg_flux is None:
            # Include all possible model types not already in 'model_types'
            bg_flux = BackgroundFlux({
                t 
                for t in BackgroundFlux._string_options 
                if t not in model_types
            })

        logger.debug(
            f">>> {model_types=}, {bg_flux=}, {original=}, {data_types=}, "
            f"{without_rejections=}, {without_absorption=}"
        )
        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            original=original,
            mode="c",
            data_types=data_types,
            bg_flux=bg_flux,
            covered=True,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=True,
            log_valid=False,
            shrink=shrink,
        )
        logger.debug(f">>> MaskedCoords: size={masked_coords.size}")

        model = self.getModel(model_types=model_types)
        if model is None:
            msg = "Could not retrieve any model(s)!"
            logger.critical(msg)
            raise ValueError(msg)

        n_free = sum(get_free_params(model).values())
        if masked_coords.size <= n_free:
            msg = "The problem is undervalued: "\
                f"size={masked_coords.size}, {n_free=}"
            logger.critical(msg)
            raise ValueError(msg)

        return masked_coords, model

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

    @pydantic.validate_call
    def getMask(
        self,
        *,
        original: bool = False,
        data_types: DataTypes | None = None,
        covered: bool = True,
        without_rejections: bool = False,
        without_absorption: bool = False,
        valid: bool = False,
        log_valid: bool = False,
        as_tuple: bool = False,
    ) -> BoolVector | tuple[BoolVector, BoolVector]:
        """
        Notes
        -----
        If 'data_types' is None, the entire spectrum is considered, i.e.
        window-like objects' masks are not applied.
        """
        original = self._update_original_flag(
            original=original,
            log=False,
        )
        suffix = "_original" if original else ""

        primary_mask = getattr(self, "mask" + suffix).copy(order="C")
        if not ((data_types is None) or (not covered)):
            for d_type in data_types:
                windows = getattr(self, d_type, None)
                if windows is None:
                    continue
                primary_mask |= getattr(windows, "mask" + suffix)

        secondary_mask = primary_mask.copy(order="C")
        if without_rejections:
            secondary_mask &= ~getattr(self, "_rejected_pixels" + suffix)
        if without_absorption:
            secondary_mask &= ~getattr(self, "_absorbed_pixels" + suffix)

        if log_valid:
            secondary_mask &= getattr(self, "_log_valid_pixels" + suffix)
        elif valid:
            secondary_mask &= getattr(self, "_valid_pixels" + suffix)

        if as_tuple:
            return primary_mask, secondary_mask
        return primary_mask & secondary_mask

    @pydantic.validate_call
    def getMaskedCoords(
        self,
        *,
        original: bool = False,
        data_types: DataTypes | None = None,
        covered: bool = True,
        mode: Literal["r", "c"] | None = None,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        valid: bool = False,
        log_valid: bool = False,
        shrink: bool | None = None,
    ) -> MaskedCoords:
        """
        Notes
        -----
        If 'data_types' is None, the entire spectrum is considered, i.e.
        window-like objects' masks are not applied.

        If 'bg_flux' is None, no background flux is subtracted.
        """
        original = self._update_original_flag(
            original=original,
            log=False,
        )
        primary_mask, secondary_mask = self.getMask.__wrapped__(
            self,
            original=original,
            data_types=data_types,
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=valid,
            log_valid=log_valid,
            as_tuple=True,
        )
        return self._get_masked_coords_helper(
            original=original,
            mode=mode,
            primary_mask=primary_mask,
            secondary_mask=secondary_mask,
            log_valid=log_valid,
            bg_flux=bg_flux,
            shrink=shrink,
        )

    @pydantic.validate_call
    def getModel(
        self,
        model_types: ModelTypes | None = None,
    ) -> Optional[ModelType]:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if model_types is None:
            model_types = ModelTypes({"all"})
            
        models: list[ModelType] = []
        for m_type in filter(lambda mt: self[mt] is not None, model_types):
            windows = self[m_type]
            model = windows.getModel()

            if model is not None:
                models.append(model)

        return None if len(models) == 0 else sum(models[1:], start=models[0])

    @pydantic.validate_call
    def updateOriginalRejections(
        self, 
        coverage: Literal["partial", "full"] = "full",
    ) -> None:
        if self._no_original:
            return
        
        _rej = np.interp(
            self._x_original,
            self._x,
            self._rejected_pixels.astype(np.float64),
            left=0.0,
            right=0.0,
        )
        mask = (_rej == 1.0) if coverage == "full" else (_rej >= 0.0)
        self.applyRejections.__wrapped__(
            self, 
            mask, 
            enforce=True, 
            original=True,
        )

    @pydantic.validate_call
    def updateOriginalAbsorption(
        self,
        coverage: Literal["partial", "full"] = "full",
    ) -> None:
        """
        Update the '_absorbed_pixels_original' and '_p_absorbed_original' 
        arrays based on the current '_absorbed_pixels' and '_p_absorbed' arrays.
        If '_y_smooth' has been calculated (i.e. not all NaN), the corresponding
        array is also updated.
        """
        if self._no_original:
            return
        
        if np.isnan(self._y_smooth).all():
            y_smooth = None
        else:
            y_smooth = np.interp(
                self._x_original,
                self._x,
                self._y_smooth,
                left=self._y_original[0],
                right=self._y_original[-1],
            )
    
        _abs = np.interp(
            self._x_original,
            self._x,
            self._absorbed_pixels.astype(np.float64),
            left=0.0,
            right=0.0,
        )
        mask = (_abs == 1.0) if coverage == "full" else (_abs >= 0.0)

        self.applyAbsorption.__wrapped__(
            self,
            mask,
            y_smooth=y_smooth,
            enforce=True,
            original=True,
        )
        self._p_absorbed_original = np.interp(
            self._x_original,
            self._x,
            self._p_absorbed,
            left=1.0,
            right=1.0,
        )

    def _update_original_emission_helper(self, data_type: str) -> None:
        model = None
        if getattr(self, data_type) is not None:
            model = getattr(self, data_type).getModel()
        self._update_emission_helper(model, True, data_type)

    def updateOriginalContinuumEmission(self) -> None:
        """
        Update the '_pl_original' array by evaluating the power law model on the 
        '_x_original' array.
        """
        self._update_original_emission_helper("pl")

    def updateOriginalIronEmission(self) -> None:
        """
        Update the '_fe_original' array by evaluating the iron emission model on 
        the '_x_original' array.
        """
        self._update_original_emission_helper("fe")

    def updateOriginalBalmerEmission(self) -> None:
        """
        Update the '_ba_original' array by evaluating the Balmer emission model 
        on the '_x_original' array.
        """
        self._update_original_emission_helper("ba")

    def updateOriginalHostGalaxyEmission(self) -> None:
        """
        Update the '_hg_original' array by evaluating the host galaxy emission 
        model on the '_x_original' array.
        """
        self._update_original_emission_helper("hg")

    def updateOriginalLinesEmission(self) -> None:
        """
        Update the '_em_original' array by evaluating the lines emission model 
        on the '_x_original' array.
        """
        self._update_original_emission_helper("em")

    @pydantic.validate_call
    def updateAllOriginals(
        self, 
        coverage: Literal["partial", "full"] = "full",
    ) -> None:
        if self._no_original:
            return
        
        # Rejections mask
        self.updateOriginalRejections.__wrapped__(self, coverage=coverage)
        # Absorption mask, absorption probabilities (, and smoothed signal)
        self.updateOriginalAbsorption.__wrapped__(self, coverage=coverage)

        # Model contributions
        self.updateOriginalContinuumEmission()
        self.updateOriginalIronEmission()
        self.updateOriginalBalmerEmission()
        self.updateOriginalHostGalaxyEmission()
        self.updateOriginalLinesEmission()

    @pydantic.validate_call
    def adoptFit(
        self,
        fit: ModelType,
        *,
        fit_info: OptimizeResultType | None = None,
        update_emission: bool = False,
    ) -> None:
        from scipy.optimize import OptimizeResult

        count = 0
        parts = get_model_parts(fit)
        for key in ("pl", "fe", "ba", "hg", "em"):
            model = parts[key]
            if model is None:
                continue

            if fit_info is None:
                finfo = None
            else:
                n_free = sum(get_free_params(model).values())
                sel = slice(count, count + n_free)
                finfo = OptimizeResult(
                    x=fit_info.x[sel],
                    success=fit_info.success,
                    message=fit_info.message,
                    status=fit_info.status,
                    fun=fit_info.fun,
                    jac=fit_info.jac[sel, sel],
                    nfev=fit_info.nfev,
                    njev=fit_info.njev,
                    nit=fit_info.get("nit", 0),
                    maxcv=fit_info.get("maxcv", 0),
                )
                count += n_free

            windows = self[key]
            windows.adoptFit.__wrapped__(
                windows,
                model,
                fit_info=finfo,
                update_emission=update_emission,
            )

    @pydantic.validate_call
    def applyFit(
        self,
        fit: ModelType,
        *,
        fit_info: OptimizeResultType | None = None,
        update_emission: bool = False,
    ) -> None:
        from scipy.optimize import OptimizeResult

        count = 0
        parts = get_model_parts(fit)
        for key in ("pl", "fe", "ba", "hg", "em"):
            model = parts[key]
            if model is None:
                continue

            if fit_info is None:
                finfo = None
            else:
                n_free = sum(get_free_params(model).values())
                sel = slice(count, count + n_free)

                jac = None \
                    if getattr(fit_info, "jac", None) is None \
                    else fit_info.jac[sel, sel]

                finfo = OptimizeResult(
                    x=fit_info.x[sel],
                    success=fit_info.success,
                    message=fit_info.message,
                    status=fit_info.status,
                    fun=fit_info.fun,
                    jac=jac,
                    nfev=fit_info.nfev,
                    njev=fit_info.njev,
                    nit=fit_info.get("nit", 0),
                    maxcv=fit_info.get("maxcv", 0),
                )
                count += n_free

            windows = self[key]
            windows.applyFit.__wrapped__(
                windows,
                model,
                fit_info=finfo,
                update_emission=update_emission,
            )

    def summariseContinuumFit(self):
        if self.continuum_windows is None:
            return None
        return self.continuum_windows.summariseContinuumFit()

    @validated_apply_info_to_method(subjects=("error", "nonlinear", "random"))
    def instantiateBootstrapper(
        self,
        *,
        original: bool = False,
        data_types: DataTypes | None = None,
        model_types: ModelTypes | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        shrink: bool | None = None,

        pool: PoolType | None = None,
        variant: Variant | None = None,
        bootstrap_type: BootstrapType | None = None,
        n_sigmas: float | None = None,
        vary_lines: VaryLines | None = None,
        iterations: int | None = None,
        replace_missing: bool | None = None,
        tqdm_disable: bool | None = None,
        tqdm_leave: bool | None = None,
        fitter_kwargs: FitterKwargs | None = None,
        random_state: RandomStateType | None = None,
    ):
        from .utils.bootstrapping import (
            args_and_kwargs_helper,
            validate_model_and_data,
        )

        self.summariseContinuumFit()

        if data_types is None:
            data_types = DataTypes({"all"})
            covered = False
        else:
            covered = True

        model_types = ModelTypes({"all"}) \
            if model_types is None \
            else model_types.copy()

        models: dict[str, ModelType] = {}
        for m_type in tuple(model_types):
            if getattr(self, m_type) is None:
                model_types.remove(m_type)
                continue

            model = getattr(self, m_type).getModel()
            if model is None:
                model_types.remove(m_type)
                continue
            models[m_type] = model

        for d_type in tuple(data_types):
            if getattr(self, d_type) is None:
                data_types.remove(d_type)
                continue

        validate_model_and_data(
            model_types=model_types,
            data_types=data_types,
            variant=variant,
        )

        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            original=original,
            data_types=data_types,
            covered=covered,
            mode="c",
            bg_flux=None,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=True,
            log_valid=False,
            shrink=shrink,
        )
        bootstrapper_cls, args, kwargs = args_and_kwargs_helper(
            masked_coords=masked_coords,
            info=self.info,
            variant=variant,
            cfit=self.summariseContinuumFit(),
            **models,
            fitter_kwargs=fitter_kwargs,
            iterations=iterations,
            random_state=random_state,
            bootstrap_type=bootstrap_type,
            n_sigmas=n_sigmas,
            vary_lines=vary_lines,
            pool=pool,
            replace_missing=replace_missing,
            tqdm_disable=tqdm_disable,
            tqdm_leave=tqdm_leave,
        )
        return bootstrapper_cls(*args, **kwargs)

    @validated_apply_info_to_method(subjects=("error", "random"))
    def instantiateNestedSampler(
        self,
        *,
        covered: bool = True,
        data_types: DataTypes | None = None,
        model_types: ModelTypes | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        pool: PoolType | None = None,
        iterations: int | None = None,
        replace_missing: bool | None = None,
        tqdm_disable: bool | None = None,
        tqdm_leave: bool | None = None,
        random_state: RandomStateType | None = None,
    ):
        """
        ...
        """   
        from quasar_errors.nested_sampling import UniformNestedSampler

        if data_types is None:
            data_types = DataTypes({"all"})
        if model_types is None:
            model_types = ModelTypes({"all"})

        return UniformNestedSampler\
            .from_spectrum(
                spectrum=self,
                covered=covered,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                data_types=data_types,
                model_types=model_types,
                pool=pool,
                iterations=iterations,
                random_state=random_state,
                replace_missing=replace_missing,
                tqdm_disable=tqdm_disable,
                tqdm_leave=tqdm_leave,
                logger=logger,
            )\
            .instantiateSampler()

    def quickplot(
        self, 
        lazy: bool = False, 
        **plotter_kwargs,
    ):
        """Basic plotting routine for 'Spectrum' classes.

        Creates and returns a 'QuickPlotter' instance, which is used to plot the 
        spectrum.

        Parameters
        ----------
        lazy : bool, optional
            If True, returns the 'QuickPlotter' instance without plotting. If 
            False, plots the spectrum and returns the 'QuickPlotter' instance. 
            Default is False.
        plotter_kwargs : dict
            See `quasar_decrypt.plotting.basics.QuickPlotter` for details.

        Returns
        -------
        plotter : QuickPlotter
            The 'QuickPlotter' instance used to plot the spectrum.
        """
        from quasar_decrypt.plotting.basics import QuickPlotter

        if "xlim" not in plotter_kwargs:
            plotter_kwargs['xlim'] = self.x_bounds
        if "original" in plotter_kwargs:
            plotter_kwargs["original"] &= not self._no_original

        plotter = QuickPlotter(self, **plotter_kwargs)
        return plotter if lazy else plotter.plot()

    def differenceplot(
        self, 
        mode: Literal['difference', 'residual'],
        lazy: bool = False, 
        **plotter_kwargs,
    ):
        """Basic plotting routine showing the data-model difference.

        Parameters
        ----------
        lazy : bool, optional
            If True, returns the plotter instance without plotting. If False, 
            plots the difference and returns the plotter instance. Default is 
            False.
        plotter_kwargs : dict
            See `quasar_decrypt.plotting.basics` for details.

        Returns
        -------
        plotter : DifferencePlotter | ResidualPlotter
            The 'DifferencePlotter' or 'ResidualPlotter' instance used to plot 
            the data-model difference or residuals.
        """
        from quasar_decrypt.plotting.basics import (
            DifferencePlotter,
            ResidualPlotter,
        )
        plotter_cls = DifferencePlotter \
            if mode == "difference" \
            else ResidualPlotter

        if "xlim" not in plotter_kwargs:
            plotter_kwargs['xlim'] = self.x_bounds
        if "original" in plotter_kwargs:
            plotter_kwargs["original"] &= not self._no_original
            
        plotter = plotter_cls(self, **plotter_kwargs)
        return plotter if lazy else plotter.plot()

    def templatesplot(
        self,
        mode: Literal["residual", "difference"],
        lazy: bool = False,
        **plotter_kwargs,
    ):
        """
        ...
        """
        from .plotting.windows_and_templates_plots import (
            AllTemplatesWithDifferencesPlot,
            AllTemplatesWithResidualsPlot,
        )
        plotter_cls = AllTemplatesWithDifferencesPlot \
            if mode == "difference" \
            else AllTemplatesWithResidualsPlot

        if "xlim" not in plotter_kwargs:
            plotter_kwargs['xlim'] = self.x_bounds
        if "original" in plotter_kwargs:
            plotter_kwargs["original"] &= not self._no_original

        plotter = plotter_cls(self.getModel(), self, **plotter_kwargs)
        return plotter if lazy else plotter.plot()
    
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
        pl_color: str | None = None,
        fe_color: str | None = None,
        ba_color: str | None = None,
        hg_color: str | None = None,
        em_color: str | None = None,
        sm_color: str | None = None,
        ab_color: str | None = None,
        xticks: tuple[float, float] | None = None,
        yticks: tuple[float, float] | None = None,
        zticks: tuple[float, float] | None = None,
        pticks: tuple[float, float] | None = None,
        logx: bool = False,
        logy: bool = False,
        logp: bool = True,
    ) -> tuple[Figure, list[Axes]]:
        from quasar_plotting import DEFAULT_COLORS, absorptionplot, get_coords
        
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
            xlabel=xlabel or "auto",
            ylabel=ylabel or "auto",
            zlabel=zlabel or "auto",
            plabel=plabel or "auto",
            xlim=xlim or "auto",
            ylim=ylim or "auto",
            zlim=zlim or "auto",
            plim=plim or "auto",
            pl_color=pl_color or DEFAULT_COLORS["pl"],
            fe_color=fe_color or DEFAULT_COLORS["fe"],
            ba_color=ba_color or DEFAULT_COLORS["ba"],
            hg_color=hg_color or DEFAULT_COLORS["hg"],
            em_color=em_color or DEFAULT_COLORS["em"],
            sm_color=sm_color or DEFAULT_COLORS["sm"],
            ab_color=ab_color or DEFAULT_COLORS["ab"],
            xticks=xticks or "auto",
            yticks=yticks or "auto",
            zticks=zticks or "auto",
            pticks=pticks or "auto",
            logx=logx,
            logy=logy,
            logp=logp,
        )