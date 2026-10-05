from collections.abc import Iterable
from dataclasses import field
from logging import getLogger
from typing import Any, ClassVar, Literal, Optional, Self, Union

from numpy import dot, inf
from pydantic.dataclasses import dataclass
from quasar_models.host import (
    HostGalaxyModel,
    HostGalaxyTemplate,
)
from quasar_models.host.io import convert_params_to_name
from quasar_models.modeling import Fitter, PrepareModel
from quasar_models.utils.astropy import get_free_params
from quasar_typing.astropy import FitInfo
from quasar_typing.bounds import AstropyBounds, CoordBounds
from quasar_typing.misc import BackgroundFlux
from quasar_typing.pathlib import AbsoluteFITSPath
from quasar_typing.scipy import OptimizeResult_
from quasar_utils.decorators import (
    validate_call,
    validated_apply_info_to_method,
)
from quasar_utils.setup import FitterKwargs

from ..plotting.windows_and_templates_plots import (
    HostTemplatesWithDifferencesPlot,
    HostTemplatesWithResidualsPlot,
    HostWindowsAndTemplatesWithDifferencesPlot,
    HostWindowsAndTemplatesWithResidualsPlot,
)
from ..plotting.windows_plots import (
    HostWindowsPlotWithDifferences,
    HostWindowsPlotWithResiduals,
)
from ..utils import _SpecWindowList
from .hwindow import HWindow

logger = getLogger(__name__)


@dataclass
class HostWindows(_SpecWindowList[HWindow]):
    templates: dict[str, HostGalaxyTemplate] = field(
        default_factory=dict, kw_only=True
    )
    models: dict[str, HostGalaxyModel] = field(
        default_factory=dict, kw_only=True
    )

    model: Optional[HostGalaxyModel] = field(default=None, init=False)
    fit: Optional[HostGalaxyModel] = field(default=None, init=False)
    fit_info: FitInfo | None = field(default=None, init=False)

    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({"all", "hg"})

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["templates"] = self.templates
        state["models"] = self.models
        state["model"] = self.model
        state["fit"] = self.fit
        state["fit_info"] = self.fit_info
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.templates = state.get("templates", {})
        self.models = state.get("models", {})
        self.model = state.get("model", None)
        self.fit = state.get("fit", None)
        self.fit_info = state.get("fit_info", None)

    @property
    def sample(self) -> None:
        raise NotImplementedError

    @validated_apply_info_to_method(
        subjects=("host",),
        specific_kwargs={"windows"},
    )
    def populate(
        self,
        *,
        windows: Iterable[CoordBounds] | None = None,
    ) -> Self:
        for x_bounds in windows:
            hwindow = HWindow.create.__wrapped__(
                HWindow,
                spectrum=self.spectrum,
                x_bounds=x_bounds,
                get_mask=None,
            )
            if hwindow.size > 0:
                self.append(hwindow)
        return self

    @validated_apply_info_to_method(subjects=("host", "nonlinear"))
    def __call__(
        self,
        *,
        template_model: Optional[HostGalaxyModel] = None,
        original: bool = False,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        covered: bool = False,
        shrink: float | None = None,
        x_min: float | None = None,
        template_files: list[str | AbsoluteFITSPath] | None = None,
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
    ) -> bool:
        if bg_flux is None:
            bg_flux = self.default_bg

        x_max = self.spectrum.x[-1]
        if x_max <= x_min:
            # Spectrum is too blue -> Do not fit host galaxy emission
            self.templates.clear()
            self.models.clear()
            self.model = None
            self.fit = None
            self.fit_info = None
            self.updateHostGalaxyEmission.__wrapped__(self)
            return False

        if template_model is not None:
            self.applyFit.__wrapped__(
                self,
                template_model,
            )
        else:
            self.loadHostGalaxyTemplates.__wrapped__(
                self,
                sources=sources,
                ages=ages,
                template_files=template_files,
            )
            self.instantiateModels.__wrapped__(
                self,
                flux=flux,
                fwhm=fwhm,
                allow_interp_fitting=allow_interp_fitting,
                fixed=fixed,
                flux_bounds=flux_bounds,
                fwhm_bounds=fwhm_bounds,
            )

        if raster:
            self.getRasterFit.__wrapped__(
                self,
                without_absorption=without_absorption,
                without_rejections=without_rejections,
                covered=covered,
                bg_flux=bg_flux,
            )

        if fine_tune:
            self.performFineTuning.__wrapped__(
                self,
                original=original,
                covered=covered,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                bg_flux=bg_flux,
                shrink=shrink,
                fitter_kwargs=fitter_kwargs,
            )

        if len(self) == 0:
            # No windows -> use default template
            template_name = convert_params_to_name(default_source, default_age)
            fit = self.chooseModel.__wrapped__(self, template_name)
            self.applyFit.__wrapped__(
                self,
                fit,
                update_emission=True,
            )

        return True

    @validate_call
    def loadHostGalaxyTemplate(
        self,
        *,
        source: Literal["bc2003"] | None = None,
        age: float | None = None,
        template_file: str | AbsoluteFITSPath | None = None,
    ) -> HostGalaxyTemplate:
        """
        Loads a HostGalaxyTemplate instance.
        """
        if template_file is None:
            if source is None:
                msg = "Must specify a 'source' when 'template_file' is not provided!"
                logger.critical(msg)
                raise ValueError(msg)
            if age is None:
                msg = "Must speficy an 'age' when 'template_file' is not provided!"
                logger.critical(msg)
                raise ValueError(msg)

            template = HostGalaxyTemplate.load_from_cache(
                name=source,
                age=age,
                info=self.info,
            )
        else:
            template = HostGalaxyTemplate.load(
                path=template_file,
                info=self.info,
            )

        if not template.is_logspace:
            template = template.createLogspace(
                sigma_res=self.info.loading.sigma_res,
                xr=self._x,
                keep_x=True,
            )

        return template

    @validated_apply_info_to_method(subjects=("balmer",))
    def loadHostGalaxyTemplates(
        self,
        *,
        sources: Iterable[Literal["bc2003"]] | None = None,
        ages: Iterable[int] | None = None,
        template_files: Iterable[str | AbsoluteFITSPath] | None = None,
    ) -> dict[str, HostGalaxyTemplate]:
        """
        Loads all HostGalaxyTemplate instances and adds them to the
        'templates' dict attribute.
        """
        self.templates.clear()

        if not template_files and not (sources and ages):
            msg = (
                "Must specify either 'template_files' or both 'sources' and "
                "'ages' to load HostGalaxyTemplates!"
            )
            logger.critical(msg)
            raise ValueError(msg)

        if template_files:
            for template_file in template_files:
                template = self.loadHostGalaxyTemplate.__wrapped__(
                    self,
                    template_file=template_file,
                )
                key = convert_params_to_name(template.name, template.age)

                if key in self.templates:
                    msg = f"Duplicate HostGalaxyTemplate: '{template.name}'."
                    logger.info(msg)
                    continue

                self.templates[key] = template

        if sources and ages:
            for source, age in zip(sources, ages):
                key = convert_params_to_name(source, age)
                template = self.loadHostGalaxyTemplate.__wrapped__(
                    self,
                    source=source,
                    age=age,
                )

                if key in self.templates:
                    msg = f"Duplicate HostGalaxyTemplate: '{template.name}'."
                    logger.info(msg)
                    continue

                self.templates[key] = template

        return self.templates

    @validated_apply_info_to_method(subjects=("balmer",))
    def instantiateModels(
        self,
        *,
        flux: float | None = None,
        fwhm: float | None = None,
        allow_interp_fitting: bool | None = None,
        fixed: dict[str, bool] | None = None,
        flux_bounds: AstropyBounds | None = None,
        fwhm_bounds: AstropyBounds | None = None,
    ) -> dict[str, HostGalaxyModel]:
        """
        Uses all available HostGalaxyTemplate instances to create corresponding
        HostGalaxyModel instances, which are added to the 'models' dict
        attribute.
        """
        self.models.clear()
        for name, template in self.templates.items():
            self.models[name] = HostGalaxyModel.create(
                flux,
                fwhm,
                template=template,
                allow_interp_fitting=allow_interp_fitting,
                name="host_galaxy",
                flux_bounds=flux_bounds,
                fwhm_bounds=fwhm_bounds,
                flux_fixed=fixed.get("flux", False),
                fwhm_fixed=fixed.get("fwhm", True), # Default to True
            )

        return self.models

    @validate_call
    def chooseModel(self, name: str) -> HostGalaxyModel:
        """
        Choose a HostGalaxyModel from the 'models' dict by name and set it as
        the current model.
        """
        if name not in self.models:
            msg = f"No HostGalaxyModel named '{name}' found in 'models'!"
            logger.critical(msg)
            raise ValueError(msg)
        elif self.model is not None and name == self.model.name:
            msg = f"HostGalaxyModel '{name}' is already the current model!"
            logger.debug(msg)
            return self.model

        self.model = self.models[name]
        return self.model

    @validate_call
    def getRasterFit(
        self,
        *,
        without_absorption: bool = False,
        without_rejections: bool = False,
        covered: bool = False,
        bg_flux: BackgroundFlux | None = None,
    ) -> Self:
        """
        Performs a raster fit on the Balmer pseudo-continuum model.
        """
        if self.models:
            if bg_flux is None:
                bg_flux = self.default_bg

            if covered and self.is_empty:
                msg = f"Setting {covered=} to False due to missing "\
                    "host windows!"

                logger.info(msg)
                covered = False

            masked_coords = self.getMaskedCoords.__wrapped__(
                self,
                original=False,
                mode="c",  # c: continuum
                covered=covered,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                valid=True,
                log_valid=False,
                bg_flux=bg_flux,
                shrink=None,
            )
            w = 1 / masked_coords.dy

            chi2s: dict[str, float] = {}
            for name, model in self.models.items():
                model.rasterFit.__wrapped__(
                    model,
                    masked_coords.x,
                    masked_coords.y,
                    masked_coords.dy,
                    inplace=True,
                )
                z = (masked_coords.y - model(masked_coords.x)) * w
                chi2s[name] = dot(z, z)

            best_model_name = min(chi2s.keys(), key=chi2s.get)
            best_chi2 = chi2s[best_model_name]

            msg = f"Best-fit HostGalaxyModel is '{best_model_name}' with chi2={best_chi2:.2f}."
            logger.debug(msg)

            self.chooseModel.__wrapped__(
                self,
                best_model_name,
            )
            self.applyFit.__wrapped__(
                self,
                self.model,
                update_emission=True,
            )

        return self

    @validated_apply_info_to_method(subjects=("nonlinear",))
    def performFineTuning(
        self,
        *,
        original: bool = False,
        only_model: bool = False,
        covered: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        bg_flux: BackgroundFlux | None = None,
        shrink: bool | None = None,
        fitter_kwargs: FitterKwargs | None = None,
    ) -> Self:
        logger.debug("Performing fine tuning:")

        original = self._update_original_flag(
            original=original,
            log=True,
            prefix=">>> ",
        )
        if not self.models:
            msg = "No HostGalaxyModel instances available for fine-tuning!"
            logger.critical(">>> " + msg)
            raise ValueError(msg)

        if only_model and self.model is None:
            logger.debug(
                ">>> Setting `only_model` to False due to missing `self.model`."
            )
            only_model = False

        if covered and self.is_empty:
            logger.debug(
                ">>> Setting `covered` to False due to missing host windows."
            )
            covered = False

        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            original=original,
            mode="c",  # c: contiguous
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=True,
            log_valid=False,
            bg_flux=bg_flux,
            shrink=shrink,
        )

        _model = next(iter(self.models.values()))
        n_pix = masked_coords.x.size
        n_free_params = sum(get_free_params(_model).values())
        if n_pix <= n_free_params:
            msg = "Cancelling fine-tuning due to insufficient number of data "\
                f"points: {n_pix=} <= {n_free_params=}"
            logger.critical(">>> " + msg)
            raise ValueError(msg)

        fit_infos: dict[str, OptimizeResult_ | None] = {}
        chi2s: dict[str, float] = {}

        was_successful_once = False
        for name, model in self.models.items():
            fitter = Fitter()
            preparer = PrepareModel.fromCoords(masked_coords, model)
            data = preparer.createDataDict(masked_coords)
            try:
                with preparer as (seq_model, where):
                    _ = fitter(
                        seq_model,
                        data["x"],
                        data["y"],
                        dy=data["dy"],
                        where=where,
                        get_model=True,
                        inplace=True,
                        **fitter_kwargs,
                    )
                    fit_infos[name] = fitter.fit_info
                    chi2s[name] = (fit_infos[name].fun ** 2).mean()
                logger.debug(f">>> Successfully fitted model: {name}")
                was_successful_once = True
            except ValueError as _:
                logger.debug(f">>> Failed fitting model: {name}")
                fit_infos[name] = None
                chi2s[name] = inf

        if not was_successful_once:
            logger.warning(">>> Failed fitting all models.")
            return self

        template_name: str = min(chi2s, key=chi2s.get)
        chi2 = chi2s[template_name]

        logger.debug(
            f">>> Best-fit template: {template_name} with chi2={chi2:.1f}"
        )

        fit = self.chooseModel.__wrapped__(self, template_name)
        self.applyFit.__wrapped__(
            self, 
            fit, 
            fit_info=fit_infos[template_name], 
            update_emission=True,
        )
        return self

    def getModel(self) -> Optional[HostGalaxyModel]:
        """
        Retrieves the Balmer pseudo-continuum model if available.

        Notes
        -----
        If a fitted Balmer pseudo-continuum model is available, it is
        returned. Otherwise an instantiated model is returned, if available.
        """
        return self.fit or self.model

    @validate_call
    def adoptFit(
        self,
        fit: HostGalaxyModel,
        *,
        fit_info: FitInfo | None = None,
        update_emission: bool = False,
    ) -> Self:
        """
        Identical to 'applyFit'.
        """
        return self.applyFit.__wrapped__(
            self,
            fit,
            fit_info=fit_info,
            update_emission=update_emission,
        )

    @validate_call
    def applyFit(
        self,
        fit: HostGalaxyModel,
        *,
        fit_info: FitInfo | None = None,
        update_emission: bool = False,
    ) -> Self:
        """
        Applies the given Host Galaxy fit.
        """
        self.fit_info = fit_info
        self.fit = fit

        if update_emission:
            self.updateHostGalaxyEmission.__wrapped__(self, fit)

        return self

    def windowsplot(
        self,
        mode: Literal["residual", "difference"],
        lazy: bool = False,
        **plotter_kwargs,
    ) -> HostWindowsPlotWithResiduals | HostWindowsPlotWithDifferences:
        """
        ...
        """
        args = (self.getModel(), self.window_bounds, self.spectrum)
        cls = HostWindowsPlotWithResiduals \
            if mode == "residual" \
            else HostWindowsPlotWithDifferences

        if 'title_prefix' not in plotter_kwargs:
            plotter_kwargs['title_prefix'] = "HostWindows of: "
        if 'title' not in plotter_kwargs:
            plotter_kwargs['title'] = self.spectrum.title
        if 'xlim' not in plotter_kwargs:
            plotter_kwargs['xlim'] = self.spectrum.x_bounds

        plotter = cls(*args, **plotter_kwargs)
        return plotter if lazy else plotter.plot()

    def templatesplot(
        self,
        mode: Literal["residual", "difference"],
        with_windows: bool = True,
        lazy: bool = False,
        **plotter_kwargs,
    ) -> Union[
        HostWindowsAndTemplatesWithResidualsPlot,
        HostWindowsAndTemplatesWithDifferencesPlot,
        HostTemplatesWithResidualsPlot,
        HostTemplatesWithDifferencesPlot,
    ]:
        """
        ...
        """
        if with_windows:
            args = (self.getModel(), self.window_bounds, self.spectrum)
            cls = HostWindowsAndTemplatesWithResidualsPlot \
                if mode == "residual" \
                else HostWindowsAndTemplatesWithDifferencesPlot
        else:
            args = (self.getModel(), self.spectrum)
            cls = HostTemplatesWithResidualsPlot \
                if mode == "residual" \
                else HostTemplatesWithDifferencesPlot

        if 'title_prefix' not in plotter_kwargs:
            plotter_kwargs['title_prefix'] = "HostTemplates/Windows of: "
        if 'title' not in plotter_kwargs:
            plotter_kwargs['title'] = self.spectrum.title
        if 'xlim' not in plotter_kwargs:
            plotter_kwargs['xlim'] = self.spectrum.x_bounds

        plotter = cls(*args, **plotter_kwargs)
        return plotter if lazy else plotter.plot()