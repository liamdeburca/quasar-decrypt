from collections.abc import Iterable
from dataclasses import field
from itertools import repeat
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, Self, Union

import pydantic
from numpy import isfinite
from quasar_models.iron import IronModel, IronTemplate
from quasar_typing.astropy import CompoundModelType
from quasar_typing.bounds import AstropyBounds, CoordBounds
from quasar_typing.numpy import FloatVector
from quasar_typing.pathlib import FITSPath
from quasar_typing.scipy import OptimizeResultType
from quasar_typing.string_selections import BackgroundFlux
from quasar_utils.decorators import validated_apply_info_to_method
from quasar_utils.setup import FitterKwargs

from quasar_decrypt.utils import _SpecWindowList

from .iwindow import IWindow

logger = getLogger(__name__)


@pydantic.dataclasses.dataclass
class IronWindows(_SpecWindowList[IWindow]):
    templates: dict[str, IronTemplate] = field(
        default_factory=dict, 
        kw_only=True,
    )
    template_models: dict[str, IronModel] = field(
        default_factory=dict, 
        kw_only=True,
    )
    fit_info: OptimizeResultType | None = field(
        default=None, 
        kw_only=True,
    )

    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({"all", "fe"})

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["templates"] = self.templates
        state["template_models"] = self.template_models
        state["fit_info"] = self.fit_info
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.templates = state.get("templates", {})
        self.template_models = state.get("template_models", {})
        self.fit_info = state.get("fit_info", None)

    @property
    def sample(self):
        from quasar_errors.model_samples import IronSampleList
        if (model := self.getModel()) is None:
            return None
        return IronSampleList.fromIronModels(model)

    @validated_apply_info_to_method(
        subjects=("iron",),
        specific_kwargs={"windows"},
    )
    def populate(
        self,
        *,
        windows: Iterable[CoordBounds] | None = None,
    ) -> Self:
        for x_bounds in windows:
            iwindow = IWindow.create.__wrapped__(
                IWindow,
                spectrum=self.spectrum,
                x_bounds=x_bounds,
                get_mask=None,
            )
            if iwindow.size > 0:
                self.append(iwindow)
        return self

    @validated_apply_info_to_method(subjects=("iron", "nonlinear"))
    def __call__(
        self,
        *,
        original: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        template_model: Union[IronModel, CompoundModelType[IronModel], None] = None,
        bg_flux: BackgroundFlux | None = None,
        shrink: float | None = None,
        template_files: list[FITSPath | str] | None = None,
        resample: bool | None = None,
        split: FloatVector | None = None,
        fwhm: FloatVector | None = None,
        bias: list[Literal["left", "right"]] | None = None,
        ratio: FloatVector | None = None,
        scale: float | None = None,
        tie_similar: bool | None = None,
        raster: bool | None = None,
        allow_interp_fitting: bool | None = None,
        flux_bounds: AstropyBounds | None = None,
        fwhm_bounds: AstropyBounds | None = None,
        fine_tune: bool | None = None,
        fitter_kwargs: FitterKwargs | None = None,
    ) -> bool:
        if bg_flux is None:
            bg_flux = self.default_bg

        if template_model is not None:
            self.applyFit.__wrapped__(
                self,
                template_model,
                update_emission=True,
            )
            self.performFineTuning.__wrapped__(
                self,
                bg_flux=bg_flux,
                covered=not self.is_empty,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                fitter_kwargs=fitter_kwargs,
            )
        else:
            self.loadTemplates.__wrapped__(
                self,
                template_files=template_files,
                resample=resample,
                split=split,
                fwhm=fwhm,
                bias=bias,
                ratio=ratio,
                scale=scale,
                allow_interp_fitting=allow_interp_fitting,
                flux_bounds=flux_bounds,
                fwhm_bounds=fwhm_bounds,
            )
            if tie_similar:
                self.tieSimilarModels()

            if raster:
                self.getRasterFit.__wrapped__(
                    self,
                    bg_flux=bg_flux,
                    covered=not self.is_empty,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                )

            if fine_tune:
                self.performFineTuning.__wrapped__(
                    self,
                    original=original,
                    bg_flux=bg_flux,
                    covered=not self.is_empty,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                    shrink=shrink,
                    fitter_kwargs=fitter_kwargs,
                )

        return True

    @validated_apply_info_to_method(subjects=("iron",))
    def loadTemplates(
        self,
        *,
        template_files: list[FITSPath | str] | None = None,
        resample: bool | None = None,
        split: FloatVector | None = None,
        fwhm: FloatVector | None = None,
        bias: list[Literal["left", "right"]] | None = None,
        ratio: FloatVector | None = None,
        scale: float | None = None,
        flux_bounds: AstropyBounds | None = None,
        fwhm_bounds: AstropyBounds | None = None,
        split_bounds: AstropyBounds | None = None,
        left_bounds: AstropyBounds | None = None,
        right_bounds: AstropyBounds | None = None,
        allow_interp_fitting: bool | None = None,
    ) -> Self:
        """
        Loads all templates:
        >   which are specified by the user.
        >   which are covered by the iron windows.

        Should adapt the templates to the current spectrum covered by the
        respective templates.
        """
        from scipy.ndimage import binary_fill_holes

        if template_files is None:
            f = self.info.iron.or_default(locals())
            template_files = list(map(Path, f("template_files")))
            resample = f("resample")
            split = f("split")
            fwhm = f("fwhm")
            bias = f("bias")
            ratio = f("ratio")
            scale = f("scale")
            allow_interp_fitting = f("allow_interp_fitting")
        else:
            n = len(template_files)
            if split is None:
                split = repeat(-1, n)
            if bias is None:
                bias = repeat("right", n)
            if ratio is None:
                ratio = repeat(1.0, n)

            resample = False if fwhm is None else (resample or False)

        self.templates.clear()
        self.template_models.clear()

        mask = isfinite(self._x)

        if self.is_empty:
            logger.debug(
                "IronWindows is empty: using entire wavelength array to "
                "check coverages of IronTemplates."
            )
        else:
            mask &= self.mask

        if mask.sum() < 2:
            msg = f"Not enough valid pixels ({mask.sum()}) available to check coverage of any IronTemplate!"
            logger.info(msg)
            return self

        x = self._x[mask]
        for template_file, s, b, r in zip(
            template_files,
            split,
            bias,
            ratio,
        ):
            logger.debug(f"Loading IronTemplate from file: {template_file!s}")
            template = IronTemplate.load_from_cache(
                name=template_file,
                info=self.info,
            ).copy(with_matrices=True)
            logger.debug(f">>> Template name: {template.name}")

            # Check template-coverage
            tx = template.x[binary_fill_holes(template.data[0] > 0)]
            template_lb = tx[0]
            template_ub = tx[-1]
            logger.debug(
                f">>> Template bounds: "
                f"lb={template_lb:.1f}, ub={template_ub:.1f}"
            )

            lb = x[0]
            ub = x[-1]
            if (ub <= template_lb):
                logger.debug(
                    ">>> IronTemplate is entirely redwards of the spectrum: "
                    f"{ub=:.1f} <= {template_lb=:.1f}."
                )
                continue
            elif (template_ub <= lb):
                logger.debug(
                    ">>> IronTemplate is entirely bluewards of the spectrum: "
                    f"{template_ub=:.1f} <= {lb=:.1f}."
                )
                continue

            logger.debug(
                f">>> IronTemplate ({template.name}) covers the spectrum: "
                f"({lb=:.1f}, {ub=:.1f})."
            )
            if not template.is_logspace:
                logger.debug(">>> Transforming to logspace!")

                tx_wide = template.x[binary_fill_holes(template.data[-1] > 0)]
                mask = (
                    isfinite(self._x)
                    & (tx_wide[0] <= self._x)
                    & (self._x <= tx_wide[-1])
                )

                template = template.createLogspace(
                    sigma_res=self.info.loading.sigma_res,
                    keep_x=True,
                )

            # Resample template if necessary
            if resample:
                template.resample(fwhm)

            if template.x[0] < s < template.x[-1]:
                logger.debug(
                    f">>> Applying split: split={s:.1f}, bias={b}, ratio={r:.2f}."
                )

                template.applySplit(
                    split=s,
                    left=1.0 if b == "left" else r,
                    right=r if b == "left" else 1.0,
                    scale=scale,
                    inplace=True,
                )

            self.templates[template.name] = template

            model = IronModel.create(
                1.0, template.fwhm[0], #? Values are automatically adapted to bounds
                scale=self.info.iron.scale,
                template=template,
                split=s,
                left=1.0,
                right=1.0,
                allow_interp_fitting=allow_interp_fitting,
                flux_bounds=flux_bounds,
                fwhm_bounds=fwhm_bounds,
                split_bounds=split_bounds,
                left_bounds=left_bounds,
                right_bounds=right_bounds,
            )
            self.template_models[model.name] = model

        return self

    def tieSimilarModels(self) -> None:
        """
        Tie the `fwhm` parameters of `IronModel` instances whose templates' 
        names are similar, i.e. they share the same prefix (`_` as a separator).
        """
        from collections import Counter

        from quasar_models.modeling import LinearTie

        # Get prefixes for models w/o fixed fwhm parameters
        prefix_counts = Counter(
            name.split("_")[0]
            for name, model in self.template_models.items()
            if not model.fwhm.fixed
        )
        for prefix in filter(
            lambda p: prefix_counts[p] > 1,
            prefix_counts.keys()
        ):
            models: list[IronModel] = sorted(
                (
                    model
                    for name, model in self.template_models.items()
                    if name.startswith(prefix) and not model.fwhm.fixed
                ),
                key=lambda m: m.sorting_key,
            )
            assert len(models) == prefix_counts[prefix]

            master_model = models.pop(0)
            for model in models:
                model.fwhm.tied = LinearTie(
                    a=1.0, 
                    b=0.0,
                    model_name=master_model.name, 
                    parameter_name="fwhm",
                )

    @validated_apply_info_to_method(subjects=("iron",))
    def getRasterFit(
        self,
        *,
        covered: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> Self:
        """
        Fits the available templates using rasterisation.
        """
        if covered and self.is_empty:
            msg = f"Setting {covered=} to False due to empty IronWindows!"
            logger.info(msg)
            covered = False

        if bg_flux is None:
            bg_flux = self.default_bg

        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            original=False,
            mode="c",  # c: contiguous
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=True,
            log_valid=False,
            bg_flux=bg_flux,
            shrink=None,
        )

        n_pix = masked_coords.size
        if n_pix <= 2:
            msg = (
                "cancelling raster fit due to insufficient no. of data "
                f"points (n_pix={n_pix} <= n_free_params=2)!"
            )
            logger.info(msg)
            return self

        for model in (
            model
            for model in self.template_models.values()
            if not model.fwhm.tied
        ):
            # Only fit models whose FWHM is not tied, as would be the case if 
            # `tieSimilarModels` had been called

            try:
                model.rasterFit(
                    masked_coords.x,
                    masked_coords.y,
                    masked_coords.dy,
                    inplace=True,
                )
            except ValueError as exc:
                logger.info(f"Raster fit failed due to ValueError: {exc}")
                return self
                
        self.updateIronEmission(self.getModel())
        return self

    def getModel(
        self,
    ) -> Optional[Union[IronModel, CompoundModelType[IronModel]]]:
        """
        Combines the available templates into a single (Split) TemplateModel or
        AstroPy compound model.
        """
        submodels = list(self.template_models.values())
        return sum(submodels[1:], start=submodels[0]) if submodels else None

    @pydantic.validate_call
    def adoptFit(
        self,
        fit: Union[IronModel, CompoundModelType[IronModel]],
        *,
        fit_info: OptimizeResultType | None = None,
        update_emission: bool = False,
    ) -> Self:
        self.fit_info = fit_info

        self.templates.clear()
        self.template_models.clear()

        for template_model in (fit,) if fit.n_submodels == 1 else fit:
            self.templates[template_model.name] = template_model.template
            self.template_models[template_model.name] = template_model

        if update_emission:
            self.updateIronEmission.__wrapped__(self, fit)

            for iwindow in filter(lambda w: w._y_fe is not self._y_fe, self):
                iwindow.updateIronEmission.__wrapped__(iwindow, fit)

        return self

    @pydantic.validate_call
    def applyFit(
        self,
        fit: Union[IronModel, CompoundModelType[IronModel], None],
        *,
        fit_info: OptimizeResultType | None = None,
        update_emission: bool = False,
    ) -> Self:
        self.fit_info = fit_info
        if fit is not None:
            for template_model in (fit,) if fit.n_submodels == 1 else fit:
                self.template_models[template_model.name] = template_model

        if update_emission:
            self.updateIronEmission.__wrapped__(self, fit)

        return self

    @validated_apply_info_to_method(subjects=("nonlinear",))
    def performFineTuning(
        self,
        *,
        original: bool = False,
        covered: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        bg_flux: BackgroundFlux | None = None,
        shrink: bool | None = None,
        fitter_kwargs: FitterKwargs | None = None,
    ) -> Self:
        """
        Fits the available templates using a nonlinear optimiser.
        """
        from quasar_models.modeling import Fitter, PrepareModel
        from quasar_models.utils.astropy import _get_free_params

        logger.debug("Performing fine tuning:")

        original = self._update_original_flag(
            original=original,
            log=True,
            prefix=">>> ",
        )

        if bg_flux is None:
            bg_flux = self.default_bg

        if covered and self.is_empty:
            logger.debug(
                ">>> Setting `covered` to False due to missing iron windows."
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

        model = self.getModel()
        if model is None:
            msg = "No IronModels available!"
            logger.critical(">>> " + msg)
            raise ValueError(msg)

        n_pix = masked_coords.size
        n_free_params = sum(_get_free_params(model).values())

        if n_pix <= n_free_params:
            msg = "Cancelling fine-tuning due to insufficient number of data "\
                f"points: {n_pix=} <= {n_free_params=}"
            logger.critical(">>> " + msg)
            raise ValueError(msg)

        try:
            preparer = PrepareModel.fromCoords(masked_coords, model)
            data = preparer.createDataDict(masked_coords)

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
                    **(fitter_kwargs or {}),
                )
                fit_info = fitter.fit_info

            logger.debug(">>> Successfully fitted model!")
        except pydantic.ValidationError as e:
            logger.critical(
                f">>> Failed fitting model due to validation error: {e}"
            )
            return self

        self.applyFit.__wrapped__(
            self,
            fit,
            fit_info=fit_info,
            update_emission=True,
        )
        return self

    def windowsplot(
        self,
        mode: Literal["residual", "difference"],
        model: Union[IronModel, CompoundModelType[IronModel], None] = None,
        lazy: bool = False,
        **plotter_kwargs,
    ):
        """
        ...
        """
        from quasar_decrypt.plotting.windows_plots import (
            IronWindowsPlotWithDifferences,
            IronWindowsPlotWithResiduals,
        )
        if model is None:
            model = self.getModel()

        args = (model, self.window_bounds, self.spectrum)
        cls = IronWindowsPlotWithResiduals \
            if mode == "residual" \
            else IronWindowsPlotWithDifferences

        return self._plot_helper(
            plotter_cls=cls,
            lazy=lazy,
            args=args,
            plotter_kwargs=plotter_kwargs,
        )

    def templatesplot(
        self,
        mode: Literal["residual", "difference"],
        model: Union[IronModel, CompoundModelType[IronModel], None] = None,
        with_windows: bool = True,
        lazy: bool = False,
        **plotter_kwargs,
    ):
        """
        ...
        """
        from quasar_decrypt.plotting.windows_and_templates_plots import (
            IronTemplatesWithDifferencesPlot,
            IronTemplatesWithResidualsPlot,
            IronWindowsAndTemplatesWithDifferencesPlot,
            IronWindowsAndTemplatesWithResidualsPlot,
        )
        if model is None:
            model = self.getModel()
            
        if with_windows:
            args = (model, self.window_bounds, self.spectrum)
            cls = IronWindowsAndTemplatesWithResidualsPlot \
                if mode == "residual" \
                else IronWindowsAndTemplatesWithDifferencesPlot
        else:
            args = (model, self.spectrum)
            cls = IronTemplatesWithResidualsPlot \
                if mode == "residual" \
                else IronTemplatesWithDifferencesPlot

        return self._plot_helper(
            plotter_cls=cls,
            lazy=lazy,
            args=args,
            plotter_kwargs=plotter_kwargs,
        )