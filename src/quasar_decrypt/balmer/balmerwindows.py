from collections.abc import Iterable
from dataclasses import field
from logging import getLogger
from typing import Any, ClassVar, Literal, Optional, Self

import pydantic
from quasar_models.balmer import (
    BalmerContinuumTemplate,
    BalmerModel,
    BalmerSeriesTemplate,
)
from quasar_typing.bounds import AstropyBounds, CoordBounds
from quasar_typing.pathlib import FITSPath
from quasar_typing.scipy import OptimizeResultType
from quasar_typing.string_selections import BackgroundFlux
from quasar_utils.decorators import validated_apply_info_to_method
from quasar_utils.setup import FitterKwargs

from quasar_decrypt.utils import _SpecWindowList

from .bwindow import BWindow

logger = getLogger(__name__)


@pydantic.dataclasses.dataclass
class BalmerWindows(_SpecWindowList[BWindow]):
    model: Optional[BalmerModel] = field(default=None, init=False)
    fit: Optional[BalmerModel] = field(default=None, init=False)
    fit_info: OptimizeResultType | None = field(default=None, init=False)

    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({"all", "ba"})

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["model"] = self.model
        state["fit"] = self.fit
        state["fit_info"] = self.fit_info
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.model = state.get("model", None)
        self.fit = state.get("fit", None)
        self.fit_info = state.get("fit_info", None)

    @property
    def sample(self):
        from quasar_errors.model_samples import BalmerSample
        if (model := self.getModel()) is None:
            return None
        return BalmerSample.fromBalmerModel(model)

    @validated_apply_info_to_method(
        subjects=("balmer",),
        specific_kwargs={"windows"},
    )
    def populate(
        self,
        *,
        windows: Iterable[CoordBounds] | None = None,
    ) -> Self:
        for x_bounds in windows:
            bwindow = BWindow.create.__wrapped__(
                BWindow,
                spectrum=self.spectrum,
                x_bounds=x_bounds,
                get_mask=None, # Create a new get_mask for each window
            )
            if bwindow.size > 0:
                self.append(bwindow)
        return self

    @validated_apply_info_to_method(subjects=("balmer", "nonlinear"))
    def __call__(
        self,
        *,
        template_model: Optional[BalmerModel] = None,
        original: bool = False,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        covered: bool = True,
        shrink: float | None = None,
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
    ) -> bool:
        if bg_flux is None:
            bg_flux = self.default_bg

        if template_model is not None:
            self.applyFit.__wrapped__(
                self,
                template_model,
                update_emission=not (raster or fine_tune),
            )
        else:
            self.instantiateModel.__wrapped__(
                self,
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
            )
            self.checkModelCoverage.__wrapped__(
                self,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                min_fittable_ratio=min_fittable_ratio,
                min_fittable_total=min_fittable_total,
                covered=False,
            )
            if self.model is None:
                return False

        if raster:
            self.getRasterFit.__wrapped__(
                self,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                covered=covered,
                bg_flux=bg_flux,
            )
        if fine_tune:
            self.performFineTuning.__wrapped__(
                self,
                original=original,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                covered=covered,
                bg_flux=bg_flux,
                shrink=shrink,
                fitter_kwargs=fitter_kwargs,
            )

        return True

    @pydantic.validate_call
    def loadBalmerSeriesTemplate(
        self,
        *,
        source: Literal["sh1995"] | None = None,
        temp: float | None = None,
        dens: float | None = None,
        n_u_min: int | None = None,
        n_u_max: int | None = None,
        template_file: str | FITSPath | None = None,
    ) -> BalmerSeriesTemplate:
        """
        Loads a BalmerSeriesTemplate instance.
        """
        if template_file is None:
            if source is None:
                msg = (
                    "Must specify 'source' if 'template_file' is not provided!"
                )
                logger.critical(msg)
                raise ValueError(msg)
            if temp is None:
                msg = "Must specify 'temp' if 'template_file' is not provided!"
                logger.critical(msg)
                raise ValueError(msg)
            if dens is None:
                msg = "Must specify 'dens' if 'template_file' is not provided!"
                logger.critical(msg)
                raise ValueError(msg)
            if n_u_min is None:
                msg = "Must specify 'n_u_min' if 'template_file' is not provided!"
                logger.critical(msg)
                raise ValueError(msg)
            if n_u_max is None:
                msg = "Must specify 'n_u_max' if 'template_file' is not provided!"
                logger.critical(msg)
                raise ValueError(msg)

            template = BalmerSeriesTemplate.load_from_cache(
                name=source,
                temp=temp,
                dens=dens,
                n_u_range=(n_u_min, n_u_max),
                info=self.info,
            )
        else:
            template = BalmerSeriesTemplate.load(
                path=template_file,
                info=self.info,
            )

        return template

    @pydantic.validate_call
    def loadBalmerContinuumTemplate(
        self,
        *,
        temp: float | None = None,
        tau: float | None = None,
        scale: float | None = None,
        template_file: str | FITSPath | None = None,
    ) -> BalmerContinuumTemplate:
        """
        Loads a BalmerContinuumTemplate instance.
        """
        if template_file is None:
            if temp is None:
                msg = "Must specify 'temp' if 'path' is not provided!"
                logger.critical(msg)
                raise ValueError(msg)
            if tau is None:
                msg = "Must specify 'tau' if 'path' is not provided!"
                logger.critical(msg)
                raise ValueError(msg)
            if scale is None:
                msg = "Must specify 'scale' if 'path' is not provided!"
                logger.critical(msg)
                raise ValueError(msg)
            return BalmerContinuumTemplate.load_from_cache(
                temp=temp,
                tau=tau,
                scale=scale,
                info=self.info,
            )

        return BalmerContinuumTemplate.load(
            path=template_file,
            info=self.info,
        )

    @validated_apply_info_to_method(subjects=("balmer",))
    def instantiateModel(
        self,
        *,
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
    ) -> BalmerModel:
        """
        Instantiates a BalmerModel instance using the given parameters and
        bounds.
        """
        logger.debug("Instantiating BalmerModel:")

        x_norm: float = self.info.balmer.edge
        renormalise: bool = (x_norm < self._x[0]) or (self._x[-1] < x_norm)
        logger.debug(f"> {x_norm=:.1f}")
        logger.debug(f"> {renormalise=}")

        series_template = self.loadBalmerSeriesTemplate.__wrapped__(
            self,
            source=source,
            temp=temp,
            dens=dens,
            n_u_min=n_u_min,
            n_u_max=n_u_max,
        ).createLogspace(
            sigma_res=self.info.loading.sigma_res,
            xr=self._x,
            keep_x=True,
        )
        logger.debug(f"> Loaded series template: {series_template!s}")

        continuum_template = self.loadBalmerContinuumTemplate.__wrapped__(
            self,
            temp=temp,
            tau=tau,
            scale=scale,
        ).createLogspace(
            sigma_res=self.info.loading.sigma_res,
            xr=self._x,
            keep_x=True,
        )
        logger.debug(f"> Loaded continuum template: {continuum_template!s}")

        if renormalise:
            logger.debug("> Renormalising templates...")
            series_template._calculate_normalisation()
            continuum_template._calculate_normalisation()

        self.model = BalmerModel.create(
            flux,
            fwhm,
            ratio,
            edge=self.info.balmer.edge,
            continuum_template=continuum_template,
            series_template=series_template,
            allow_interp_fitting=allow_interp_fitting,
            name="balmer",
            flux_bounds=flux_bounds,
            fwhm_bounds=fwhm_bounds,
            ratio_bounds=ratio_bounds,
            flux_fixed=fixed.get("flux", False),
            fwhm_fixed=fixed.get("fwhm", False),
            ratio_fixed=fixed.get("ratio", False),
        )
        logger.debug(f"> Instantiated BalmerModel: {self.model!s}")

        return self.model

    @validated_apply_info_to_method(subjects=("balmer",))
    def checkModelCoverage(
        self,
        *,
        without_rejections: bool = False,
        without_absorption: bool = False,
        min_fittable_ratio: float | None = None,
        min_fittable_total: int | None = None,
        covered: bool = False,
    ) -> bool:
        """
        Checks the degree of coverage of each side of the Balmer
        pseudo-continuum.
        """
        model = self.getModel()
        assert model is not None, "Model has not been instantiated!"

        fittable_pixels = self.getMask.__wrapped__(
            self,
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=True,
            log_valid=False,
        )

        # The blue side: Balmer continuum + attenuation contribution
        is_blue = self._x <= model.edge
        n_blue: int = fittable_pixels[is_blue].sum()
        r_blue: float = fittable_pixels[is_blue].mean() if n_blue > 0 else 0.0
        cond1 = (r_blue < min_fittable_ratio) or (n_blue < min_fittable_total)

        # The red side: Balmer series contribution
        is_red = ~is_blue
        n_red: int = fittable_pixels[is_red].sum()
        r_red: float = fittable_pixels[is_red].mean() if n_red > 0 else 0.0
        cond2 = (r_red < min_fittable_ratio) or (n_red < min_fittable_total)

        if cond1 and cond2:
            msg = "Insufficient coverage on both sides of the Balmer "\
                "pseudo-continuum: removing model!"
            logger.debug(msg)
            self.model = None
            return False
        elif cond1:
            msg = "Insufficient coverage on the blue side of the Balmer "\
                "pseudo-continuum: freezing 'ratio' parameter!"
            logger.debug(msg)
        elif cond2:
            msg = "Insufficient coverage on the red side of the Balmer "\
                "pseudo-continuum: freezing 'ratio' parameter!"
            logger.debug(msg)
        else:
            msg = "Sufficient coverage on both sides of the Balmer "\
                "pseudo-continuum: proceeding with fitting!"
            logger.debug(msg)
            return True

        model.ratio.value = 1.0
        model.ratio.bounds = (
            model.ratio.bounds[0],
            min(model.ratio.bounds[1], 1.0),
        )
        model.ratio.fixed = True

        return True

    @pydantic.validate_call
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
        assert self.model is not None, "Model has not been instantiated!"

        if bg_flux is None:
            bg_flux = self.default_bg

        if covered and self.is_empty:
            msg = f"Setting {covered=} to False due to missing host windows!"
            logger.info(msg)
            covered = False

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
        self.model.rasterFit.__wrapped__(
            self.model,
            masked_coords.x,
            masked_coords.y,
            masked_coords.dy,
            inplace=True,
        )
        return self

    @validated_apply_info_to_method(subjects=("nonlinear",))
    def performFineTuning(
        self,
        *,
        original: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        covered: bool = False,
        bg_flux: BackgroundFlux | None = None,
        shrink: bool | None = None,
        fitter_kwargs: FitterKwargs | None = None,
    ) -> Self:
        from quasar_models.modeling import Fitter, PrepareModel
        
        msg = f"Performing fine-tuning fit on {self.__str__(True)}: "

        original = self._update_original_flag(
            original=original,
            log=True,
            prefix=">>> ",
        )

        if bg_flux is None:
            bg_flux = self.default_bg

        model = self.getModel()
        if model is None:
            msg = "No BalmerModel instance available!"
            raise ValueError(msg)

        if covered and self.is_empty:
            msg += f"setting {covered=} to False due to missing host windows, "
            logger.info(msg)
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

        self.applyFit.__wrapped__(
            self,
            fit,
            fit_info=fit_info,
            update_emission=True,
        )
        return self

    def getModel(self) -> Optional[BalmerModel]:
        """
        Retrieves the Balmer pseudo-continuum model if available.

        Notes
        -----
        If a fitted Balmer pseudo-continuum model is available, it is
        returned. Otherwise an instantiated model is returned, if available.
        """
        return self.fit or self.model

    @pydantic.validate_call
    def adoptFit(
        self,
        fit: BalmerModel,
        *,
        fit_info: OptimizeResultType | None = None,
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

    @pydantic.validate_call
    def applyFit(
        self,
        fit: Optional[BalmerModel],
        *,
        fit_info: OptimizeResultType | None = None,
        update_emission: bool = False,
    ) -> Self:
        """
        Applies the given Balmer pseudo-continuum fit.
        """
        self.fit = fit
        self.fit_info = fit_info

        if update_emission:
            self.updateBalmerEmission.__wrapped__(self, fit)

        return self

    def windowsplot(
        self,
        mode: Literal["residual", "difference"],
        model: Optional[BalmerModel] = None,
        lazy: bool = False,
        **plotter_kwargs,
    ):
        """
        ...
        """
        from quasar_decrypt.plotting.windows_plots import (
            BalmerWindowsPlotWithDifferences,
            BalmerWindowsPlotWithResiduals,
        )
        if model is None:
            model = self.getModel()
        
        args = (model, self.window_bounds, self)
        cls = BalmerWindowsPlotWithResiduals \
            if mode == "residual" \
            else BalmerWindowsPlotWithDifferences

        return self._plot_helper(
            plotter_cls=cls,
            lazy=lazy,
            args=args,
            plotter_kwargs=plotter_kwargs,
        )

    def templatesplot(
        self,
        mode: Literal["residual", "difference"],
        model: Optional[BalmerModel] = None,
        with_windows: bool = True,
        lazy: bool = False,
        **plotter_kwargs,
    ):
        """
        ...
        """
        from quasar_decrypt.plotting.windows_and_templates_plots import (
            BalmerTemplatesWithDifferencesPlot,
            BalmerTemplatesWithResidualsPlot,
            BalmerWindowsAndTemplatesWithDifferencesPlot,
            BalmerWindowsAndTemplatesWithResidualsPlot,
        )

        if model is None:
            model = self.getModel()

        if with_windows:
            args = (model, self.window_bounds, self.spectrum)
            cls = BalmerWindowsAndTemplatesWithResidualsPlot \
                if mode == "residual" \
                else BalmerWindowsAndTemplatesWithDifferencesPlot
        else:
            args = (model, self.spectrum)
            cls = BalmerTemplatesWithResidualsPlot \
                if mode == "residual" \
                else BalmerTemplatesWithDifferencesPlot

        return self._plot_helper(
            plotter_cls=cls,
            lazy=lazy,
            args=args,
            plotter_kwargs=plotter_kwargs,
        )