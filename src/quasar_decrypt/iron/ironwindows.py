from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import repeat
from logging import getLogger
from pathlib import Path
from typing import ClassVar, Literal, Optional, Self, Union

from numpy import isfinite
from pydantic import ValidationError
from quasar_errors.model_samples import IronSampleList
from quasar_models.iron import IronModel, IronTemplate
from quasar_models.modeling import Fitter, PrepareModel
from quasar_models.utils.astropy import apply_bounds, get_free_params
from quasar_typing.astropy import CompoundModel_, FitInfo
from quasar_typing.bounds import AstropyBounds, CoordBounds
from quasar_typing.misc import BackgroundFlux
from quasar_typing.numpy import FloatVector
from quasar_typing.pathlib import AbsoluteFITSPath
from quasar_utils.decorators import (
    validate_call,
    validated_apply_info_to_method,
)
from quasar_utils.setup import FitterKwargs
from scipy.ndimage import binary_fill_holes

from ..utils import _SpecWindowList, stopwatch
from .iwindow import IWindow

logger = getLogger(__name__)


@dataclass
class IronWindows(_SpecWindowList[IWindow]):
    templates: dict[str, IronTemplate] = field(
        default_factory=dict, 
        kw_only=True,
    )
    template_models: dict[str, IronModel] = field(
        default_factory=dict, 
        kw_only=True,
    )
    fit_info: FitInfo | None = field(
        default=None, 
        kw_only=True,
    )

    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({"all", "fe"})

    @property
    def sample(self) -> IronSampleList | None:
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
        template_model: Union[IronModel, CompoundModel_[IronModel], None] = None,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = True,
        without_absorption: bool = True,
        template_files: list[AbsoluteFITSPath | str] | None = None,
        resample: bool | None = None,
        split: FloatVector | None = None,
        fwhm: FloatVector | None = None,
        bias: list[Literal["left", "right"]] | None = None,
        ratio: FloatVector | None = None,
        scale: float | None = None,
        raster: bool | None = None,
        allow_interp_fitting: bool | None = None,
        flux_bounds: AstropyBounds | None = None,
        fwhm_bounds: AstropyBounds | None = None,
        fine_tune: bool | None = None,
        fitter_kwargs: FitterKwargs | None = None,
    ) -> bool:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
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
                    bg_flux=bg_flux,
                    covered=not self.is_empty,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                    fitter_kwargs=fitter_kwargs,
                )

        return True

    @validated_apply_info_to_method(subjects=("iron",))
    def loadTemplates(
        self,
        *,
        template_files: list[AbsoluteFITSPath | str] | None = None,
        resample: bool | None = None,
        split: FloatVector | None = None,
        fwhm: FloatVector | None = None,
        bias: list[Literal["left", "right"]] | None = None,
        ratio: FloatVector | None = None,
        scale: float | None = None,
        flux_bounds: AstropyBounds | None = None,
        fwhm_bounds: AstropyBounds | None = None,
        allow_interp_fitting: bool | None = None,
    ) -> Self:
        """
        Loads all templates:
        >   which are specified by the user.
        >   which are covered by the iron windows.

        Should adapt the templates to the current spectrum covered by the
        respective templates.
        """
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
            msg = (
                "IronWindows is empty: using entire wavelength array to "
                "check coverages of IronTemplates."
            )
            logger.debug(msg)
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
            try:
                template = IronTemplate.load_from_cache(
                    name=template_file,
                    info=self.info,
                ).copy(with_matrices=True)
            except FileNotFoundError:
                msg = (
                    f"Could not find template file '{template_file}' -> "
                    "Skipping."
                )
                logger.info(msg)
                continue

            # Check template-coverage
            tx = template.x[binary_fill_holes(template.data[0] > 0)]

            add_template: bool = False
            if self.is_empty:
                add_template = not (x[-1] < tx[0] or tx[-1] < x[0])
            else:
                for iwindow in self:
                    lb, ub = iwindow.x_bounds
                    add_template = not (ub < tx[0] or tx[-1] < lb)
                    if add_template or tx[-1] < lb:
                        break

            if not add_template:
                if tx[-1] < x[0]:
                    msg = (
                        f"IronTemplate @ {template_file} is entirely "
                        "bluewards of the spectrum! Skipping."
                    )
                elif x[-1] < tx[0]:
                    msg = (
                        f"IronTemplate @ {template_file} is entirely "
                        "redwards of the spectrum! Skipping."
                    )
                else:
                    msg = (
                        f"IronTemplate @ {template_file} does not cover any "
                        "of the iron windows! Skipping."
                    )

                logger.debug(msg)
                continue

            # Transform template if necessary
            if template.is_logspace:
                msg = (
                    f"IronTemplate @ {template_file} is already in logspace!"
                    "Using template as is."
                )
                logger.debug(msg)

            if not template.is_logspace:
                msg = (
                    f"IronTemplate @ {template_file} is not in logspace! "
                    "Creating logspace-equivalent version."
                )
                logger.debug(msg)

                msg = (
                    f"IronTemplate @ {template_file} will be transformed to "
                    "logspace, which may be inefficient for pipelines. "
                    "Consider caching a logspace-equivalent of this IronTemplate."
                )
                logger.info(msg)

                tx_wide = template.x[binary_fill_holes(template.data[-1] > 0)]
                mask = (
                    isfinite(self._x)
                    & (tx_wide[0] <= self._x)
                    & (self._x <= tx_wide[-1])
                )

                template = template.createLogspace(
                    sigma_res=self.info.loading.sigma_res,
                    xr=self._x[mask],
                    keep_x=True,
                )

            # Resample template if necessary
            if resample:
                template.resample(fwhm, inplace=True)

            if template.x[0] < s < template.x[-1]:
                msg = f"Applying split to IronTemplate @ {template_file}: split={s:.1f}, bias={b}, ratio={r:.2f}."
                logger.debug(msg)

                template.applySplit(
                    split=s,
                    left=1.0 if b == "left" else r,
                    right=r if b == "left" else 1.0,
                    scale=scale,
                    inplace=True,
                )

            self.templates[template.name] = template

            model = IronModel.create(
                apply_bounds.__wrapped__(1.0, flux_bounds),
                apply_bounds.__wrapped__(template.fwhm[0], fwhm_bounds),
                scale=self.info.iron.scale,
                template=template,
                split=s,
                left=1.0,
                right=1.0,
                allow_interp_fitting=allow_interp_fitting,
            )
            model.flux.bounds = flux_bounds
            model.fwhm.bounds = fwhm_bounds

            self.template_models[model.name] = model

        return self

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
            mode="c",  # c: contiguous
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=True,
            log_valid=False,
            bg_flux=bg_flux,
        )

        n_pix = masked_coords.size
        if n_pix <= 2:
            msg = (
                "cancelling raster fit due to insufficient no. of data "
                f"points (n_pix={n_pix} <= n_free_params=2)!"
            )
            logger.info(msg)
            return self

        for model in self.template_models.values():
            model.rasterFit.__wrapped__(
                model,
                masked_coords.x,
                masked_coords.y,
                masked_coords.dy,
                inplace=True,
            )

        self.updateIronEmission.__wrapped__(self, self.getModel())
        return self

    def getModel(
        self,
    ) -> Optional[Union[IronModel, CompoundModel_[IronModel]]]:
        """
        Combines the available templates into a single (Split) TemplateModel or
        AstroPy compound model.
        """
        submodels = list(self.template_models.values())
        return sum(submodels[1:], start=submodels[0]) if submodels else None

    @validate_call
    def adoptFit(
        self,
        fit: Union[IronModel, CompoundModel_[IronModel]],
        *,
        fit_info: FitInfo | None = None,
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

    @validate_call
    def applyFit(
        self,
        fit: Union[IronModel, CompoundModel_[IronModel]],
        *,
        fit_info: FitInfo | None = None,
        update_emission: bool = False,
    ) -> Self:
        self.fit_info = fit_info
        for template_model in (fit,) if fit.n_submodels == 1 else fit:
            self.template_models[template_model.name] = template_model

        if update_emission:
            self.updateIronEmission.__wrapped__(self, fit)
            for iwindow in filter(lambda w: w._y_fe is not self._y_fe, self):
                iwindow.updateIronEmission.__wrapped__(iwindow, fit)

        return self

    @validated_apply_info_to_method(subjects=("nonlinear",))
    def performFineTuning(
        self,
        *,
        covered: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        bg_flux: BackgroundFlux | None = None,
        fitter_kwargs: FitterKwargs | None = None,
    ) -> Self:
        """
        Fits the available templates using a nonlinear optimiser.
        """
        s = self.__str__(simple=True).removesuffix(".")
        msg = f"Performing fine-tuning on {s}: "

        if bg_flux is None:
            bg_flux = self.default_bg

        if covered and self.is_empty:
            msg += f"setting {covered=} to False due to missing iron windows, "
            logger.info(msg)
            covered = False

        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            mode="c",  # c: contiguous
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=True,
            log_valid=False,
            bg_flux=bg_flux,
        )

        model = self.getModel()
        if model is None:
            msg += "tried to fine-tune IronModels, but none are available!"
            logger.critical(msg)
            return self

        n_pix = masked_coords.size
        n_free_params = sum(get_free_params(model).values())

        if n_pix <= n_free_params:
            msg += (
                f"cancelling fine-tuning due to insufficient no. of data "
                f"points (n_pix={n_pix} <= n_free_params={n_free_params})!"
            )
            logger.info(msg)
            return self

        try:
            with (
                stopwatch() as watch,
                PrepareModel(x=masked_coords.x, model=model, copy=True),
            ):
                fitter = Fitter()
                fit = fitter(
                    model,
                    masked_coords.x,
                    masked_coords.y,
                    dy=masked_coords.dy,
                    get_model=True,
                    inplace=False,
                    **fitter_kwargs,
                )

            msg += f"Successfully performed fine-tuning in {1e3 * watch.elapsed:.1f} ms."
            logger.debug(msg)
        except ValidationError as e:
            msg += f"failed fitting due to a validation error: {e}"
            logger.warning(msg)
            return self
        except Exception as e:
            msg += f"failed fitting due to an unexpected error: {e}"
            logger.warning(msg)
            return self

        self.applyFit.__wrapped__(
            self,
            fit,
            fit_info=fitter.fit_info,
            update_emission=True,
        )
        return self
