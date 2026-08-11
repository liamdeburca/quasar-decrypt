__all__ = ["ContinuumWindows"]

from collections.abc import Iterable
from dataclasses import dataclass, field
from logging import getLogger
from typing import ClassVar, Literal, Optional, Self

from pydantic import ValidationError
from quasar_errors.model_samples import PowerLawSample
from quasar_models.continuum import PowerLawModel
from quasar_models.modeling import Fitter, PrepareModel
from quasar_models.utils.astropy import apply_bounds, get_free_params
from quasar_typing.astropy import FitInfo
from quasar_typing.bounds import AstropyBounds, CoordBounds
from quasar_typing.misc import BackgroundFlux, Suffix
from quasar_utils.continuum_fit_result import ContinuumFitResult
from quasar_utils.decorators import (
    validate_call,
    validated_apply_info_to_method,
)
from quasar_utils.setup import FitterKwargs

from ..utils.general import stopwatch
from ..utils.speclist import SpecList
from .cwindow import CWindow

logger = getLogger(__name__)


@dataclass(init=False)
class ContinuumWindows(SpecList[CWindow]):
    fit_info: FitInfo | None = field(default=None, init=False)

    fit_raw: Optional[PowerLawModel] = field(default=None, init=False)
    fit_sc: Optional[PowerLawModel] = field(default=None, init=False)
    fit: Optional[PowerLawModel] = field(default=None, init=False)

    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({"all", "pl"})

    @property
    def sample(self) -> PowerLawSample | None:
        if (model := self.getModel()) is None:
            return None
        return PowerLawSample.fromPowerLawModel(model)

    @validated_apply_info_to_method(
        subjects=("continuum",),
        specific_kwargs={"windows"},
    )
    def populate(
        self,
        *,
        windows: Iterable[CoordBounds] | None = None,
    ) -> Self:
        kwargs = self.get_kwargs_from_specdata.__wrapped__(
            self.__class__,
            self.spectrum or self,
        )
        kwargs.pop("x_bounds")

        for x_bounds in windows:
            cwindow = CWindow(x_bounds=x_bounds, **kwargs)
            if cwindow.size > 0:
                self.append(cwindow)

        return self

    @validated_apply_info_to_method(subjects=("continuum", "nonlinear"))
    def __call__(
        self,
        *,
        template_model: Optional[PowerLawModel] = None,
        bg_flux: BackgroundFlux | None = None,
        sigmas: list[float] | None = None,
        flux_bounds: AstropyBounds | None = None,
        alpha_bounds: AstropyBounds | None = None,
        min_fittable_total: int | None = None,
        fitter_kwargs: FitterKwargs | None = None,
    ) -> bool:
        logger.debug(f"Starting pipeline for {self.__str__(True)}")

        if bg_flux is None:
            bg_flux = self.default_bg

        if template_model is not None:
            logger.debug("Applying template model.")
            self.applyFit.__wrapped__(
                self,
                template_model,
            )
        else:
            msg = "Performing linear fit: "
            success = self.getLinearFit.__wrapped__(
                self,
                without_rejections=False,
                without_absorption=True,
                suffix="raw",
                bg_flux=bg_flux,
                flux_bounds=flux_bounds,
                alpha_bounds=alpha_bounds,
                min_fittable_total=min_fittable_total,
            )
            if not success:
                logger.warning(msg + "failed!")
                return False
            
            logger.debug(msg + "success!")

            if sigmas:
                logger.debug("Performing sigma-clipping...")
                _ = self.performSigmaClipping.__wrapped__(
                    self,
                    without_absorption=True,
                    bg_flux=bg_flux,
                    sigmas=sigmas,
                    flux_bounds=flux_bounds,
                    alpha_bounds=alpha_bounds,
                    min_fittable_total=min_fittable_total,
                )

        msg = "Performing fine-tuning: "
        success = self.performFineTuning.__wrapped__(
            self,
            update_flux=True,
            without_rejections=True,
            without_absorption=True,
            bg_flux=bg_flux,
            flux_bounds=flux_bounds,
            alpha_bounds=alpha_bounds,
            min_fittable_total=min_fittable_total,
            fitter_kwargs=fitter_kwargs,
        )
        if not success:
            logger.warning(msg + "failed!")
            return False
        logger.debug(msg + "success!")

        return True

    @validated_apply_info_to_method(subjects=("continuum",))
    def getLinearFit(
        self,
        *,
        without_rejections: bool = False,
        without_absorption: bool = False,
        suffix: Suffix | None = "raw",
        bg_flux: BackgroundFlux | None = None,
        flux_bounds: AstropyBounds | None = None,
        alpha_bounds: AstropyBounds | None = None,
        min_fittable_total: int | None = None,
    ) -> bool:
        s = self.__str__(simple=True).removesuffix(".")
        msg = f"Performing linear fit on {s}:"

        if bg_flux is None:
            bg_flux = self.default_bg

        if self.is_empty:
            log = logger.warning
            msg += " No continuum windows -> performing global fit!"
            covered = False
        else:
            log = logger.debug
            covered = True

        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            mode="c",
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=True,
            log_valid=True,
            bg_flux=bg_flux,
        )
        n_points: int = masked_coords.size
        n_min: int = min_fittable_total
        msg += f" valid frac.={n_points}/{(self.x if covered else self._x).size} is "
        if n_points < n_min:
            msg += f"insufficient (<{n_min}): cancelling!"
            logger.critical(msg)
            return False
        else:
            msg += "sufficient: proceeding. "

        if self.fit_raw is None:
            msg += "[NOTE: no linear fit found] "
            prev_model = PowerLawModel.create(
                self.x0,
                self.y0,
                apply_bounds.__wrapped__(masked_coords.y.mean(), flux_bounds),
                apply_bounds.__wrapped__(0, alpha_bounds),
                name="powerlaw_model",
            )
            prev_model.flux.bounds = flux_bounds
            prev_model.alpha.bounds = alpha_bounds
        else:
            msg += "[NOTE: retrieving previous model] "
            prev_model = self.getModel.__wrapped__(self, suffix=suffix)

        with stopwatch() as watch:
            fit = prev_model.from_linear_fit(
                masked_coords.x,
                masked_coords.y,
                masked_coords.dy,
            )

        log(msg + f"Finished linear fit in {1e3 * watch.elapsed:.1f} ms.")
        self.applyFit.__wrapped__(
            self,
            fit,
            suffix=suffix,
            update_emission=True,
        )
        return True

    @validated_apply_info_to_method(subjects=("continuum",))
    def performSigmaClipping(
        self,
        without_absorption: bool = True,
        bg_flux: BackgroundFlux | None = None,
        *,
        sigmas: list[float] | None = None,
        flux_bounds: AstropyBounds | None = None,
        alpha_bounds: AstropyBounds | None = None,
        min_fittable_total: int | None = None,
    ) -> Literal[-1, 0] | int:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        s = self.__str__(simple=True).removesuffix(".")
        msg = f"Performing sigma clipping on {s}: "

        if bg_flux is None:
            bg_flux = self.default_bg

        if self.fit_raw is None:
            msg += " (got initial linear fit)."
            success = self.getLinearFit.__wrapped__(
                self,
                without_rejections=False,
                without_absorption=True,
                suffix="raw",
                bg_flux=bg_flux,
                flux_bounds=flux_bounds,
                alpha_bounds=alpha_bounds,
                min_fittable_total=min_fittable_total,
            )
            if not success:
                msg += " Failed initial fit: cancelling sigma-clipping!"
                return 0

        if self.is_empty:
            msg += " No continuum windows: cancelling sigma-clipping!"
            logger.warning(msg)
            self.applyFit.__wrapped__(
                self,
                self.fit_raw.copy(),
                suffix="sc",
            )
            return -1

        self.resetRejections()
        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            mode="c",
            covered=True,
            without_rejections=False,
            without_absorption=False,
            valid=True,
            log_valid=True,
            bg_flux=bg_flux,
        )

        n_clips: int = 0
        n_min: int = min_fittable_total
        f = self.fit_raw

        log = logger.debug
        with stopwatch() as watch:
            for sigma in sigmas:
                z = f.getResiduals.__wrapped__(
                    f,
                    masked_coords.x,
                    masked_coords.y,
                    masked_coords.dy,
                    log=True,
                )
                self._rejected_pixels[masked_coords.mask] = abs(z) > sigma

                masked_coords = self.getMaskedCoords.__wrapped__(
                    self,
                    mode="c",
                    covered=True,
                    without_rejections=True,
                    without_absorption=without_absorption,
                    valid=True,
                    log_valid=True,
                    bg_flux=bg_flux,
                )

                n_pix = masked_coords.size
                if n_pix < n_min:
                    if n_clips == 0:
                        msg += f" Raw fit rejected all but {n_pix} points (min. {n_min}): cancelling sigma-clipping and resetting rejections!"
                        n_clips = -1
                        self.resetRejections()
                        break

                    msg += f" Fit rejected all but {n_pix} points (min. {n_min}): stopping after {n_clips} iterations."
                    break

                success = self.getLinearFit.__wrapped__(
                    self,
                    without_rejections=True,
                    without_absorption=without_absorption,
                    suffix="sc",
                    flux_bounds=flux_bounds,
                    alpha_bounds=alpha_bounds,
                    min_fittable_total=min_fittable_total,
                )
                if success:
                    f = self.fit_sc
                    n_clips += 1
                else:
                    msg += f" Failed fit: stopping after {n_clips} iterations."
                    break

        msg += f" Finished sigma-clipping in {1e3 * watch.elapsed:.1f} ms."
        log(msg)

        return n_clips

    @validated_apply_info_to_method(subjects=("continuum", "nonlinear"))
    def performFineTuning(
        self,
        *,
        update_flux: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        bg_flux: BackgroundFlux | None = None,
        flux_bounds: AstropyBounds | None = None,
        alpha_bounds: AstropyBounds | None = None,
        min_fittable_total: int | None = None,
        fitter_kwargs: FitterKwargs | None = None,
    ) -> bool:
        s = self.__str__(simple=True).removesuffix(".")
        msg = f"Performing fine-tuning on {s}: "

        if bg_flux is None:
            bg_flux = self.default_bg

        model = self.getModel()
        if model is None:
            msg += "(got initial linear fit), "
            success = self.getLinearFit.__wrapped__(
                self,
                without_rejections=True,
                without_absorption=True,
                suffix="raw",
                bg_flux=bg_flux,
                flux_bounds=flux_bounds,
                alpha_bounds=alpha_bounds,
                min_fittable_total=min_fittable_total,
            )
            if not success:
                logger.warning(msg + "Failed initial fit: cancelling fine-tuning!")
                return False

            model = self.fit_raw

        msg += "flux={:.1f} ({:.1f},{:.1f})".format(
            model.flux.value,
            *model.flux.bounds,
        )
        msg += " alpha={:.2f} | ({:.2f},{:.2f})".format(
            model.alpha.value,
            *model.alpha.bounds,
        )

        if self.is_empty:
            msg += "performing global fit due to missing continuum windows -> "

        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            mode="c",
            covered=not self.is_empty,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=True,
            bg_flux=bg_flux,
        )

        n_pix = masked_coords.x.size
        n_min = min_fittable_total
        n_free_params = sum(get_free_params(model).values())

        if n_pix < max(n_free_params, n_min):
            msg += (
                "cancelling fine-tuning due to insufficient no. of data "
                f"points (n_pix={n_pix} < max(n_free_params={n_free_params}, n_min={n_min}))!"
            )
            self.applyFit.__wrapped__(
                self,
                self.getModel(),
            )
            logger.warning(msg)
            return False

        try:
            with (
                stopwatch() as watch,
                PrepareModel(x=masked_coords.x, model=model),
            ):
                fitter = Fitter()
                fit = fitter(
                    model,
                    masked_coords.x,
                    masked_coords.y,
                    dy=masked_coords.dy,
                    get_model=True,
                    inplace=False,
                    **(fitter_kwargs or {}),
                )
                fit_info = fitter.fit_info

            msg += (
                "Successfully performed fine-tuning in {:.1f} ms: "
                "flux={:.1f} ({:.1f},{:.1f}), "
                "alpha={:.2f} ({:.2f},{:.2f})".format(
                    1e3 * watch.elapsed,
                    fit.flux.value,
                    *fit.flux.bounds,
                    fit.alpha.value,
                    *fit.alpha.bounds,
                )
            )
            logger.debug(msg)

        except ValidationError as e:
            msg += f"failed fitting due to a validation error: {e}"
            logger.warning(msg)
            self.applyFit.__wrapped__(self, self.getModel())
            return False
        except Exception as e:
            msg += f"failed fitting due to an unexpected error: {e}"
            logger.warning(msg)
            raise ValueError(msg) from e

        if self.is_empty:
            msg += " No continuum windows -> performing global fit!"

        self.applyFit.__wrapped__(
            self,
            fit,
            fit_info=fit_info,
            update_emission=update_flux,
        )
        return True

    @validate_call
    def getModel(
        self,
        suffix: Suffix | None = None,
    ) -> Optional[PowerLawModel]:
        if suffix is None:
            return self.fit or self.fit_sc or self.fit_raw
        elif suffix == "sc":
            return self.fit_sc or self.fit_raw
        return self.fit_raw

    @validate_call
    def adoptFit(
        self,
        fit: PowerLawModel,
        *,
        fit_info: FitInfo | None = None,
        update_emission: bool = False,
    ) -> Self:
        """
        Identical to 'applyFit' with 'suffix=None'.
        """
        return self.applyFit.__wrapped__(
            self,
            fit,
            fit_info=fit_info,
            suffix=None,
            update_emission=update_emission,
        )

    @validate_call
    def applyFit(
        self,
        fit: PowerLawModel,
        *,
        fit_info: FitInfo | None = None,
        suffix: Suffix | None = None,
        update_emission: bool = False,
    ) -> Self:
        self.fit_info = fit_info

        match suffix:
            case "raw":
                self.fit_raw = fit
            case "sc":
                self.fit_sc = fit
            case None:
                self.fit = fit

        if update_emission:
            self.updateContinuumEmission.__wrapped__(self, fit)

        for window in self:
            window.applyFit.__wrapped__(
                window,
                fit,
                fit_info=fit_info,
                suffix=suffix,
                update_emission=update_emission
                and (self._y_pl is not window._y_pl),
            )

        return self

    def summariseContinuumFit(self) -> ContinuumFitResult:
        return ContinuumFitResult(
            self.fit_info.x,
            self.fit_info.param_cov,
            x0=self.x0,
        )
