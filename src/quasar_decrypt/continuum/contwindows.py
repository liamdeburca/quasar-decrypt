__all__ = ["ContinuumWindows"]

from collections.abc import Iterable
from dataclasses import field
from logging import getLogger
from typing import Any, ClassVar, Literal, Optional, Self

import pydantic
from quasar_models.continuum import PowerLawModel
from quasar_typing.bounds import AstropyBounds, CoordBounds
from quasar_typing.errors import MissingModelError, OutsideBoundsError
from quasar_typing.literals import Suffix
from quasar_typing.scipy import OptimizeResultType
from quasar_typing.string_selections import BackgroundFlux
from quasar_utils.continuum_fit_result import ContinuumFitResult
from quasar_utils.decorators import validated_apply_info_to_method
from quasar_utils.logging import flag
from quasar_utils.setup import FitterKwargs

from quasar_decrypt.utils import _SpecWindowList, stopwatch

from .cwindow import CWindow

logger = getLogger(__name__)


@pydantic.dataclasses.dataclass
class ContinuumWindows(_SpecWindowList[CWindow]):
    fit_info: OptimizeResultType | None = field(default=None, init=False)

    fit_raw: Optional[PowerLawModel] = field(default=None, init=False)
    fit_sc: Optional[PowerLawModel] = field(default=None, init=False)
    fit: Optional[PowerLawModel] = field(default=None, init=False)

    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({"all", "pl"})

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["fit_info"] = self.fit_info
        state["fit_raw"] = self.fit_raw
        state["fit_sc"] = self.fit_sc
        state["fit"] = self.fit
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.fit_info = state.get("fit_info", None)
        self.fit_raw = state.get("fit_raw", None)
        self.fit_sc = state.get("fit_sc", None)
        self.fit = state.get("fit", None)

    @property
    def sample(self):
        from quasar_errors.model_samples import PowerLawSample
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
        for x_bounds in windows:
            cwindow = CWindow.create.__wrapped__(
                CWindow,
                spectrum=self.spectrum,
                x_bounds=x_bounds,
                get_mask=None, # Creates new, dedicated get_mask for each window
            )
            if cwindow.size > 0:
                self.append(cwindow)

        flag("N_CONT_WINDOWS", len(self))
        return self

    @validated_apply_info_to_method(subjects=("continuum", "nonlinear"))
    def __call__(
        self,
        *,
        original: bool = False,
        without_absorption: bool = False,
        without_rejections: bool = False,
        template_model: Optional[PowerLawModel] = None,
        bg_flux: BackgroundFlux | None = None,
        shrink: float | None = None,
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
            self.instantiateModel.__wrapped__(
                self,
                update_flux=False,
                without_rejections=False,
                without_absorption=without_absorption,
                suffix="raw",
                bg_flux=bg_flux,
                flux_bounds=flux_bounds,
                alpha_bounds=alpha_bounds,
            )

            msg = "Performing linear fit: "
            try:
                success = self.getLinearFit.__wrapped__(
                    self,
                    without_rejections=False, # Ignore rejections
                    without_absorption=without_absorption,
                    suffix="raw",
                    bg_flux=bg_flux,
                    flux_bounds=flux_bounds,
                    alpha_bounds=alpha_bounds,
                    min_fittable_total=min_fittable_total,
                )
            except OutsideBoundsError:
                # Linear regression was unsuccessful, but nonlinear optimisation
                # may still work:
                # - If that fails, return False
                # - If that succeeds: continue to sigma-clipping. If not 
                #   enabled return early.
                msg += "[OutsideBoundsError] failed linear regression "
                msg += "-> trying nonlinear optimisation. "

                try:
                    success = self.performFineTuning.__wrapped__(
                        self,
                        update_flux=True,
                        without_rejections=False, # Ignore rejections due to error
                        without_absorption=without_absorption,
                        bg_flux=bg_flux,
                        min_fittable_total=min_fittable_total,
                        fitter_kwargs=fitter_kwargs,
                    )
                except ValueError:
                    msg += "Unexpected error during nonlinear optimisation!"
                    logger.info(msg)
                    return False

                if success:
                    msg += "Nonlinear optimisation succeeded -> returning early."
                    logger.info(msg)
                    return True
                                
            if not success:
                logger.info(msg + "failed!")
                return False
            
            logger.debug(msg + "success!")

            if sigmas:
                logger.debug("Performing sigma-clipping...")
                _ = self.performSigmaClipping.__wrapped__(
                    self,
                    without_absorption=without_absorption,
                    bg_flux=bg_flux,
                    sigmas=sigmas,
                    flux_bounds=flux_bounds,
                    alpha_bounds=alpha_bounds,
                    min_fittable_total=min_fittable_total,
                )

        msg = "Performing fine-tuning: "
        success = self.performFineTuning.__wrapped__(
            self,
            original=original,
            update_flux=True,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            bg_flux=bg_flux,
            min_fittable_total=min_fittable_total,
            shrink=shrink,
            fitter_kwargs=fitter_kwargs,
        )
        if not success:
            logger.info(msg + "failed!")
            return False
        logger.debug(msg + "success!")

        return True

    @pydantic.validate_call
    def instantiateModel(
        self,
        *,
        update_flux: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        suffix: Suffix | None = None,
        bg_flux: BackgroundFlux | None = None,
        flux_bounds: AstropyBounds | None = None,
        alpha_bounds: AstropyBounds | None = None,
    ) -> None:
        from quasar_models.utils.astropy import _apply_bounds

        msg = f"Instantiating model for {self.__str__(simple=True)}: "
        if bg_flux is None:
            bg_flux = self.default_bg

        covered = not self.is_empty

        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            original=False,
            mode="c",
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=True,
            log_valid=False,
            bg_flux=bg_flux,
            shrink=None,
        )
        if masked_coords.size == 0:
            msg += f"No valid points found w/ {covered=}, {without_rejections=}, {without_absorption=}"
            logger.critical(msg)
            raise ValueError(msg)

        msg += f"Instantiating model using {masked_coords.size} points: "

        flux = _apply_bounds(masked_coords.y.mean(), flux_bounds)
        msg += f"{flux=:.1f} ({flux_bounds[0]:.1f},{flux_bounds[1]:.1f}), "
        alpha = sum(alpha_bounds) / 2
        msg += f"{alpha=:.2f} ({alpha_bounds[0]:.2f},{alpha_bounds[1]:.2f})"

        logger.debug(msg)

        model = PowerLawModel.create(
            self.x0, self.y0,
            flux, alpha,
            name="powerlaw",
            flux_bounds=flux_bounds,
            alpha_bounds=alpha_bounds,
            flux_fixed=False,
            alpha_fixed=False,
        )
        self.applyFit.__wrapped__(
            self,
            model,
            suffix=suffix,
            update_emission=update_flux,
        )

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
        """
        Raises
        ------
        ValidationError
            If Pydantic validation fails.
        OutsideBoundsError
            If the derived parameters from linear regression fall outside the 
            specified bounds. In this case, nonlinear optimisation is 
            recommended.
        """
        s = self.__str__(simple=True).removesuffix(".")
        msg = f"Performing linear fit on {s}:"

        if bg_flux is None:
            bg_flux = self.default_bg

        if self.is_empty:
            log = logger.info
            msg += " No continuum windows -> performing global fit!"
            covered = False
        else:
            log = logger.debug
            covered = True

        masked_coords = self.getMaskedCoords(
            original=False,
            mode="c",
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=True,
            log_valid=True,
            bg_flux=bg_flux,
            shrink=None, # Disable res. kernel corrections
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

        prev_model = self.getModel(suffix=None)
        if prev_model is None:
            msg += "[NOTE: could not find any model, instantiating one] "
            self.instantiateModel(
                update_flux=False,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                suffix="raw",
                bg_flux=bg_flux,
                flux_bounds=flux_bounds,
                alpha_bounds=alpha_bounds,
            )
            prev_model = self.fit_raw
        else:
            msg += "[NOTE: found a model] "

        with stopwatch() as watch:
            try:
                fit = prev_model.from_linear_fit.__wrapped__(
                    prev_model,
                    masked_coords.x_log,
                    masked_coords.y_log,
                    masked_coords.dy_log,
                )
            except ValueError as e:
                msg += f"Unexpected error during linear regression: {e}"
                logger.info(msg)
                return False
            except OutsideBoundsError as e:
                msg += "linear regression produced parameters outside bounds. "
                msg += "Unfitted model will be set to the 'fit_raw' attribute, "
                msg += "but nonlinear optimisation is recommended."
                logger.info(msg)
                self.applyFit.__wrapped__(
                    self,
                    prev_model,
                    suffix='raw',
                    update_emission=True,
                )
                raise OutsideBoundsError(e) from e

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
        *,
        without_absorption: bool = True,
        bg_flux: BackgroundFlux | None = None,
        suffix: Suffix | None = "raw",
        sigmas: list[float] | None = None,
        flux_bounds: AstropyBounds | None = None,
        alpha_bounds: AstropyBounds | None = None,
        min_fittable_total: int | None = None,
    ) -> Literal[-1, 0] | int:
        """
        Raises
        ------
        MissingModelError
            If no model can be found using the specified suffix. See `getModel` 
            for details.
        OutsideBoundsError
            If the derived parameters from linear regression (during 
            sigma-clipping iterations) fall outside the specified bounds.
        """
        logger.debug(f"Performing sigma clipping: {sigmas}")

        if bg_flux is None:
            bg_flux = self.default_bg

        if self.is_empty:
            logger.debug(">>> No continuum windows: cancelling sigma-clipping!")
            self.applyFit.__wrapped__(
                self,
                self.fit_raw.copy(),
                suffix="sc",
            )
            return -1

        self.resetRejections()
        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            original=False,
            mode="c",
            covered=True,
            without_rejections=False,
            without_absorption=False,
            valid=True,
            log_valid=True,
            bg_flux=bg_flux,
            shrink=None,
        )

        n_clips: int = 0
        n_min: int = min_fittable_total

        f = self.getModel.__wrapped__(self, suffix=suffix)
        if f is None:
            raise MissingModelError(f"No model found with suffix '{suffix}'")

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
                    original=False,
                    mode="c",
                    covered=True,
                    without_rejections=True,
                    without_absorption=without_absorption,
                    valid=True,
                    log_valid=True,
                    bg_flux=bg_flux,
                    shrink=None,
                )

                n_pix = masked_coords.size
                if n_pix < n_min:
                    if n_clips == 0:
                        logger.debug(
                            f">>> Raw fit rejected all but {n_pix} points "
                            f"(min. {n_min}): cancelling sigma-clipping and "
                            "resetting rejections!"
                        )
                        n_clips = -1
                        self.resetRejections()
                        break

                    logger.debug(
                        f">>> Fit rejected all but {n_pix} points "
                        f"(min. {n_min}): stopping after {n_clips} iterations."
                    )
                    break

                try:
                    success = self.getLinearFit.__wrapped__(
                        self,
                        without_rejections=True,
                        without_absorption=without_absorption,
                        suffix="sc",
                        bg_flux=bg_flux,
                        flux_bounds=flux_bounds,
                        alpha_bounds=alpha_bounds,
                        min_fittable_total=min_fittable_total,
                    )
                except OutsideBoundsError:
                    logger.debug(
                        ">>> Linear regression failed for iteration "
                        f"{n_clips+1}. Stopping sigma-clipping."
                    )
                    break

                if success:
                    f = self.fit_sc
                    n_clips += 1
                else:
                    logger.debug(
                        f">>> Failed fit: stopping after {n_clips} iterations."
                    )
                    break

        logger.debug(
            f">>> Finished sigma-clipping in {1e3 * watch.elapsed:.1f} ms."
        )
        return n_clips

    @validated_apply_info_to_method(subjects=("continuum", "nonlinear"))
    def performFineTuning(
        self,
        *,
        original: bool = False,
        update_flux: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        bg_flux: BackgroundFlux | None = None,
        min_fittable_total: int | None = None,
        shrink: bool | None = None,
        fitter_kwargs: FitterKwargs | None = None,
    ) -> bool:
        """
        Raises
        ------
        MissingModelError
            If no previous model can be found.
        """
        from quasar_models.modeling import Fitter, PrepareModel
        from quasar_models.utils.astropy import get_free_params

        logger.debug("Performing fine-tuning:")

        original = self._update_original_flag(
            original=original,
            log=True,
            prefix=">>> ",
        )
        if bg_flux is None:
            bg_flux = self.default_bg

        model = self.getModel.__wrapped__(self, suffix=None)
        if model is None:
            raise MissingModelError("No model found for fine-tuning.")

        logger.debug(
            ">>> Initial values: "
            f"flux={model.flux.value:.1f} ({model.flux.bounds[0]:.1f},{model.flux.bounds[1]:.1f}), "
            f"alpha={model.alpha.value:.2f} ({model.alpha.bounds[0]:.2f},{model.alpha.bounds[1]:.2f})"
        )

        if self.is_empty:
            logger.debug(">>> Performing global fit due to missing continuum windows.")

        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            original=original,
            mode="c",
            covered=not self.is_empty,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=True,
            bg_flux=bg_flux,
            shrink=shrink,
        )

        n_pix = masked_coords.x.size
        n_free_params = sum(get_free_params(model).values())

        if n_pix < max(n_free_params, min_fittable_total):
            logger.debug(
                ">>> Cancelling fine-tuning due to insufficient number of data "
                f"points: {n_pix=} < max({n_free_params=}, {min_fittable_total=})"
            )
            return False

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
                    **(fitter_kwargs or {}),
                )
                fit_info = fitter.fit_info
        except pydantic.ValidationError as e:
            logger.critical(f">>> Validation error occured: {e}")
            self.applyFit.__wrapped__(self, self.getModel())
            return False
        except Exception as e:
            logger.critical(f">>> Unexpected error occured: {e}")
            raise ValueError("Unexpected error occurred during fitting.") from e

        logger.debug(
            ">>> Final values: "
            f"flux={fit.flux.value:.1f} ({fit.flux.bounds[0]:.1f},{fit.flux.bounds[1]:.1f}), "
            f"alpha={fit.alpha.value:.2f} ({fit.alpha.bounds[0]:.2f},{fit.alpha.bounds[1]:.2f})"
        )
        self.applyFit.__wrapped__(
            self,
            fit,
            fit_info=fit_info,
            update_emission=update_flux,
        )
        return True

    @pydantic.validate_call
    def getModel(
        self,
        suffix: Suffix | None = None,
    ) -> Optional[PowerLawModel]:
        if suffix is None:
            return self.fit or self.fit_sc or self.fit_raw
        elif suffix == "sc":
            return self.fit_sc or self.fit_raw
        return self.fit_raw

    @pydantic.validate_call
    def adoptFit(
        self,
        fit: PowerLawModel,
        *,
        fit_info: OptimizeResultType | None = None,
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

    @pydantic.validate_call
    def applyFit(
        self,
        fit: Optional[PowerLawModel],
        *,
        fit_info: OptimizeResultType | None = None,
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

        return self

    def summariseContinuumFit(self) -> ContinuumFitResult | None:
        if self.fit_info is None:
            return None
        return ContinuumFitResult(
            self.fit_info.x,
            self.fit_info.jac,
            x0=self.x0,
            y0=self.y0,
        )

    def windowsplot(
        self,
        mode: Literal["residual", "difference"],
        model: Optional[PowerLawModel] = None,
        lazy: bool = False,
        **plotter_kwargs,
    ):
        """
        ...
        """
        from quasar_decrypt.plotting.windows_plots import (
            ContinuumWindowsPlotWithDifferences,
            ContinuumWindowsPlotWithResiduals,
        )
        if model is None:
            model = self.getModel()

        args = (model, self.window_bounds, self)
        cls = ContinuumWindowsPlotWithResiduals \
            if mode == "residual" \
            else ContinuumWindowsPlotWithDifferences

        return self._plot_helper(
            plotter_cls=cls,
            lazy=lazy,
            args=args,
            plotter_kwargs=plotter_kwargs,
        )