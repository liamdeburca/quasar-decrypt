__all__ = ['ContinuumWindows']

from logging import getLogger
from typing import Self, ClassVar, Iterable
from dataclasses import dataclass, field
from scipy.optimize import OptimizeResult
from numpy import zeros_like

from .cwindow import CWindow
from ..utils.speclist import SpecList
from ..utils.general import stopwatch

from pydantic import ValidationError

from quasar_typing.bounds import CoordBounds, AstropyBounds
from quasar_typing.astropy import FitterInstance, FitInfo
from quasar_typing.misc import BackgroundFlux, Suffix

from quasar_utils.decorators import validate_call, validated_apply_info_to_method
from quasar_utils.continuum_fit_result import ContinuumFitResult

from quasar_models.continuum import PowerLawModel
from quasar_models.utils.astropy import apply_bounds, get_free_params

from quasar_errors.model_samples import PowerLawSample

logger = getLogger(__name__)

@dataclass(init=False)
class ContinuumWindows(SpecList[CWindow]):
    fit_info: FitInfo | None = field(default=None, init=False)
    fit_raw: PowerLawModel | None = field(default=None, init=False)
    fit_sc: PowerLawModel | None = field(default=None, init=False)
    fit: PowerLawModel | None = field(default=None, init=False)

    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({'all', 'pl'})

    @property
    def sample(self) -> PowerLawSample | None:
        if (model := self.getModel()) is None:
            return None
        return PowerLawSample.fromPowerLawModel(model)

    @validated_apply_info_to_method(
        subjects=('continuum',), 
        specific_kwargs={'windows'},
    )
    def populate(
        self,
        *,
        windows: Iterable[CoordBounds] | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD // INFO APPLIED TO METHOD **
        """
        kwargs = {}
        if self.spectrum is None:
            coords_or_spectrum = self._coords
            kwargs['info'] = self.info
            kwargs['y_smooth'] = self._y_smooth
            kwargs['y_pl'] = self._y_pl
            kwargs['y_fe'] = self._y_fe
            kwargs['y_ba'] = self._y_ba
            kwargs['y_hg'] = self._y_hg
            kwargs['y_em'] = self._y_em
            kwargs['rejected_pixels'] = self._rejected_pixels
            kwargs['absorbed_pixels'] = self._absorbed_pixels
            kwargs['valid_pixels'] = self._valid_pixels
            kwargs['log_valid_pixels'] = self._log_valid_pixels
            kwargs['p_absorbed'] = self._p_absorbed
            kwargs['x0'] = self.x0
            kwargs['y0'] = self.y0
            kwargs['x_log'] = self._x_log
            kwargs['y_log'] = self._y_log
            kwargs['dy_log'] = self._dy_log
            kwargs['get_mask'] = self.get_mask
        else:
            coords_or_spectrum = self.spectrum

        for x_bounds in windows:
            cwindow = CWindow(coords_or_spectrum, x_bounds=x_bounds, **kwargs)
            if cwindow.size > 0:
                self.append(cwindow)

        return self
    
    @validated_apply_info_to_method(subjects=('continuum', 'nonlinear'))
    def __call__(
        self,
        bg_flux: BackgroundFlux | None = None,
        *,
        sigmas: list[float] | None = None,
        flux_bounds: AstropyBounds | None = None,
        alpha_bounds: AstropyBounds | None = None,
        fitter: FitterInstance | None = None,
    ) -> bool:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        logger.debug(f"Starting pipeline for {self.__str__(True)}")
        
        if bg_flux is None:
            bg_flux = self.default_bg

        with stopwatch() as watch:
            logger.debug(">>> [1/3] Getting initial (linear) fit.")
            success = self.getLinearFit.__wrapped__(
                self,
                without_rejections = False,
                without_absorption = True,
                suffix = 'raw',
                bg_flux = bg_flux,
                flux_bounds = flux_bounds,
                alpha_bounds = alpha_bounds,
            )
            if not success: 
                return False

            logger.debug(">>> [2/3] Performing sigma-clipping.")
            _ = self.performSigmaClipping.__wrapped__(
                self,
                without_absorption = True,
                bg_flux = bg_flux,
                sigmas = sigmas,
                flux_bounds = flux_bounds,
                alpha_bounds = alpha_bounds,
            )
            
            logger.debug(">>> [3/3] Fine tuning.")
            _ = self.performFineTuning.__wrapped__(
                self,
                update_flux = True,
                without_rejections = True,
                without_absorption = True,
                suffix = 'sc',
                bg_flux = bg_flux,
                flux_bounds = flux_bounds,
                alpha_bounds = alpha_bounds,
                fitter = fitter,
            )

        logger.debug(
            "Finished entire pipeline in {:.1f} ms." \
            .format(1e3 * watch.elapsed)
        )
        return True
    
    @validated_apply_info_to_method(subjects=('continuum',))
    def getLinearFit(
        self,
        *,
        without_rejections: bool = True,
        without_absorption: bool = True,
        suffix: Suffix | None = 'raw',
        bg_flux: BackgroundFlux | None = None,
        flux_bounds: AstropyBounds | None = None,
        alpha_bounds: AstropyBounds | None = None,
    ) -> bool:
        """
        ** PYDANTIC VALIDATED METHOD **
        """      
        s = self.__str__(simple=True).removesuffix('.')
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

        coords = self.getMaskedCoords.__wrapped__(
            self,
            covered = covered,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
            valid = True,
            log_valid = True,
            bg_flux = bg_flux,
        )
        msg += f" No. of data points: {len(coords[0])}."

        if self.fit_raw is None:
            prev_model = PowerLawModel(
                self.x0, 
                self.y0, 
                apply_bounds.__wrapped__(coords[1].mean(), flux_bounds),
                apply_bounds.__wrapped__(0, alpha_bounds),
                name='powerlaw_model',
            )
            prev_model.flux.bounds = flux_bounds
            prev_model.alpha.bounds = alpha_bounds
        else:
            prev_model = self.getModel.__wrapped__(self, suffix=suffix)
        
        with stopwatch() as watch:
            fit = prev_model.from_linear_fit(*coords)

        msg += " Finished linear fit in {:.1f} ms.".format(1e3 * watch.elapsed)
        log(msg)

        self.applyFit.__wrapped__(
            self, 
            fit, 
            OptimizeResult(),
            suffix=suffix,
        )
        self.updateContinuumEmission.__wrapped__(self, fit)
        return True

    @validated_apply_info_to_method(subjects=('continuum',))
    def performSigmaClipping(
        self,
        without_absorption: bool = True,
        bg_flux: BackgroundFlux | None = None,
        *,
        sigmas: list[float] | None = None,
        flux_bounds: AstropyBounds | None = None,
        alpha_bounds: AstropyBounds | None = None,
    ) -> int:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        s = self.__str__(simple=True).removesuffix('.')
        msg = f"Performing sigma clipping on {s}: "
        
        if bg_flux is None:
            bg_flux = self.default_bg

        if self.fit_raw is None:
            msg += " (got initial linear fit)."
            success = self.getLinearFit.__wrapped__(
                self,
                without_rejections = False,
                without_absorption = True,
                suffix = 'raw',
                bg_flux = bg_flux,
                flux_bounds = flux_bounds,
                alpha_bounds = alpha_bounds,
            )
            if not success:
                msg += " Failed initial fit: cancelling sigma-clipping!"
                return 0

        if self.is_empty:
            msg += " No continuum windows: cancelling sigma-clipping!"
            logger.warning(msg)
            self.applyFit.__wrapped__(
                self, self.fit_raw.copy(), OptimizeResult(), suffix='sc',
            )
            return 0

        self.resetRejections()
        mask = self.getMask.__wrapped__(
            self,
            covered = True,
            valid = True,
            log_valid = True,
        )
        coords = self.getMaskedCoords.__wrapped__(
            self,
            covered = True,
            valid = True,
            log_valid = True,
            bg_flux = bg_flux,
        )

        out: bool = 0
        f  = self.fit_raw
        with stopwatch() as watch:
            for sigma in sigmas:
                rejections = zeros_like(self._x, dtype=bool)

                z = f.getResiduals.__wrapped__(
                    f,
                    *coords,
                    log = True,
                )
                rejections[mask] = (abs(z) > sigma)
                self.applyRejections.__wrapped__(
                    self, rejections, enforce=True,
                )
                success = self.getLinearFit.__wrapped__(
                    self,
                    without_rejections = True,
                    without_absorption = without_absorption,
                    suffix = 'sc',
                    flux_bounds = flux_bounds,
                    alpha_bounds = alpha_bounds,
                )
                if success:
                    f = self.fit_sc
                    out += 1
                else:
                    msg += " Failed fit: stopping sigma-clipping early!"
                    break

        msg += " Finished sigma-clipping in {:.1f} ms."\
            .format(1e3 * watch.elapsed)
        logger.debug(msg)

        return out

    @validated_apply_info_to_method(subjects=('continuum', 'nonlinear'))
    def performFineTuning(
        self,
        *,
        update_flux: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        suffix: Suffix = 'sc',
        bg_flux: BackgroundFlux | None = None,
        flux_bounds: AstropyBounds | None = None,
        alpha_bounds: AstropyBounds | None = None,
        fitter: FitterInstance | None = None,
    ) -> bool:      
        """
        ** PYDANTIC VALIDATED METHOD **
        """  
        s = self.__str__(simple=True).removesuffix('.')
        msg = f"Performing fine-tuning on {s}: "

        if bg_flux is None:
            bg_flux = self.default_bg

        if self.fit_raw is None:
            msg += "(got initial linear fit), "
            success = self.getLinearFit.__wrapped__(
                self,
                without_rejections = True,
                without_absorption = True,
                suffix = 'raw',
                bg_flux = bg_flux,
                flux_bounds = flux_bounds,
                alpha_bounds = alpha_bounds,
            )
            if not success:
                logger.warning(
                    msg + "Failed initial fit: cancelling fine-tuning!",
                )
                return False

        model = self.getModel.__wrapped__(
            self, suffix=suffix,
        )
        msg += "flux={:.1f} ({:.1f},{:.1f})".format(
            model.flux.value, *model.flux.bounds,
        )
        msg += " alpha={:.2f} | ({:.2f},{:.2f})".format(
            model.alpha.value, *model.alpha.bounds,
        )

        if self.is_empty:
            msg += "performing global fit due to missing continuum windows -> "

        coords = self.getMaskedCoords.__wrapped__(
            self,
            covered=not self.is_empty,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=True,
            bg_flux=bg_flux,
        )
        n_pix = coords[0].size
        n_free_params = sum(get_free_params(model).values())

        if n_pix <= n_free_params:
            msg += "cancelling fine-tuning due to insufficient no. of data " \
                f"points (n_pix={n_pix} <= n_free_params={n_free_params})!"
            self.applyFit.__wrapped__(
                self, self.getModel(), OptimizeResult(),
            )
            logger.warning(msg)
            return False
        
        try:
            with stopwatch() as watch:
                fit, fit_info = fitter(model, *coords, inplace=False)
            msg += "Successfully performed fine-tuning in {:.1f} ms: " \
                "flux={:.1f} ({:.1f},{:.1f}), " \
                "alpha={:.2f} | ({:.2f},{:.2f})" \
                .format(
                    1e3 * watch.elapsed, 
                    fit.flux.value, *fit.flux.bounds, 
                    fit.alpha.value, *fit.alpha.bounds,
                )
            logger.debug(msg)
        except ValidationError as e:
            msg += f"failed fitting due to a validation error: {e}"
            logger.warning(msg)
            self.applyFit.__wrapped__(
                self, self.getModel(), OptimizeResult(),
            )
            return False
        except Exception as e:
            msg += f"failed fitting due to an unexpected error: {e}"
            logger.warning(msg)
            self.applyFit.__wrapped__(
                self, self.getModel(), OptimizeResult(),
            )
            return False

        if self.is_empty:
            msg += " No continuum windows -> performing global fit!"
            coords = self.getMaskedCoords.__wrapped__(
                self,
                without_rejections = without_rejections,
                without_absorption = without_absorption,
                valid = True,
            )
        else:
            coords = self.getMaskedCoords.__wrapped__(
                self,
                covered = True,
                without_rejections = without_rejections,
                without_absorption = without_absorption,
                valid = True,
            )

        self.applyFit.__wrapped__(self, fit, fit_info)
        if update_flux:
            self.updateContinuumEmission.__wrapped__(self, fit)
        return True

    @validate_call
    def getModel(
        self,
        suffix: Suffix | None = None,
    ) -> PowerLawModel | None:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if self.fit_raw is None:
            return None
        
        if (suffix is None):
            if self.fit is not None: 
                return self.fit
            return self.getModel.__wrapped__(self, suffix='sc')
        elif suffix == 'sc':
            return self.fit_raw if self.fit_sc is None else self.fit_sc
        else:
            return self.fit_raw
            
    @validate_call
    def applyFit(
        self,
        fit: PowerLawModel,
        fit_info: FitInfo | None,
        suffix: Suffix | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self.fit_info = fit_info

        match suffix:
            case 'raw':
                self.fit_raw = fit
            case 'sc': 
                self.fit_sc = fit
            case None:
                self.fit = fit
        
        for window in self:
            window.applyFit.__wrapped__(window, fit, fit_info, suffix=suffix)

        return self

    @validate_call
    def adoptFit(
        self,
        fit: PowerLawModel,
        suffix: Suffix | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self.fit_info = None

        match suffix:
            case 'raw':
                self.fit_raw = fit
            case 'sc': 
                self.fit_sc = fit
            case None:
                self.fit = fit

        for window in self:
            window.adoptFit.__wrapped__(window, fit, suffix=suffix)

        return self

    def summariseContinuumFit(self) -> ContinuumFitResult:
        return ContinuumFitResult(
            self.fit_info.x,
            self.fit_info.param_cov,
            x0 = self.x0,
        )