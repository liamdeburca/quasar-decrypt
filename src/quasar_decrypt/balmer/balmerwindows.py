from logging import getLogger
from typing import Self, ClassVar, Iterable
from numpy import invert, linspace, stack
from functools import partial
from scipy.optimize import OptimizeResult
from dataclasses import dataclass, field

from pydantic import ValidationError

from .bwindow import BWindow
from ..utils.speclist import SpecList

from ..utils.general import stopwatch

from quasar_utils.decorators import validate_call, validated_apply_info_to_method

from quasar_typing.numpy import FloatVector
from quasar_typing.bounds import CoordBounds, AstropyBounds
from quasar_typing.pathlib import AbsoluteFITSPath
from quasar_typing.astropy import FitterInstance, FitInfo
from quasar_typing.misc import BackgroundFlux

from quasar_models.balmer import (
    BalmerSeriesTemplate, 
    BalmerContinuumTemplate, 
    BalmerModel,
)
from quasar_models.utils.astropy import apply_bounds

from quasar_errors.model_samples import BalmerSample

logger = getLogger(__name__)

@dataclass(init=False)
class BalmerWindows(SpecList[BWindow]):
    model: BalmerModel | None = field(default=None, init=False)
    fit: BalmerModel | None = field(default=None, init=False)
    fit_info: FitInfo | None = field(default=None, init=False)

    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({'all', 'ba'})

    @property
    def sample(self) -> BalmerSample | None:
        if (model := self.getModel()) is None:
            return None
        return BalmerSample.fromBalmerModel(model)

    @validated_apply_info_to_method(
        subjects=('balmer',), 
        specific_kwargs={'windows'},
    )
    def populate(
        self,
        *,
        windows: Iterable[CoordBounds] | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
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
            bwindow = BWindow(coords_or_spectrum, x_bounds=x_bounds, **kwargs)
            if bwindow.size > 0:
                self.append(bwindow)

        return self

    @validated_apply_info_to_method(subjects=('balmer', 'nonlinear'))
    def __call__(
        self,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        covered: bool = True,
        *,
        template_file: str | AbsoluteFITSPath | None = None,
        flux: float | None = None,
        fwhm: float | None = None,
        temp: float | None = None,
        tau: float | None = None,
        scale: float | None = None,
        ratio: float | None = None,
        edge: float | None = None,
        waves: FloatVector | None = None,
        weights: FloatVector | None = None,
        flux_bounds: AstropyBounds | None = None,
        fwhm_bounds: AstropyBounds | None = None,
        temp_bounds: AstropyBounds | None = None,
        tau_bounds: AstropyBounds | None = None,
        scale_bounds: AstropyBounds | None = None,
        ratio_bounds: AstropyBounds | None = None,
        allow_interp_fitting: bool | None = None,
        fixed: dict[str, bool] | None = None,
        min_fittable_ratio: float | None = None,
        min_fittable_total: int | None = None,
        raster_n: int | None = None,
        fitter: FitterInstance | None = None,
    ) -> bool:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        self.loadTemplate.__wrapped__(
            self,
            template_file=template_file,
            flux=flux,
            fwhm=fwhm,
            temp=temp,
            tau=tau,
            scale=scale,
            ratio=ratio,
            edge=edge,
            waves=waves,
            weights=weights,
            flux_bounds=flux_bounds,
            fwhm_bounds=fwhm_bounds,
            temp_bounds=temp_bounds,
            tau_bounds=tau_bounds,
            scale_bounds=scale_bounds,
            ratio_bounds=ratio_bounds,
            allow_interp_fitting=allow_interp_fitting,
            fixed=fixed,
        )
        self.checkModelCoverage.__wrapped__(
            self,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            min_fittable_ratio=min_fittable_ratio,
            min_fittable_total=min_fittable_total,
        )
        self.getRasterFit.__wrapped__(
            self,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            bg_flux=bg_flux,
            raster_n=raster_n,
            covered=covered,
        )
        self.performFineTuning.__wrapped__(
            self,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            bg_flux=bg_flux,
            fitter=fitter,
        )

    @validated_apply_info_to_method(
            subjects=('balmer',),
            specific_kwargs={'source', 'temp', 'dens', 'n_u_min', 'n_u_max'},
        )
    def loadBalmerSeriesTemplate(
        self,
        *,
        source: str | None = None,
        temp: float | None = None,
        dens: float | None = None,
        n_u_min: int | None = None,
        n_u_max: int | None = None,
        path: str | AbsoluteFITSPath | None = None,
    ) -> BalmerSeriesTemplate:
        """
        Loads a BalmerSeriesTemplate instance.
        """
        if path is None:
            BalmerSeriesTemplate.load_from_cache(
                source, temp, dens, (n_u_min, n_u_max), 
                info=self.info,
            )
        return BalmerSeriesTemplate.load(path, self.info)

    @validated_apply_info_to_method(
            subjects=('balmer',),
            specific_kwargs={'temp', 'tau', 'scale'},
        )
    def loadBalmerContinuumTemplate(
        self,
        *,
        temp: float | None = None,
        tau: float | None = None,
        scale: float | None = None,
        path: str | AbsoluteFITSPath | None = None,
    ) -> BalmerContinuumTemplate:
        """
        Loads a BalmerContinuumTemplate instance.
        """
        if path is None:
            BalmerContinuumTemplate.load_from_cache(
                temp, tau, scale,
                info=self.info,
            )
        return BalmerContinuumTemplate.load(path, self.info)

    @validated_apply_info_to_method(subjects=('balmer',))
    def instantiateModel(
        self,
        *,
        flux: float | None = None,
        fwhm: float | None = None,
        ratio: float | None = None,

        source: str | None = None,
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
        series_template = self.loadBalmerSeriesTemplate.__wrapped__(
            self,
            source=source,
            temp=temp,
            dens=dens,
            n_u_min=n_u_min,
            n_u_max=n_u_max,
        ).createLogspace(self._x, inplace=False, keep_x=True)

        continuum_template = self.loadBalmerContinuumTemplate.__wrapped__(
            self,
            temp=temp,
            tau=tau,
            scale=scale,
        ).createLogspace(self._x, inplace=False, keep_x=True)
        
        model = BalmerModel(
            flux, fwhm, ratio,
            info=self.info,
            series_template=series_template,
            continuum_template=continuum_template,
            allow_interp_fitting=allow_interp_fitting,
        )
        
        # flux
        model.flux.value = apply_bounds(flux, flux_bounds)
        model.flux.bounds = flux_bounds
        model.flux.fixed = fixed['flux']
        # fwhm
        model.fwhm.value = apply_bounds(fwhm, fwhm_bounds)
        model.fwhm.bounds = fwhm_bounds
        model.fwhm.fixed = fixed['fwhm']
        # ratio
        model.ratio.value = apply_bounds(ratio, ratio_bounds)
        model.ratio.bounds = ratio_bounds
        model.ratio.fixed = fixed['ratio']

        return model

    @validated_apply_info_to_method(subjects=('balmer',))
    def checkModelCoverage(
        self,
        *,
        without_rejections: bool = False,
        without_absorption: bool = False,
        min_fittable_ratio: float | None = None,
        min_fittable_total: int | None = None,
    ) -> bool:
        """
        ** PYDANTIC VALIDATED METHOD **

        Checks the degree of coverage of each side of the Balmer 
        pseudo-continuum.
        """
        self.model
        model: BalmerModel = self.getModel()
        assert model is not None, "Model has not been instantiated!"
        
        fittable_pixels = self.valid_pixels
        if without_rejections: 
            fittable_pixels &= invert(self.rejected_pixels)
        if without_absorption: 
            fittable_pixels &= invert(self.absorbed_pixels)

        # The blue side: Balmer continuum + attenuation contribution

        is_blue = (self.x <= model.edge)
        n_blue: int = fittable_pixels[is_blue].sum()
        r_blue: float = fittable_pixels[is_blue].mean() if n_blue > 0 else 0.0
        cond1 = (r_blue < min_fittable_ratio) or (n_blue < min_fittable_total)

        is_red = invert(is_blue)
        n_red: int = fittable_pixels[is_red].sum()
        r_red: float = fittable_pixels[is_red].mean() if n_red > 0 else 0.0
        cond2 = (r_red < min_fittable_ratio) or (n_red < min_fittable_total)

        if cond1 and cond2:
            msg = "Insufficient coverage on both sides of the Balmer "\
                "pseudo-continuum: removing model!"
            logger.debug(msg)
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
        assert self.model is not None, "Model has not been instantiated!"

        if bg_flux is None:
            bg_flux = self.default_bg

        coords = self.getMaskedCoords.__wrapped__(
            self,
            covered = covered,
            log = False,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
            valid = True,
            log_valid = False,
            bg_flux = bg_flux,
        )
        self.model.rasterFit.__wrapped__(
            self.model,
            *coords,
        )
        return self

    @validated_apply_info_to_method(subjects=('nonlinear',))
    def performFineTuning(
        self,
        without_rejections: bool = False,
        without_absorption: bool = False,
        bg_flux: BackgroundFlux | None = None,
        *,
        fitter: FitterInstance | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        msg = f"Performing fine-tuning fit on{self.__str__(True)}: "

        if bg_flux is None:
            bg_flux = self.default_bg

        model = self.getModel()
        if model is None:
            msg = "No BalmerModel instance available!"
            raise ValueError(msg)
        
        coords = self.getMaskedCoords.__wrapped__(
            self,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            covered=True,
            log=False,
            valid=True,
            log_valid=False,
            bg_flux=bg_flux,
        )
        try:
            with stopwatch() as watch:
                fit, fit_info = fitter(
                    model,
                    *coords,
                    inplace=False,
                )
            msg += f"Finished fine-tuning in {1e3*watch.elapsed:.1f} ms."
        except ValidationError as e:
            msg += f"Failed fitting due to validation error: {e}"
            logger.critical(msg)

            fit = model
            fit_info = OptimizeResult()

        self.applyFit.__wrapped__(self, fit, fit_info)
        self.updateBalmerEmission.__wrapped__(self, fit)

        return self
        

    def getModel(self) -> BalmerModel | None:
        """
        Retrieves the Balmer pseudo-continuum model if available.

        Notes
        -----
        If a fitted Balmer pseudo-continuum model is available, it is
        returned. Otherwise an instantiated model is returned, if available.
        """
        return self.fit or self.model
    
    @validate_call
    def applyFit(
        self,
        fit: BalmerModel,
        fit_info: FitInfo,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **

        Applies the given Balmer pseudo-continuum fit.

        Notes
        -----
        This method does NOT update the Balmer pseudo-continuum emission array,
        '_y_ba'. This should be updated separately using 'updateBalmerEmission'.
        """
        self.fit = fit
        self.fit_info = fit_info

        for bwindow in self:
            bwindow.applyFit.__wrapped__(bwindow, fit, fit_info)

        return self
    
    @validate_call
    def adoptFit(
        self,
        fit: BalmerModel,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self.fit = fit
        self.fit_info = None

        for bwindow in self:
            bwindow.adoptFit.__wrapped__(bwindow, fit)

        return self