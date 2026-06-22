from logging import getLogger
from typing import Self, ClassVar, Iterable, Literal, Optional
from numpy import invert
from dataclasses import dataclass, field

from pydantic import ValidationError

from quasar_decrypt.utils.speclist import SpecList
from quasar_decrypt.balmer.bwindow import BWindow
from quasar_decrypt.utils.general import stopwatch

from quasar_utils.decorators import validate_call, validated_apply_info_to_method
from quasar_utils.fitting import FitterInstance

from quasar_typing.bounds import CoordBounds, AstropyBounds
from quasar_typing.pathlib import AbsoluteFITSPath
from quasar_typing.astropy import FitInfo
from quasar_typing.misc import BackgroundFlux

from quasar_models.balmer import (
    BalmerSeriesTemplate, 
    BalmerContinuumTemplate, 
    BalmerModel,
)
from quasar_models.utils.astropy import apply_bounds
from quasar_models.utils.prepare_model import PrepareModel

from quasar_errors.model_samples import BalmerSample

logger = getLogger(__name__)

@dataclass(init=False)
class BalmerWindows(SpecList[BWindow]):
    model: Optional[BalmerModel] = field(default=None, init=False)
    fit: Optional[BalmerModel] = field(default=None, init=False)
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
        kwargs = {}
        if self.spectrum is None:
            kwargs['x'] = self._x
            kwargs['y'] = self._y
            kwargs['dy'] = self._dy
            kwargs['dx'] = self._dx

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

            kwargs['info'] = self.info
            kwargs['get_mask'] = self.get_mask
        else:
            kwargs['spectrum'] = self.spectrum

        for x_bounds in windows:
            kwargs['x_bounds'] = x_bounds
            bwindow = BWindow.create.__wrapped__(BWindow, **kwargs)
            if bwindow.size > 0:
                self.append(bwindow)

        return self

    @validated_apply_info_to_method(subjects=('balmer', 'nonlinear'))
    def __call__(
        self,
        *,
        template_model: Optional[BalmerModel] = None,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        
        covered: bool = True,

        flux: float | None = None,
        fwhm: float | None = None,
        ratio: float | None = None,
        source: Literal['sh1995'] | None = None,
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

        fitter: FitterInstance | None = None,
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

        if raster:
            try:
                self.getRasterFit.__wrapped__(
                    self,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                    covered=covered,
                    bg_flux=bg_flux,
                )
            except Exception:
                return False
            
        if fine_tune:
            try:
                self.performFineTuning.__wrapped__(
                    self,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                    covered=covered,
                    bg_flux=bg_flux,
                    fitter=fitter,
                )
            except Exception:
                return False
    
        return True

    @validate_call
    def loadBalmerSeriesTemplate(
        self,
        *,
        source: Literal['sh1995'] | None = None,
        temp: float | None = None,
        dens: float | None = None,
        n_u_min: int | None = None,
        n_u_max: int | None = None,
        template_file: str | AbsoluteFITSPath | None = None,
    ) -> BalmerSeriesTemplate:
        """
        Loads a BalmerSeriesTemplate instance.
        """
        if template_file is None:
            if source is None:
                msg = "Must specify 'source' if 'template_file' is not provided!"
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
            
            return BalmerSeriesTemplate.load_from_cache(
                name=source, 
                temp=temp, 
                dens=dens, 
                n_u_range=(n_u_min, n_u_max),
                info=self.info,
            )
        
        return BalmerSeriesTemplate.load(
            path=template_file, 
            info=self.info,
        )

    @validate_call
    def loadBalmerContinuumTemplate(
        self,
        *,
        temp: float | None = None,
        tau: float | None = None,
        scale: float | None = None,
        template_file: str | AbsoluteFITSPath | None = None,
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

    @validated_apply_info_to_method(subjects=('balmer',))
    def instantiateModel(
        self,
        *,
        flux: float | None = None,
        fwhm: float | None = None,
        ratio: float | None = None,

        source: Literal['sh1995'] | None = None,
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
        ).createLogspace(
            sigma_res=self.info.loading.sigma_res,
            xr=self._x,
            keep_x=True,
        )

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
        
        self.model = BalmerModel.create(
            flux, fwhm, ratio,
            edge=self.info.balmer.edge,
            continuum_template=continuum_template,
            series_template=series_template,
            allow_interp_fitting=allow_interp_fitting,
            name='balmer',
        )
        
        # flux
        self.model.flux.value = apply_bounds.__wrapped__(flux, flux_bounds)
        self.model.flux.bounds = flux_bounds
        self.model.flux.fixed = fixed.get('flux', False)
        # fwhm
        self.model.fwhm.value = apply_bounds.__wrapped__(fwhm, fwhm_bounds)
        self.model.fwhm.bounds = fwhm_bounds
        self.model.fwhm.fixed = fixed.get('fwhm', False)
        # ratio
        self.model.ratio.value = apply_bounds.__wrapped__(ratio, ratio_bounds)
        self.model.ratio.bounds = ratio_bounds
        self.model.ratio.fixed = fixed.get('ratio', False)

        return self.model

    @validated_apply_info_to_method(subjects=('balmer',))
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
        is_blue = (self._x <= model.edge)
        n_blue: int = fittable_pixels[is_blue].sum()
        r_blue: float = fittable_pixels[is_blue].mean() if n_blue > 0 else 0.0
        cond1 = (r_blue < min_fittable_ratio) or (n_blue < min_fittable_total)

        # The red side: Balmer series contribution
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

        if covered and self.is_empty:
            msg = f"Setting {covered=} to False due to missing host windows!"
            logger.warning(msg)
            covered = False

        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            mode='c', # c: contiguous
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=True,
            log_valid=False,
            bg_flux=bg_flux,
        )
        self.model.rasterFit.__wrapped__(
            self.model,
            masked_coords.x,
            masked_coords.y,
            masked_coords.dy,
            inplace=True,
        )
        return self

    @validated_apply_info_to_method(subjects=('nonlinear',))
    def performFineTuning(
        self,
        *,
        without_rejections: bool = False,
        without_absorption: bool = False,
        covered: bool = False,
        bg_flux: BackgroundFlux | None = None,
        fitter: FitterInstance | None = None,
    ) -> Self:
        msg = f"Performing fine-tuning fit on {self.__str__(True)}: "

        if bg_flux is None:
            bg_flux = self.default_bg

        model = self.getModel()
        if model is None:
            msg = "No BalmerModel instance available!"
            raise ValueError(msg)
        
        if covered and self.is_empty:
            msg += f"setting {covered=} to False due to missing host windows, "
            logger.warning(msg)
            covered = False
        
        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            mode='c', # c: contiguous
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=True,
            log_valid=False,
            bg_flux=bg_flux,
        )
        try:
            with stopwatch() as watch, \
                PrepareModel(x=masked_coords.x, model=model, copy=True) as fit:
                _, fit_info = fitter(
                    fit,
                    masked_coords.x,
                    masked_coords.y,
                    masked_coords.dy,
                    inplace=True,
                )
                
            msg += f"Finished fine-tuning in {1e3*watch.elapsed:.1f} ms."
            self.applyFit.__wrapped__(
                self, 
                fit, 
                fit_info=fit_info, 
                update_emission=True,
            )

        except ValidationError as e:
            msg += f"Failed fitting due to validation error: {e}"
            logger.critical(msg)

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
    
    @validate_call
    def adoptFit(
        self,
        fit: BalmerModel,
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
        fit: BalmerModel,
        *,
        fit_info: FitInfo | None = None,
        update_emission: bool = False,
    ) -> Self:
        """
        Applies the given Balmer pseudo-continuum fit.
        """
        self.fit = fit
        self.fit_info = fit_info
        
        if update_emission:
            self.updateBalmerEmission.__wrapped__(self, fit)

            for bwindow in filter(lambda w: w._y_ba is not self._y_ba, self):
                bwindow.updateBalmerEmission.__wrapped__(
                    bwindow,
                    fit,
                )

        return self