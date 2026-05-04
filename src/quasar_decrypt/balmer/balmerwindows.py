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

from quasar_models.balmer import BalmerModel, BalmerTemplate, evaluation
from quasar_models.utils.astropy import apply_bounds

from quasar_errors.model_samples import BalmerSample

logger = getLogger(__name__)

@dataclass(init=False)
class BalmerWindows(SpecList[BWindow]):
    template: BalmerTemplate | None = field(default=None, init=False)
    model: BalmerModel | None = field(default=None, init=False)
    fit: BalmerModel | None = field(default=None, init=False)
    fit_info: FitInfo | None = field(default=None, init=False)

    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({'all', 'ba'})

    @property
    def sample(self) -> BalmerSample | None:
        if (model := self.getModel()) is None:
            return None
        return BalmerSample.fromBalmerModel(model)

    @validated_apply_info_to_method(subjects=('balmer',), specific_kwargs={'windows'})
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

    @classmethod
    def _load_balmer_series_data(
        cls,
        *args,
    ) -> tuple[FloatVector, FloatVector]:
        """
        Loads Balmer series data (wavelengths and weights).
        """
        raise NotImplementedError

    def instantiateModel(
        self,
        *args,
    ) -> Self:
        """
        Convenience function for creating either a custom BalmerModel instance 
        or loading a template from file. 
        """
        raise NotImplementedError
    
    @validated_apply_info_to_method(subjects=('balmer',))
    def createModel(
        self,
        *,
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
        raster_n: int | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **

        Creates a BalmerModel instance using the given parameters and bounds. 
        If 'allow_interp_fitting' is True, this function will generate a custom
        BalmerTemplate instance s.t. the BalmerModel instance can perform
        template fitting by interpolation between 'raster_n' templates. 
        """
        self.model: BalmerModel = BalmerModel(
            apply_bounds.__wrapped__(flux, flux_bounds),
            apply_bounds.__wrapped__(fwhm, fwhm_bounds),
            _temp := apply_bounds.__wrapped__(temp, temp_bounds),
            _tau := apply_bounds.__wrapped__(tau, tau_bounds),
            _scale := apply_bounds.__wrapped__(scale, scale_bounds),
            _ratio := apply_bounds.__wrapped__(ratio, ratio_bounds),
            edge=edge,
            waves=waves,
            weights=weights,
            boltz=self.info.units.getBoltzmannFactor(),
            allow_interp_fitting=allow_interp_fitting,
            name='balmer_model',
        )
        if allow_interp_fitting:
            if None in fwhm_bounds:
                msg = "Tried to generated template for BalmerModel instance, \
                    but no closed bounds on FWHM parameter."
                logger.critical(msg)
            else:
                msg = "Generating template for BalmerModel instance, which may \
                    be inefficient for pipelines. Consider caching a similar \
                    BalmerTemplate instance instead." 
                logger.warning(msg)

                _fwhm = linspace(*fwhm_bounds, raster_n, endpoint=True)
                _x = self._x[self._valid_pixels]

                forward = partial(
                    evaluation.evaluate,
                    flux=1.0, temp=_temp, tau=_tau, scale=_scale, ratio=_ratio,
                    sigma_res=self.info.loading['sigma_res'],
                    edge=edge, 
                    waves=waves, weights=weights,
                    boltz=self.info.units.getBoltzmannFactor(),
                    x_grid=_x,
                )
                _data = stack([forward(_f) for _f in _fwhm], axis=0)

                self.template = BalmerTemplate(
                    _fwhm, _x, _data,
                    info=self.info,
                    is_logspace=True,
                    based_on_template=True,
                    name='generated_balmer_template',
                )
                self.model.template = self.template

        for param_name in self.model.param_names:
            getattr(self.model, param_name).fixed = fixed.get(param_name, False)

        if self.model._perform_template_fitting:
            msg = "Template fitting enabled!"
            logger.debug(msg)

        return self

    @validated_apply_info_to_method(subjects=('balmer',))
    def loadTemplate(
        self,
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
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **

        Loads a BalmerTemplate from the given file and creates a corresponding
        BalmerModel instance. 
        """
        try: 
            template = BalmerTemplate.load(template_file, info=self.info)
        except FileNotFoundError:
            msg = f"Could not find template file '{template_file}' -> Skipping!"
            logger.critical(msg)
            return self
        
        if not template.is_logspace:
            msg = f"BalmerTemplate @ {template_file} will be transformer to \
                logspace, which may be inefficient for pipelines. Consider \
                caching a logspace-equivalent of this BalmerTemplate."
            logger.warning(msg)

            template.createLogspace(
                self._x[self._valid_pixels], inplace=True, keep_x=True,
            )

        # Uses the template-assigned value by default
        #? Apply these values to the BalmerInfo instance?
        _temp    = getattr(template, 'temp', temp)
        _tau     = getattr(template, 'tau', tau)
        _scale   = getattr(template, 'scale', scale)
        _ratio   = getattr(template, 'ratio', ratio)
        _edge    = getattr(template, 'edge', edge)
        _waves   = getattr(template, 'waves', waves)
        _weights = getattr(template, 'weights', weights)
        
        self.template: BalmerTemplate = template
        self.model: BalmerModel = BalmerModel(
            apply_bounds.__wrapped__(flux, flux_bounds),
            apply_bounds.__wrapped__(fwhm, fwhm_bounds),
            apply_bounds.__wrapped__(_temp, temp_bounds),
            apply_bounds.__wrapped__(_tau, tau_bounds),
            apply_bounds.__wrapped__(_scale, scale_bounds),
            apply_bounds.__wrapped__(_ratio, ratio_bounds),

            edge=_edge,
            sigma_res=self.info.loading['sigma_res'],
            waves=_waves,
            weights=_weights,
            boltz=self.info.units.getBoltzmannFactor(),
            template=self.template,
            allow_interp_fitting=allow_interp_fitting,
            name='balmer_model',
        )
        for param_name in self.model.param_names:
            getattr(self.model, param_name).fixed = fixed.get(param_name, False)

        if self.model._perform_template_fitting:
            msg = "Template fitting enabled!"
            logger.debug(msg)

        return self

    @validated_apply_info_to_method(subjects=('balmer',))
    def checkModelCoverage(
        self,
        *,
        without_rejections: bool = False,
        without_absorption: bool = False,
        min_fittable_ratio: float | None = None,
        min_fittable_total: int | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **

        Checks the degree of coverage of each side of the Balmer 
        pseudo-continuum.
        """
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
        
        if (r_blue < min_fittable_ratio) or (n_blue < min_fittable_total):
            model._ignore_blue_side()

        # The red side: Balmer series contribution

        is_red = invert(is_blue)
        n_red: int = fittable_pixels[is_red].sum()
        r_red: float = fittable_pixels[is_red].mean() if n_red > 0 else 0.0

        if (r_red < min_fittable_ratio) or (n_red < min_fittable_total):
            model._ignore_red_side()

        return self
    
    @validated_apply_info_to_method(subjects=('balmer',), specific_kwargs={'raster_n'})
    def getRasterFit(
        self,
        *,
        raster_n: int | None = None,
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
            raster_n = raster_n,
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