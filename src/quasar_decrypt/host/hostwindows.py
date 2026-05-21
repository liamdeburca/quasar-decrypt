from logging import getLogger
from typing import Self, ClassVar, Iterable, Literal
from numpy import dot, inf
from scipy.optimize import OptimizeResult
from dataclasses import dataclass, field

from pydantic import ValidationError

from .hwindow import HWindow
from ..utils.speclist import SpecList

from ..utils.general import stopwatch

from quasar_utils.decorators import validate_call, validated_apply_info_to_method

from quasar_typing.bounds import CoordBounds, AstropyBounds
from quasar_typing.pathlib import AbsoluteFITSPath
from quasar_typing.astropy import FitterInstance, FitInfo
from quasar_typing.misc import BackgroundFlux

from quasar_models.host import (
    HostGalaxyTemplate, 
    HostGalaxyModel,
)
from quasar_models.utils.astropy import get_free_params

# from quasar_errors.model_samples

logger = getLogger(__name__)

@dataclass(init=False)
class HostWindows(SpecList[HWindow]):

    templates: dict[str, HostGalaxyTemplate] = field(default_factory=dict, init=False)
    models: dict[str, HostGalaxyModel] = field(default_factory=dict, init=False)

    model: HostGalaxyModel | None = field(default=None, init=False)
    fit: HostGalaxyModel | None = field(default=None, init=False)
    fit_info: FitInfo | None = field(default=None, init=False)

    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({'all', 'hg'})

    @property
    def sample(self) -> None:
        if (model := self.getModel()) is None:
            return None
        return None

    @validated_apply_info_to_method(
        subjects=('host',), 
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
            bwindow = HWindow(coords_or_spectrum, x_bounds=x_bounds, **kwargs)
            if bwindow.size > 0:
                self.append(bwindow)

        return self

    @validated_apply_info_to_method(subjects=('host', 'nonlinear'))
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
        flux_bounds: AstropyBounds | None = None,
        fwhm_bounds: AstropyBounds | None = None,
        allow_interp_fitting: bool | None = None,
        fixed: dict[str, bool] | None = None,
        min_fittable_ratio: float | None = None,
        min_fittable_total: int | None = None,
        fitter: FitterInstance | None = None,
    ) -> bool:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if bg_flux is None:
            bg_flux = self.default_bg

    @validate_call
    def loadHostGalaxyTemplate(
        self,
        *,
        source: Literal['bc2003'] | None = None,
        age: float | None = None,
        path: str | AbsoluteFITSPath | None = None,
    ) -> HostGalaxyTemplate:
        """
        Loads a HostGalaxyTemplate instance.
        """
        if path is None:
            if source is None:
                msg = "Must specify a 'source' when 'path' is not provided!"
                logger.critical(msg)
                raise ValueError(msg)
            if age is None:
                msg = "Must speficy an 'age' when 'path' is not provided!"
                logger.critical(msg)
                raise ValueError(msg)
            
            try:
                template = HostGalaxyTemplate.load_from_cache(
                    source, 
                    age, 
                    info=self.info,
                )
            except Exception as e:
                msg = f"Could not load HostGalaxyTemplate with {source=}, "\
                    f"{age=} due to: {e}"
                logger.critical(msg)
                raise FileNotFoundError(msg)
        else:
            try:
                template = HostGalaxyTemplate.load(
                    path,
                    info=self.info,
                )
            except Exception as e:
                msg = f"Could not load HostGalaxyTemplate from {path=} due "\
                    f"to: {e}"
                logger.critical(msg)
                raise FileNotFoundError(msg)
            
        return template

    @validated_apply_info_to_method(subjects=('balmer',))
    def instantiateModel(
        self,
        *,
        sources: Iterable[Literal['bc2003']] | None = None,
        ages: Iterable[float] | None = None,
        paths: Iterable[str | AbsoluteFITSPath] | None = None,
    ) -> dict[str, HostGalaxyTemplate]:
        """
        Loads all HostGalaxyTemplate instances and adds them to the 
        'templates' dict attribute.
        """
        self.templates.clear()
        if paths is None:
            if sources is None:
                msg = "Must specify 'sources' when 'paths' is not provided!"
                logger.critical(msg)
                raise ValueError(msg)
            if ages is None:
                msg = "Must specify 'ages' when 'paths' is not provided!"
                logger.critical(msg)
                raise ValueError(msg)
            
            for source, age in zip(sources, ages):
                try:
                    template = self.loadHostGalaxyTemplate.__wrapped__(
                        self,
                        source=source,
                        age=age,
                    )
                except Exception:
                    pass

                key = template.path.stem
                if key in self.templates:
                    msg = f"Duplicate HostGalaxyTemplate with key '{key}'." 
                    logger.warning(msg)
                    continue

                self.templates[key] = template
        else:
            for path in paths:
                try:
                    template = self.loadHostGalaxyTemplate.__wrapped__(
                        self,
                        path=path,
                    )
                except Exception:
                    pass

                key = template.path.stem
                if key in self.templates:
                    msg = f"Duplicate HostGalaxyTemplate with key '{key}'." 
                    logger.warning(msg)
                    continue

                self.templates[key] = template

        return self.templates
    
    @validated_apply_info_to_method(subjects=('balmer',))
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
            model = HostGalaxyModel(
                flux, fwhm,
                info=self.info,
                template=template,
                allow_interp_fitting=allow_interp_fitting,
                name=name,
            )
            model.flux.fixed = fixed.get('flux', False)
            model.flux.bounds = flux_bounds

            model.fwhm.fixed = fixed.get('fwhm', False)
            model.fwhm.bounds = fwhm_bounds

            self.models[name] = model

        return self.models
    
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

            coords = self.getMaskedCoords.__wrapped__(
                self,
                covered=covered,
                log=False,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                valid=True,
                log_valid=False,
                bg_flux=bg_flux,
            )
            chi2s: dict[str, float] = {}
            for name, model in self.models.items():
                model.rasterFit.__wrapped__(
                    model, *coords, 
                    inplace=True,
                )
                z = (coords[1] - model(coords[0])) / coords[2]
                chi2s[name] = dot(z, z)

            best_model_name = min(chi2s.keys(), key=chi2s.get)
            best_chi2 = chi2s[best_model_name]

            msg = "Best-fit HostGalaxyModel is '{}' with chi2={:.2f}.".format(
                best_model_name, best_chi2,
            )
            logger.debug(msg)

            self.model = self.models[best_model_name]
            self.applyFit(self.model, OptimizeResult())
            
        return self

    @validated_apply_info_to_method(subjects=('nonlinear',))
    def performFineTuning(
        self,
        only_model: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        bg_flux: BackgroundFlux | None = None,
        *,
        fitter: FitterInstance | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        s = self.__str__(simple=True).removesuffix('.')
        msg = f"Performing fine-tuning fit on {s}: "

        if not self.models:
            msg += "no HostGalaxyModel instances available for fine-tuning!"
            logger.critical(msg)
            raise ValueError(msg)

        if only_model and self.model is None:
            msg += "setting `only_model=True` to `False` due to missing "\
                "`self.model`, "
            logger.warning(msg)
            only_model = False

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
        
        _model = next(self.models.values())
        n_pix = coords[0].size
        n_free_params = sum(get_free_params(_model).values())
        if n_pix <= n_free_params:
            msg += f"cancelling fint-tuning due to insufficient no. of data "\
                f"points ({n_pix=} <= {n_free_params=})!"
            logger.critical(msg)
            raise ValueError(msg)
        
        if only_model:
            try:
                fit, fit_info = fitter(self.model, *coords, False)
            except ValidationError as e:
                msg += f"failed fitting due to validation error: {e}"
                logger.warning(msg)
                return self
            except Exception as e:
                msg += f"failed fitting due to an unexpected error: {e}"
                logger.warning(msg)
                return self
        else:
            chi2s: dict[str, float] = {}
            was_successful_once = False
            for name, model in self.models.items():
                try:
                    _, fit_info = fitter(model, *coords, True)
                    was_successful_once = True
                except Exception:
                    msg += f"failed fitting model '{name}', "
                    chi2s[name] = inf

                z = (coords[1] - model(coords[0])) / coords[2]
                chi2s[name] = dot(z, z)

            if not was_successful_once:
                msg += "failed fitting all models!"
                logger.warning(msg)
                return self

            best_model_name = min(chi2s.keys(), key=chi2s.get)
            best_chi2 = chi2s[best_model_name]

            msg += "best-fit HostGalaxyModel is '{}' with chi2={:.2f}.".format(
                best_model_name, best_chi2,
            )
            logger.debug(msg)

            self.model = fit = self.models[best_model_name]

        self.applyFit.__wrapped__(self, fit, fit_info)
        self.updateHostGalaxyEmission.__wrapped__(self, fit)
        return self

    def getModel(self) -> HostGalaxyModel | None:
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
        fit: HostGalaxyModel,
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
        fit: HostGalaxyModel,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self.fit = fit
        self.fit_info = None

        for bwindow in self:
            bwindow.adoptFit.__wrapped__(bwindow, fit)

        return self