from logging import getLogger
from typing import Self, ClassVar, Iterable, Literal, Optional
from numpy import dot, inf
from dataclasses import field
from pydantic.dataclasses import dataclass

from pydantic import ValidationError

from quasar_decrypt.host import HWindow
from quasar_decrypt.utils.speclist import SpecList

from quasar_utils.fitting import FitterInstance
from quasar_utils.decorators import validate_call, validated_apply_info_to_method

from quasar_typing.bounds import CoordBounds, AstropyBounds
from quasar_typing.pathlib import AbsoluteFITSPath
from quasar_typing.astropy import FitInfo
from quasar_typing.misc import BackgroundFlux

from quasar_models.host import (
    HostGalaxyTemplate, 
    HostGalaxyModel,
)
from quasar_models.host.io import convert_params_to_name
from quasar_models.utils.astropy import get_free_params
from quasar_models.utils.prepare_model import PrepareModel

logger = getLogger(__name__)

@dataclass(init=False)
class HostWindows(SpecList[HWindow]):
    templates: dict[str, HostGalaxyTemplate] = field(default_factory=dict, kw_only=True)
    models: dict[str, HostGalaxyModel] = field(default_factory=dict, kw_only=True)

    model: Optional[HostGalaxyModel] = field(default=None, init=False)
    fit: Optional[HostGalaxyModel] = field(default=None, init=False)
    fit_info: FitInfo | None = field(default=None, init=False)

    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({'all', 'hg'})

    @property
    def sample(self) -> None:
        raise NotImplementedError

    @validated_apply_info_to_method(
        subjects=('host',), 
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
            hwindow = HWindow.create.__wrapped__(HWindow, **kwargs)
            if hwindow.size > 0:
                self.append(hwindow)

        return self

    @validated_apply_info_to_method(subjects=('host', 'nonlinear'))
    def __call__(
        self,
        *,
        template_model: Optional[HostGalaxyModel] = None,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,

        covered: bool = False,

        template_files: list[str | AbsoluteFITSPath] | None = None,
        sources: list[Literal['bc2003']] | None = None,
        ages: list[int] | None = None,

        flux: float | None = None,
        fwhm: float | None = None,
        flux_bounds: AstropyBounds | None = None,
        fwhm_bounds: AstropyBounds | None = None,
        allow_interp_fitting: bool | None = None,
        fixed: dict[str, bool] | None = None,

        raster: bool = True,
        fine_tune: bool = True,
        only_model: bool = False,

        min_fittable_ratio: float | None = None,
        min_fittable_total: int | None = None,
        fitter: FitterInstance | None = None,
    ) -> bool:
        if bg_flux is None:
            bg_flux = self.default_bg

        if template_model is not None:
            self.applyFit.__wrapped__(
                self,
                template_model,
            )
        else:
            self.loadHostGalaxyTemplates.__wrapped__(
                self,
                sources=sources,
                ages=ages,
                template_files=template_files,
            )
            self.instantiateModels.__wrapped__(
                self,
                flux=flux,
                fwhm=fwhm,
                allow_interp_fitting=allow_interp_fitting,
                fixed=fixed,
                flux_bounds=flux_bounds,
                fwhm_bounds=fwhm_bounds,
            )

        if raster:
            try:
                self.getRasterFit.__wrapped__(
                    self,
                    without_absorption=without_absorption,
                    without_rejections=without_rejections,
                    covered=covered,
                    bg_flux=bg_flux,
                )
            except Exception:
                return False
        
        if fine_tune:
            try:
                self.performFineTuning.__wrapped__(
                    self,
                    only_model=only_model,
                    covered=covered,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                    bg_flux=bg_flux,
                    fitter=fitter,
                )
            except Exception:
                return False

        return True

    @validate_call
    def loadHostGalaxyTemplate(
        self,
        *,
        source: Literal['bc2003'] | None = None,
        age: float | None = None,
        template_file: str | AbsoluteFITSPath | None = None,
    ) -> HostGalaxyTemplate:
        """
        Loads a HostGalaxyTemplate instance.
        """
        if template_file is None:
            if source is None:
                msg = "Must specify a 'source' when 'template_file' is not provided!"
                logger.critical(msg)
                raise ValueError(msg)
            if age is None:
                msg = "Must speficy an 'age' when 'template_file' is not provided!"
                logger.critical(msg)
                raise ValueError(msg)
            
            try:
                template = HostGalaxyTemplate.load_from_cache(
                    name=source, 
                    age=age, 
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
                    path=template_file,
                    info=self.info,
                )
            except Exception as e:
                msg = f"Could not load HostGalaxyTemplate from {template_file=} due "\
                    f"to: {e}"
                logger.critical(msg)
                raise FileNotFoundError(msg)
            
        return template

    @validated_apply_info_to_method(subjects=('balmer',))
    def loadHostGalaxyTemplates(
        self,
        *,
        sources: Iterable[Literal['bc2003']] | None = None,
        ages: Iterable[int] | None = None,
        template_files: Iterable[str | AbsoluteFITSPath] | None = None,
    ) -> dict[str, HostGalaxyTemplate]:
        """
        Loads all HostGalaxyTemplate instances and adds them to the 
        'templates' dict attribute.
        """
        self.templates.clear()

        if not template_files and not (sources and ages):
            msg = "Must specify either 'template_files' or both 'sources' and "\
                "'ages' to load HostGalaxyTemplates!"
            logger.critical(msg)
            raise ValueError(msg)

        if template_files:
            for template_file in template_files:
                template = self.loadHostGalaxyTemplate.__wrapped__(
                    self,
                    template_file=template_file,
                )
                key = convert_params_to_name(template.name, template.age)

                if key in self.templates:
                    msg = f"Duplicate HostGalaxyTemplate: '{template.name}'." 
                    logger.warning(msg)
                    continue

                self.templates[key] = template

        if sources and ages:
            for source, age in zip(sources, ages):
                template = self.loadHostGalaxyTemplate.__wrapped__(
                    self,
                    source=source,
                    age=age,
                )
                key = convert_params_to_name(template.name, template.age)

                if key in self.templates:
                    msg = f"Duplicate HostGalaxyTemplate: '{template.name}'." 
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
            model = HostGalaxyModel.create(
                flux, fwhm,
                template=template,
                allow_interp_fitting=allow_interp_fitting,
                name='host_galaxy',
            )
            model.flux.fixed = fixed.get('flux', False)
            model.flux.bounds = flux_bounds

            model.fwhm.fixed = fixed.get('fwhm', False)
            model.fwhm.bounds = fwhm_bounds

            self.models[name] = model

        return self.models
    
    @validate_call
    def chooseModel(
        self,
        name: str,
    ) -> HostGalaxyModel:
        """
        Choose a HostGalaxyModel from the 'models' dict by name and set it as 
        the current model.
        """
        if name not in self.models:
            msg = f"No HostGalaxyModel named '{name}' found in 'models'!"
            logger.critical(msg)
            raise ValueError(msg)
        elif self.model is not None and name == self.model.name:
            msg = f"HostGalaxyModel '{name}' is already the current model!"
            logger.debug(msg)
            return self.model

        self.model = self.models[name]
        return self.model

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

            if covered and self.is_empty:
                msg = f"Setting {covered=} to False due to missing host windows!"
                logger.warning(msg)
                covered = False

            masked_coords = self.getMaskedCoords.__wrapped__(
                self,
                mode='c', # c: continuum
                covered=covered,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                valid=True,
                log_valid=False,
                bg_flux=bg_flux,
            )
            chi2s: dict[str, float] = {}
            for name, model in self.models.items():
                model.rasterFit.__wrapped__(
                    model, 
                    masked_coords.x,
                    masked_coords.y,
                    masked_coords.dy,
                    inplace=True,
                )
                z = (masked_coords.y - model(masked_coords.x)) / masked_coords.dy
                chi2s[name] = dot(z, z)

            best_model_name = min(chi2s.keys(), key=chi2s.get)
            best_chi2 = chi2s[best_model_name]

            msg = "Best-fit HostGalaxyModel is '{}' with chi2={:.2f}.".format(
                best_model_name, best_chi2,
            )
            logger.debug(msg)

            self.chooseModel.__wrapped__(
                self, 
                best_model_name,
            )
            self.applyFit.__wrapped__(
                self,
                self.model, 
                update_emission=True,
            )
            
        return self

    @validated_apply_info_to_method(subjects=('nonlinear',))
    def performFineTuning(
        self,
        *,
        only_model: bool = False,
        covered: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        bg_flux: BackgroundFlux | None = None,
        fitter: FitterInstance | None = None,
    ) -> Self:
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
        
        _model = next(iter(self.models.values()))
        n_pix = masked_coords.x.size
        n_free_params = sum(get_free_params(_model).values())
        if n_pix <= n_free_params:
            msg += f"cancelling fint-tuning due to insufficient no. of data "\
                f"points ({n_pix=} <= {n_free_params=})!"
            logger.critical(msg)
            raise ValueError(msg)
        
        if only_model:
            try:
                with PrepareModel(x=masked_coords.x, model=self.model, copy=True) as fit:
                    _, fit_info = fitter(
                        fit, 
                        masked_coords.x, 
                        masked_coords.y, 
                        masked_coords.dy, 
                        inplace=False,
                    )
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
                    with PrepareModel(x=masked_coords.x, model=model):
                        _, fit_info = fitter(
                            model, 
                            masked_coords.x, 
                            masked_coords.y, 
                            masked_coords.dy, 
                            inplace=True,
                        )
                    was_successful_once = True
                    chi2 = dot(fit_info.fun, fit_info.fun)
                except Exception:
                    msg += f"failed fitting model '{name}', "
                    chi2 = inf

                chi2s[name] = chi2

            if not was_successful_once:
                msg += "failed fitting all models!"
                logger.warning(msg)
                return self

            best_model_name: str = min(chi2s.keys(), key=chi2s.get)
            best_chi2 = chi2s[best_model_name]

            msg += "best-fit HostGalaxyModel is '{}' with chi2={:.2f}."\
                .format(best_model_name, best_chi2)
            logger.debug(msg)

            self.model = fit = self.models[best_model_name]

        self.applyFit.__wrapped__(self, fit, fit_info=fit_info)
        self.updateHostGalaxyEmission.__wrapped__(self, fit)
        
        return self

    def getModel(self) -> Optional[HostGalaxyModel]:
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
        fit: HostGalaxyModel,
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
        fit: HostGalaxyModel,
        *,
        fit_info: FitInfo | None = None,
        update_emission: bool = False,
    ) -> Self:
        """
        Applies the given Host Galaxy fit.
        """        
        self.fit_info = fit_info
        self.fit = fit

        if update_emission:
            self.updateHostGalaxyEmission.__wrapped__(self, fit)
            for hwindow in filter(lambda w: w._y_hg is not self._y_hg, self):
                hwindow.updateHostGalaxyEmission.__wrapped__(hwindow, fit)

        return self