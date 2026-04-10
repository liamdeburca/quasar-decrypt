from logging import getLogger
from typing import Self, Literal, Union, ClassVar, Iterable
from pathlib import Path
from scipy.ndimage import binary_fill_holes
from itertools import repeat
from dataclasses import dataclass, field
from pydantic import validate_call

from quasar_utils.wrappers import apply_info_to_method

from quasar_typing.numpy import FloatVector
from quasar_typing.bounds import CoordBounds, AstropyBounds
from quasar_typing.pathlib import AnyFITSPath
from quasar_typing.astropy import FitterInstance, FitInfo, CompoundModel_
from quasar_typing.misc.literals import BGFlux

from quasar_models.iron import IronModel, IronTemplate
from quasar_models.utils.astropy import apply_bounds

from .iwindow import IWindow
from ..utils.speclist import SpecList

logger = getLogger(__name__)
logger.disabled = not getLogger().hasHandlers()

@dataclass(init=False)
class IronWindows(SpecList):
    templates: dict[str, IronTemplate] = field(default_factory=dict, init=False)
    template_models: dict[str, IronModel] = field(default_factory=dict, init=False)
    fit_info: FitInfo | None = field(default=None, init=False)

    default_bg: ClassVar[BGFlux] = {'pl', 'ba', 'hg', 'em'}

    def __post_init__(self):
        self.templates = {}
        self.template_models = {}
        
    @validate_call(validate_return=False)
    @apply_info_to_method('iron')
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
            iwindow = IWindow(coords_or_spectrum, x_bounds=x_bounds, **kwargs)
            if iwindow.size > 0:
                self.append(iwindow)

        return self

    @validate_call(validate_return=False)
    @apply_info_to_method('iron', 'nonlinear')
    def __call__(
        self,
        without_rejections: bool = True,
        without_absorption: bool = True,
        *,
        template_files: list[AnyFITSPath] | None = None, 
        resample: bool | None = None,
        split: FloatVector | None = None,
        fwhm: FloatVector | None = None,
        bias: list[Literal['left', 'right']] | None = None,
        ratio: FloatVector | None = None,
        scale: float | None = None,
        raster: bool | None = None,
        allow_interp_fitting: bool | None = None,
        flux_bounds: AstropyBounds | None = None,
        fwhm_bounds: AstropyBounds | None = None,
        fine_tune: bool | None = None,
        fitter: FitterInstance | None = None,
    ) -> bool:
        """
        ** PYDANTIC VALIDATED METHOD **
        """  
        self.loadTemplates.__wrapped__.raw(
            self,
            template_files = template_files,
            resample = resample,
            split = split,
            fwhm = fwhm,
            bias = bias,
            ratio = ratio,
            scale = scale,
            allow_interp_fitting = allow_interp_fitting,
            flux_bounds = flux_bounds,
            fwhm_bounds = fwhm_bounds,
        )
        if raster:
            self.getRasterFit.__wrapped__.raw(
                self,
                covered = True,
                without_rejections = without_rejections,
                without_absorption = without_absorption,
            )
        if fine_tune: 
            self.performFineTuning.__wrapped__.raw(self, fitter=fitter)

        return True
    
    @validate_call(validate_return=False)
    @apply_info_to_method('iron')
    def loadTemplates(
        self,
        *,
        template_files: list[AnyFITSPath] | None = None, 
        resample: bool | None = None,
        split: FloatVector | None = None,
        fwhm: FloatVector | None = None,
        bias: list[Literal['left', 'right']] | None = None,
        ratio: FloatVector | None = None,
        scale: float | None = None,
        flux_bounds: AstropyBounds | None = None,
        fwhm_bounds: AstropyBounds | None = None,
        allow_interp_fitting: bool | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **

        Loads all templates:
        >   which are specified by the user.
        >   which are covered by the iron windows.

        Should adapt the templates to the current spectrum covered by the 
        respective templates. 
        """        
        if template_files is None:
            f = self.info.iron.or_default(locals())
            template_files = list(map(Path, f('template_files')))
            resample = f('resample')
            split = f('split')
            fwhm = f('fwhm')
            bias = f('bias')
            ratio = f('ratio')
            scale = f('scale')
            allow_interp_fitting = f('allow_interp_fitting')
        else:
            n = len(template_files)
            if split is None: 
                split = repeat(-1, n)
            if bias is None:
                bias  = repeat('right', n)
            if ratio is None:
                ratio = repeat(1.0, n)

            resample = False if fwhm is None else (resample or False)

        self.templates.clear()
        self.template_models.clear()
        for template_file, s, b, r in zip(
            template_files, split, bias, ratio,
        ):
            try:
                template: IronTemplate = IronTemplate.load(
                    template_file,
                    info = self.info,
                ).copy(with_matrices=False)
            except FileNotFoundError:
                msg = f"Could not find template file '{template_file}' -> \
                    Skipping."
                logger.warning(msg)
                continue

            # Check template-coverage
            tx = template.x[binary_fill_holes(template.data[0] > 0)]

            add_template: bool = False
            for iwindow in self:
                lb, ub = iwindow.x_bounds
                
                if ub < tx[0]:
                    continue
                if tx[-1] < lb:
                    break

                add_template = True
                break
            
            if not add_template:
                if (tx[-1] < self[0].x_bounds[0]): 
                    msg = f"IronTemplate @ {template_file} is entirely \
                        bluewards of the spectrum! Skipping."
                elif (self[-1].x_bounds[1] < tx[0]): 
                    msg = f"IronTemplate @ {template_file} is entirely \
                        redwards of the spectrum! Skipping."
                else: 
                    msg = f"IronTemplate @ {template_file} does not cover any \
                        of the iron windows! Skipping."

                logger.debug(msg)
                continue

            # Transform template if necessary
            if template.is_logspace:
                msg = f"IronTemplate @ {template_file} is already in logspace! \
                    Using template as is."
                logger.debug(msg)

            if not template.is_logspace:
                msg = f"IronTemplate @ {template_file} is not in logspace! \
                    Creating logspace-equivalent version."
                logger.debug(msg)

                msg = f"IronTemplate @ {template_file} will be transformed to \
                    logspace, which may be inefficient for pipelines. Consider \
                    caching a logspace-equivalent of this IronTemplate."
                logger.warning(msg)

                tx_wide = template.x[binary_fill_holes(template.data[-1] > 0)]
                mask = self._valid_pixels \
                    & (tx_wide[0] <= self._x) & (self._x <= tx_wide[-1])
                
                template.createLogspace(self._x[mask], inplace=True, keep_x=True)

            # Resample template if necessary
            if resample:
                template.resample(fwhm, inplace=True)

            if template.x[0] < s and s < template.x[-1]:
                msg = f"Applying split to IronTemplate @ {template_file}: \
                    split={s:.1f}, bias={b}, ratio={r:.2f}."
                logger.debug(msg)

                template.applySplit(
                    split = s,
                    left =  1.0 if b == 'left' else r,
                    right = r   if b == 'left' else 1.0,
                    scale = scale,
                    inplace = True,
                )

            self.templates[template.name] = template

            model: IronModel = IronModel(
                template, 
                apply_bounds.__wrapped__(1.0, flux_bounds),
                apply_bounds.__wrapped__(template.fwhm[0], fwhm_bounds),
                self.info.loading['sigma_res'],
                split=s,
                left=1.0,
                right=1.0,
                scale=scale,
                allow_interp_fitting=allow_interp_fitting, 
                name=template.name
            )
            model.flux.bounds = flux_bounds
            model.fwhm.bounds = fwhm_bounds

            self.template_models[model.name] = model

        return self

    @validate_call(validate_return=False)
    @apply_info_to_method('iron')
    def getRasterFit(
        self,
        *,
        covered: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = True,
        bg_flux: BGFlux | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **

        Fits the available templates using rasterisation.
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        coords = self.getMaskedCoords.__wrapped__(
            self,
            covered = covered,
            valid = True,
            bg_flux = bg_flux,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
        )
        for model in self.template_models.values():
            model.rasterFit(*coords, inplace=True)

        self.updateIronEmission.__wrapped__(
            self, self.getModel(),
        )
        return self

    def getModel(self) -> Union[IronModel, CompoundModel_[IronModel], None]:
        """
        Combines the available templates into a single (Split) TemplateModel or
        AstroPy compound model.
        """
        submodels = list(self.template_models.values())
        return sum(submodels[1:], start=submodels[0]) if submodels else None
    
    @validate_call(validate_return=False)
    def applyFit(
        self,
        fit: Union[IronModel, CompoundModel_[IronModel]],
        fit_info: FitInfo,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self.fit_info = fit_info
        for submodel in ((fit,) if fit.n_submodels == 1 else fit):
            self.template_models[submodel.name] = submodel
        return self

    @validate_call(validate_return=False)
    @apply_info_to_method('nonlinear')
    def performFineTuning(
        self,
        covered: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        bg_flux: BGFlux | None = None,
        *,
        fitter: FitterInstance | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **

        Fits the available templates using a nonlinear optimiser. 
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        x, y, dy = self.getMaskedCoords.__wrapped__(
            self,
            covered = covered,
            valid = True,
            bg_flux = bg_flux,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
        )

        model = self.getModel()
        if model is None:
            msg = "Tried to fine-tune IronModels, but none are available!"
            logger.critical(msg)
            return self
        
        # Calculate interpolation matrices for all templates
        for submodel in (model if model.n_submodels > 1 else [model]):
            _ = submodel._calculate_interpolation_matrix(x)

        fit, fit_info = fitter(model, x, y, dy, inplace=False)
        self.applyFit.__wrapped__(self, fit, fit_info)
        self.updateIronEmission.__wrapped__(self, fit)
        
        return self