__all__ = ['LineWindows']

from logging import getLogger
from typing import Self, ClassVar
from numpy import zeros_like
from pathlib import Path
from itertools import product
from scipy.optimize import OptimizeResult
from dataclasses import dataclass, field


from .lwindow import LWindow
from .graph_utils import Graph
from ..utils.general import stopwatch
from ..utils import SpecList, get_log

from quasar_typing.numpy import FloatVector, BoolVector
from quasar_typing.pathlib import AbsoluteFilePath
from quasar_typing.pandas import LineList
from quasar_typing.astropy import FitterInstance, CompoundModel_, FitInfo
from quasar_typing.misc import BackgroundFlux

from quasar_utils.decorators import validate_call, validated_apply_info_to_method
from quasar_utils.pipeline.linelist import read_linelist

from quasar_models.line import GaussianModel
from quasar_models.utils.astropy import get_free_params

from quasar_errors.model_samples import GaussianSampleList

logger = getLogger(__name__)

@dataclass(init=False)
class LineWindows(SpecList[LWindow]):
    graph: Graph | None = field(default=None, init=False)
    
    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({'all', 'em'})

    @property
    def sample(self) -> GaussianSampleList | None:
        if (model := self.getModel()) is None:
            return None
        return GaussianSampleList.fromGaussianModels(model)
    
    @validated_apply_info_to_method(subjects=('lines',), specific_kwargs={'v_sep'})
    def getMask(
        self,
        *,
        covered: bool = True,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        valid: bool = False,
        log_valid: bool = False,
        bg_flux: BackgroundFlux | None = None,

        line: float | None = None,
        limited: bool = True,
        v_sep: float | None = None,
    ) -> BoolVector:
        """
        ** PYDANTIC VALIDATED METHOD **
        """        
        if line is None:
            mask = zeros_like(self._x, dtype=bool)
            for lwindow in self:
                mask |= lwindow.getMask.__wrapped__(
                    lwindow,
                    covered=covered,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                    with_neighbours=with_neighbours,
                    valid=valid,
                    log_valid=log_valid,
                )
        else:
            for lwindow in filter(lambda window: line in window.lines, self):
                mask = lwindow.getMask.__wrapped__(
                    lwindow,
                    covered=covered,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                    with_neighbours=with_neighbours,
                    valid=valid,
                    log_valid=log_valid,
                    line=line,
                    limited=limited,
                    v_sep=v_sep,
                )
                break

        return mask
    
    @validated_apply_info_to_method(subjects=('loading', 'lines', 'nonlinear'))
    def __call__(
        self,
        linelist: AbsoluteFilePath | LineList,
        *,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        limited: bool = False,

        sigma_res: float | None = None,

        v_sep: float | None = None,
        forced_splits: FloatVector | None = None,
        w: int | None = None,
        min_fittable_total: int | None = None,
        min_fittable_ratio: float | None = None,
        evaluate_initial: float | int | None = None,
        aggressive: bool | None = None,
        crop: bool | None = None,
        measure: str | None = None,
        reverse: bool | None = None,
        make_copies: bool | None = None,

        fitter: FitterInstance | None = None,
    ) -> bool:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if bg_flux is None:
            bg_flux = self.default_bg
        
        _n: int = 3 + len(self)

        logger.debug(f"Starting pipeline for {self.__str__(True)}")
        self.updateLinesEmission()

        with stopwatch() as watch:
            logger.debug(f">>> [1/{_n}] Applying line list.")
            success = self.applyLineList.__wrapped__(
                self,
                linelist,
                sigma_res = sigma_res,
                v_sep = v_sep,
                forced_splits = forced_splits,
                min_fittable_total = min_fittable_total,
                min_fittable_ratio = min_fittable_ratio,
            )
            if not success: 
                logger.warning("Failed pipeline during 'applyLineList'!")
                return False
            
            logger.debug(f">>> [2/{_n}] Instantiating models.")
            for lwindow in self:
                success = lwindow.instantiateModels.__wrapped__(
                    lwindow,
                    bg_flux = bg_flux,
                    without_rejections = without_rejections,
                    without_absorption = without_absorption,
                    with_neighbours = with_neighbours,
                )
                if not success:
                    logger.warning(
                        ">>> Failed pipeline during 'instantiateModels' on {}!" \
                        .format(lwindow.__str__(True).removesuffix('.'))
                    )
                    return False

            logger.debug(f">>> [3/{_n}] Identifying appropriate fitting sequence.")
            fitting_sequence = self.getFittingSequence()

            apply_vel_copies: bool = make_copies
            if apply_vel_copies:
                logger.debug(
                    ">>> Circular graph -> no velocity profiles copied!" \
                    if self.graph.is_circular else \
                    ">>> Graph is valid!"
                )
                apply_vel_copies ^= self.graph.is_circular

            logger.debug(f">>> Fitting sequence: {fitting_sequence}")
            for count, idx in enumerate(fitting_sequence, start=4):
                logger.debug(f">>> [{count}/{_n}] Fitting 'LWindow' no. {idx}.")

                lwindow = self[idx]
                success = lwindow.makeInitialFit.__wrapped__(
                    lwindow,
                    bg_flux=bg_flux,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                    with_neighbours=with_neighbours,
                    evaluate_initial=evaluate_initial,
                    fitter=fitter,
                )
                if not success:
                    logger.warning(
                        ">>> Failed pipeline during 'makeInitialFit' on {}!" \
                        .format(lwindow.__str__(True).removesuffix('.'))
                    )
                    return False

                success = lwindow.makeFinalFit.__wrapped__(
                    lwindow,
                    bg_flux=bg_flux,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                    with_neighbours=with_neighbours,
                    limited=limited,
                    w=w,
                    aggressive=aggressive,
                    crop=crop,
                    measure=measure,
                    reverse=reverse,
                    evaluate_initial=evaluate_initial,
                    v_sep=v_sep,
                    fitter=fitter,
                )
                if not success:
                    logger.warning(
                        ">>> Failed pipeline during 'makeFinalFit' on {}!" \
                        .format(lwindow.__str__(True).removesuffix('.'))
                    )
                    return False
                
                if apply_vel_copies:
                    lwindow.applyMyself()
            
        logger.debug(
            "Finished entire pipeline in {:.1f} ms." \
            .format(1e3 * watch.elapsed)
        )
        return True
    
    @validated_apply_info_to_method(subjects=('lines',), specific_kwargs={'v_sep'})
    def getMaskedCoords(
        self, 
        *,
        covered: bool = False,
        log: bool = False,
        without_rejections: bool = False, 
        without_absorption: bool = False,
        with_neighbours: bool = False,
        valid: bool = False,
        log_valid: bool = False,
        bg_flux: BackgroundFlux | None = None,

        line: float | None = None,
        limited: bool = True,
        v_sep: float | None = None,
    ) -> tuple[FloatVector, FloatVector, FloatVector, FloatVector]:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        x = self._x_log if log else self._x
        dy = self._dy_log if log else self._dy

        y = self._y.copy()
        y_smooth = self._y_smooth.copy()
        if bg_flux is not None:
            for bg in bg_flux:
                y        -= getattr(self, f"_y_{bg}")
                y_smooth -= getattr(self, f"_y_{bg}")

        if log: 
            y        = get_log(y, self.y0, self._log_valid_pixels)
            y_smooth = get_log(y_smooth, self.y0, self._log_valid_pixels)

        mask = self.getMask.__wrapped__(
            self,
            covered = covered,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
            with_neighbours = with_neighbours,
            valid = valid,
            log_valid = log_valid,
            line = line,
            limited = limited,
            v_sep = v_sep
        )
        return x[mask], y[mask], dy[mask], y_smooth[mask]
    
    @validated_apply_info_to_method(subjects=('loading', 'lines'))
    def applyLineList(
        self,
        linelist: AbsoluteFilePath | LineList,
        *,
        sigma_res: float | None = None,
        v_sep: float | None = None,
        forced_splits: FloatVector | None = None,
        min_fittable_total: int | None = None,
        min_fittable_ratio: float | None = None,
    ) -> bool:
        """
        ** PYDANTIC VALIDATED METHOD **
        !!! Ensuring Lyman-alpha?
        """
        kwargs = {'x_bounds': self.x_bounds}
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
        
        if isinstance(linelist, Path):
            linelist = read_linelist.__wrapped__(path=linelist, info=self.info)

        df = linelist.sort_values('line', inplace=False)
        if self.x_bounds[0] is not None:
            df.drop(df[df['line'] < self.x_bounds[0]].index, inplace=True)
        if self.x_bounds[1] is not None:
            df.drop(df[df['line'] >= self.x_bounds[1]].index, inplace=True)
        df = df.reset_index(drop=True)
                
        prev_line: float = None
        for idx, row in df.iterrows():
            line = row['line']

            cond = (idx == 0)
            if not cond:
                _line = prev_line

                cond = \
                    ((_line < forced_splits) & (forced_splits < line)).any() \
                    or \
                    (line - _line) > (_line + line) * v_sep

            if cond:
                self.append(LWindow(coords_or_spectrum, **kwargs))

            _ = self[-1].add.__wrapped__(
                self[-1],
                row['name'],
                row['line'],
                row['n_max'],

                needs_line      = row['needs_line'],
                is_copy_of      = row['is_copy_of'],

                strength_bounds = (row['strength_lower'], row['strength_upper']),
                v_off_bounds    = (row['v_off_lower'], row['v_off_upper']),
                sigma_v_bounds  = (row['sigma_v_lower'], row['sigma_v_upper']),
                
                scale_init      = row['scale_init'],
                scale_bounds    = (row['scale_lower'], row['scale_upper']),
            )
            
            prev_line = line

        # Check for line dependencies!
        self.checkLineDependencies.__wrapped__(self, linelist)

        for lwindow in self:
            lwindow.prepareLines.__wrapped__(
                lwindow,
                v_sep=v_sep,
                min_fittable_total=min_fittable_total,
                min_fittable_ratio=min_fittable_ratio,
            )
            lwindow.prepareNeighbours.__wrapped__(
                lwindow,
                sigma_res=sigma_res,
            )

        return True
    
    @validate_call
    def checkLineDependencies(
        self, 
        linelist: AbsoluteFilePath | LineList,
    ) -> bool:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if isinstance(linelist, Path):
            linelist = read_linelist.__wrapped__(path=linelist, info=self.info)

        all_added_lines: set[str] = set()
        for lwindow in self:
            all_added_lines.update(lwindow.lines.keys())

        repeat_call: bool = False
        for lwindow in self:
            for needed_line in lwindow.needs_line.values():
                if needed_line in lwindow.names:
                    logger.warning(
                        f"Needed line '{needed_line}' is already covered by "
                        "the same 'LWindow'?"
                    )
                    continue

                if needed_line in all_added_lines:
                    # Line is added by another 'LWindow' class?!
                    logger.warning(
                        f"Needed line '{needed_line}' is covered by another "
                        "'LWindow'?"
                    )
                    return False
                
                ser = linelist[linelist['name'] == needed_line]
                if len(ser) == 0:
                    logger.warning(
                        f"Needed line '{needed_line}' could not be found in "
                        "line list -> skipping!"
                    )
                    continue

                if len(ser) > 1:
                    logger.warning(
                        f"Found duplicates for '{needed_line}' in line list "
                        "-> skipping!"
                    )
                    continue

                row = next(ser.iterrows())[1]
                def get_bounds(s: str) -> tuple[float, float]:
                    return (row[f'{s}_lower'], row[f'{s}_upper'])

                lwindow.add.__wrapped__(
                    lwindow,
                    row['name'],
                    row['line'],
                    row['n_max'],

                    needs_line=row['needs_line'],
                    strength_bounds=get_bounds('strength'),
                    v_off_bounds=get_bounds('v_off'),
                    sigma_v_bounds=get_bounds('sigma_v'),
                    is_copy_of=row['is_copy_of'],
                    scale_init=row['scale_init'],
                    scale_bounds=get_bounds('scale'),
                    force_add=True,
                )
                all_added_lines.add(needed_line)

                repeat_call |= bool(row['needs_line'])

        if repeat_call:
            return self.checkLineDependencies.__wrapped__(self, linelist)
        return True
    
    def checkVelocityProfileCopies(self) -> bool:
        """
        This method is supposed to do the following:

        Loop through the created 'LWindow' classes:
        1.  If a 'LWindow' has a non-empty 'is_copy_of' dictionary, finds the 
            'LWindow' class with the corresponding model to copy from, i.e.
            the 'LWindow' whose 'names' set contains the value of the 
            'is_copy_of' item. 

            ! FOR NOW the model to copy from must not be covered by the same
            ! 'LWindow' class. 
            ! We can therefore skip everything if: len(self) < 2

        2.  Once the other 'LWindow' class has been found, update its 
            'copies_to' dictionary. Update the 'graph_edges' dictionary to
            represent this new connection.
        """
        self.graph: Graph = Graph(len(self))

        if len(self) < 2:
            return False

        for (idx_o, orig), (idx_d, dest) in product(
            enumerate(self),
            filter(lambda tup: bool(tup[1].is_copy_of), enumerate(self)),
        ):
            if idx_o == idx_d:
                continue

            for mimic, master in dest.is_copy_of.items():
                if master not in orig.names:
                    continue

                self.graph[idx_o].add(idx_d)
                orig.copies_to[master].append((idx_d, mimic))

        return True
    
    def getFittingSequence(self) -> list[int]:
        if self.graph is None:
            success = self.checkVelocityProfileCopies()
            if not success:
                return list(range(len(self)))
        return self.graph.expand(inplace=True).createChain()

    def getModel(self) -> GaussianModel | CompoundModel_[GaussianModel] | None:
        """
        Retrieves and combines each 'LWindow's current fit/model, combining them
        into a single model. 

        NOTES
        -----
        The keyword argument 'thaw' is set to True when retrieving models, 
        meaning that copies of models are retrieved, and that any models copying 
        other models' velocity profiles will have their parameters unfrozen and
        their 'tie' attributes enabled. 
        """
        models = [
            mod 
            for lwindow in self
            if (mod := lwindow.getModel(thaw=True)) is not None
        ]
        return sum(models[1:], start=models[0]) if models else None
    
    @validate_call
    def applyFit(
        self,
        fit: GaussianModel | CompoundModel_[GaussianModel],
        fit_info: FitInfo,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **

        Applies the 'fit' and 'fit_info' values to all 'LWindow' classes.

        Notes
        -----
        The keyword argument 'freeze' is set to True when applying models,
        meaning that copies of models are applied, and that any models copying
        other models' velocity profiles will have their parameters frozen and
        their 'tie' attributes disabled.
        """
        fs = fit if (fit.n_submodels > 1) else [fit]

        count = 0
        for lwindow in self:
            submodels = [f for f in fs if f.pure_name in lwindow.names]
            if len(submodels) == 0:
                continue

            model = sum(submodels[1:], start=submodels[0])
            n_free = sum(get_free_params.__wrapped__(model).values())

            sel = slice(count, count+n_free)
            finfo = OptimizeResult(
                message =    fit_info.message,
                success =    fit_info.success,
                status =     fit_info.status,
                fun =        fit_info.fun,
                x =          fit_info.x[sel],
                cost =       fit_info.cost,
                jac =        fit_info.jac[sel,sel],
                grad =       fit_info.grad[sel],
                optimality = fit_info.optimality,
                nfev =       fit_info.nfev,
                njev =       fit_info.njev,
                param_cov =  fit_info.param_cov[sel,sel]
            )
            lwindow.applyFit.__wrapped__(
                lwindow,
                model, 
                finfo,
                freeze=True,
            )
            count += n_free

        return self
    
    @validate_call
    def adoptFit(
        self,
        fit: GaussianModel | CompoundModel_[GaussianModel],
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        fs = fit if (fit.n_submodels > 1) else (fit,)
        for lwindow in self:
            submodels = [f for f in fs if f.pure_name in lwindow.names]
            if len(submodels) == 0:
                continue

            model = sum(submodels[1:], start=submodels[0])
            lwindow.adoptFit.__wrapped__(
                lwindow,
                model,
            )
        return self