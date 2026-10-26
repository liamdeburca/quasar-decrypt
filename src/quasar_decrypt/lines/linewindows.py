__all__ = ["LineWindows"]

from dataclasses import field
from itertools import product
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, Self

import pydantic
from numpy import bool_, zeros_like
from quasar_typing.numpy import BoolVector, FloatVector
from quasar_typing.pathlib import FilePath
from quasar_typing.scipy import OptimizeResultType
from quasar_typing.string_selections import BackgroundFlux
from quasar_utils.decorators import validated_apply_info_to_method
from quasar_utils.linelist import LineList
from quasar_utils.setup import FitterKwargs
from scipy.optimize import OptimizeResult

from quasar_decrypt.utils import MaskedCoords, _SpecWindowList

from .graph_utils import Graph
from .lwindow import LineModel, LWindow

logger = getLogger(__name__)


@pydantic.dataclasses.dataclass
class LineWindows(_SpecWindowList[LWindow]):
    graph: Graph | None = field(default=None, init=False)

    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({"all", "em"})

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["graph"] = self.graph
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.graph = state.get("graph")

    @property
    def sample(self):
        from quasar_errors.model_samples.line import GaussianSampleList
        if (model := self.getModel()) is None:
            return None
        return GaussianSampleList.fromGaussianModels(model)

    @validated_apply_info_to_method(
        subjects=("lines",), specific_kwargs={"v_sep"}
    )
    def getMask(
        self,
        *,
        original: bool = False,
        covered: bool = True,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        valid: bool = False,
        log_valid: bool = False,
        line: float | None = None,
        limited: bool = True,
        v_sep: float | None = None,
        as_tuple: bool = False,
    ) -> BoolVector | tuple[BoolVector, BoolVector]:
        original = self._update_original_flag(
            original=original,
            log=False,
        )
        suffix = "_original" if original else ""

        if line is None:
            x = getattr(self, "_x" + suffix)
            primary_mask = zeros_like(x, dtype=bool_)
            secondary_mask = zeros_like(x, dtype=bool_)

            for lwindow in self:
                _p, _s = lwindow.getMask.__wrapped__(
                    lwindow,
                    original=original,
                    covered=covered,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                    with_neighbours=with_neighbours,
                    valid=valid,
                    log_valid=log_valid,
                    as_tuple=True,
                )
                primary_mask |= _p
                secondary_mask |= _s
        else:
            for lwindow in filter(
                lambda window: line in window.waves.values(), 
                self,
            ):
                primary_mask, secondary_mask = lwindow.getMask.__wrapped__(
                    lwindow,
                    original=original,
                    covered=covered,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                    with_neighbours=with_neighbours,
                    valid=valid,
                    log_valid=log_valid,
                    line=line,
                    limited=limited,
                    v_sep=v_sep,
                    as_tuple=True,
                )
                break

        if as_tuple:
            return primary_mask, secondary_mask
        return primary_mask & secondary_mask

    @validated_apply_info_to_method(subjects=("loading", "lines", "nonlinear"))
    def __call__(
        self,
        linelist: FilePath | LineList,
        *,
        template_model: Optional[LineModel] = None,
        original: bool = False,
        shrink: float | None = None,
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
        evaluate_initial: float | None = None,
        aggressive: bool | None = None,
        crop: bool | None = None,
        measure: str | None = None,
        reverse: bool | None = None,
        fitter_kwargs: FitterKwargs | None = None,
    ) -> bool:
        if bg_flux is None:
            bg_flux = self.default_bg

        logger.debug(f"Starting pipeline for {self.__str__(True)}")
        self.updateLinesEmission()

        ### 'applyLineList'
        msg = "Applying line list: "
        success = self.applyLineList(
            linelist,
            sigma_res=sigma_res,
            v_sep=v_sep,
            forced_splits=forced_splits,
            min_fittable_total=min_fittable_total,
            min_fittable_ratio=min_fittable_ratio,
        )
        if success:
            logger.debug(msg + "success!")
        if not success:
            logger.info(msg + "failed!")
            return False

        if template_model is not None:
            logger.debug("Applying template model.")
            self.applyFit(
                template_model,
                update_emission=True,
            )
            for lwindow in self:
                lwindow.fitModel(
                    update_flux=True,
                    bg_flux=bg_flux,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                    with_neighbours=with_neighbours,
                    fitter_kwargs=(fitter_kwargs or {}),
                )
        else:
            ### 'instantiateModels'
            msg = "Instantiating models: "
            for i, lwindow in enumerate(self):
                success = lwindow.instantiateModels(
                    bg_flux=bg_flux,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                    with_neighbours=with_neighbours,
                )
                if not success:
                    logger.info(msg + f"failed on LWindow no. {i}!")
                    return False
            logger.debug(msg + "success!")

            ### 'getFittingSequence'
            msg = "Getting fitting sequence: "
            fitting_sequence = self.getFittingSequence()
            logger.debug(msg + "success!")

            msg = "Applying velocity profile copies: "
            if self.graph.is_circular:
                logger.info(
                    msg + "Circular graph detected -> continuing."
                )
            else:
                logger.debug(msg + f"success: {fitting_sequence=}!")

            msg = "Fitting Lwindows: "
            for idx in fitting_sequence:
                lwindow = self[idx]

                success = lwindow.makeInitialFit(
                    bg_flux=bg_flux,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                    with_neighbours=with_neighbours,
                    evaluate_initial=evaluate_initial,
                    fitter_kwargs=fitter_kwargs,
                )
                if not success:
                    msg += "failed during 'makeInitialFit' on LWindow no. "\
                        f"{idx}!"
                    logger.info(msg)
                    return False

                success = lwindow.makeFinalFit(
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
                    fitter_kwargs=fitter_kwargs,
                )
                if not success:
                    msg += (
                        f"failed during 'makeFinalFit' on LWindow no. {idx}!"
                    )
                    logger.info(msg)
                    return False

                lwindow.applyMyself()

            self.makeCombinedFit(
                original=original,
                covered=True,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                with_neighbours=with_neighbours,
                bg_flux=bg_flux,
                shrink=shrink,
                fitter_kwargs=fitter_kwargs,
            )

            logger.debug(msg + "success!")

        return True

    @validated_apply_info_to_method(
        subjects=("nonlinear",),
        specific_kwargs={"fitter_kwargs"}
    )
    def makeCombinedFit(
        self,
        *,
        original: bool = False,
        covered: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        bg_flux: BackgroundFlux | None = None,
        shrink: bool | None = None,
        fitter_kwargs: FitterKwargs | None = None,
    ) -> bool:
        from quasar_models.modeling import Fitter, PrepareModel

        if bg_flux is None:
            bg_flux = self.default_bg

        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            original=original,
            mode="c",
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            with_neighbours=with_neighbours,
            valid=True,
            bg_flux=bg_flux,
            shrink=shrink,
        )
        model = self.getModel()

        preparer = PrepareModel.fromCoords(masked_coords, model)
        data = preparer.createDataDict(masked_coords)

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

        self.applyFit.__wrapped__(
            self,
            fit,
            fit_info=fitter.fit_info,
            update_emission=True,
        )
        return True

    @validated_apply_info_to_method(
        subjects=("lines",), specific_kwargs={"v_sep"}
    )
    def getMaskedCoords(
        self,
        *,
        original: bool = False,
        mode: Literal["c", "r"] | None = None,
        covered: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        valid: bool = False,
        log_valid: bool = False,
        bg_flux: BackgroundFlux | None = None,
        line: float | None = None,
        limited: bool = True,
        v_sep: float | None = None,
        shrink: bool | None = None,
    ) -> MaskedCoords:
        original = self._update_original_flag(
            original=original,
            log=False,
        )
        if bg_flux is None:
            bg_flux = self.default_bg

        primary_mask, secondary_mask = self.getMask.__wrapped__(
            self,
            original=original,
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            with_neighbours=with_neighbours,
            valid=valid,
            log_valid=log_valid,
            line=line,
            limited=limited,
            v_sep=v_sep,
            as_tuple=True,
        )
        return self._get_masked_coords_helper(
            original=original,
            mode=mode,
            primary_mask=primary_mask,
            secondary_mask=secondary_mask,
            log_valid=log_valid,
            bg_flux=bg_flux,
            shrink=shrink,
        )
    
    @validated_apply_info_to_method(subjects=("loading", "lines"))
    def applyLineList(
        self,
        linelist: FilePath | LineList,
        *,
        sigma_res: float | None = None,
        v_sep: float | None = None,
        forced_splits: FloatVector | None = None,
        min_fittable_total: int | None = None,
        min_fittable_ratio: float | None = None,
    ) -> bool:
        """
        !!! Ensuring Lyman-alpha?
        """
        if isinstance(linelist, Path):
            linelist = LineList.read_csv(
                path=linelist,
                info=self.info,
            )
        
        df = linelist.toDataFrame()
        
        if self.x_bounds[0] is not None:
            df = df[self.x_bounds[0] <= df["wave"]]

        if self.x_bounds[1] is not None:
            df = df[df["wave"] <= self.x_bounds[1]]

        prev_wave: float = None
        df = df.sort_values("wave", ignore_index=True)
        for idx, row in df.iterrows():
            wave = row["wave"]

            cond = (idx == 0)
            if not cond:
                _wave = prev_wave
                cond = ((_wave < forced_splits) & (forced_splits < wave)).any()
                cond |= (wave - _wave) > (_wave + wave) * v_sep

            if cond:
                lwindow = LWindow._create(
                    spectrum=self.spectrum,
                    x_bounds=self.x_bounds,
                )
                self.append(lwindow)

            _ = self[-1].add(
                name=row["name"],
                complex_name=row["complex"],
                wave=row["wave"],
                linetype=row["linetype"],
                n_max=row["n_max"],
                needs_line=row["needs_line"],
                is_copy_of=row["is_copy_of"],
                strength_bounds=(row["strength_lower"], row["strength_upper"]),
                v_off_bounds=(row["v_off_lower"], row["v_off_upper"]),
                fwhm_v_bounds=(row["fwhm_v_lower"], row["fwhm_v_upper"]),
                scale_init=row["scale_init"],
                scale_bounds=(row["scale_lower"], row["scale_upper"]),
                scale_fixed=row["scale_fixed"],
            )
            prev_wave = wave

        # Check for line dependencies!
        self.checkLineDependencies(linelist)
        self.checkVelocityProfileCopies()

        for lwindow in self:
            lwindow.prepareLines(
                v_sep=v_sep,
                min_fittable_total=min_fittable_total,
                min_fittable_ratio=min_fittable_ratio,
            )

        if self.spectrum is not None and self.spectrum.pl is not None:
            for lwindow in self:
                lwindow.prepareNeighbours(sigma_res=sigma_res)

        return True

    @pydantic.validate_call
    def checkLineDependencies(
        self, 
        linelist: FilePath | LineList,
    ) -> bool:
        """
        ...
        """
        if isinstance(linelist, Path):
            linelist = LineList.read_csv(
                path=linelist,
                info=self.info,
            )

        all_added_lines: set[str] = {
            name
            for lwindow in self
            for name in lwindow.names
        }

        repeat_call: bool = False
        for lwindow in self:
            for needed_line in lwindow.needs_line.values():
                if needed_line in lwindow.names:
                    logger.info(
                        f"Needed line '{needed_line}' is already covered by "
                        "the same 'LWindow'?"
                    )
                    continue

                if needed_line in all_added_lines:
                    # Line is added by another 'LWindow' class?!
                    logger.info(
                        f"Needed line '{needed_line}' is covered by another "
                        "'LWindow'?"
                    )
                    return False

                
                indices = [
                    i 
                    for i, name in enumerate(linelist.name) 
                    if name == needed_line
                ]
                if len(indices) == 0:
                    logger.info(
                        f"Needed line '{needed_line}' could not be found in "
                        "line list -> skipping!"
                    )
                    continue

                if len(indices) > 1:
                    logger.info(
                        f"Found duplicates for '{needed_line}' in line list "
                        "-> skipping!"
                    )
                    continue

                row = linelist[indices[0]]
                def get_bounds(s: str, row=row) -> tuple[float, float]:
                    return (row[f"{s}_lower"], row[f"{s}_upper"])

                lwindow.add.__wrapped__(
                    lwindow,
                    name=row["name"],
                    complex_name=row["complex"],
                    wave=row["wave"],
                    linetype=row["linetype"],
                    n_max=row["n_max"],
                    needs_line=row["needs_line"],
                    strength_bounds=get_bounds("strength"),
                    v_off_bounds=get_bounds("v_off"),
                    fwhm_v_bounds=get_bounds("fwhm_v"),
                    is_copy_of=row["is_copy_of"],
                    scale_init=row["scale_init"],
                    scale_bounds=get_bounds("scale"),
                    force_add=True,
                )
                all_added_lines.add(needed_line)

                repeat_call |= bool(row["needs_line"])

        if repeat_call:
            return self.checkLineDependencies(linelist)
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
            ! We can therefore skip everything if: len(self) <= 1

        2.  Once the other 'LWindow' class has been found, update its
            'copies_to' dictionary. Update the 'graph_edges' dictionary to
            represent this new connection.
        """
        self.graph: Graph = Graph(len(self))

        if len(self) <= 1:
            return False

        for (idx_o, orig), (idx_d, dest) in product(
            enumerate(self),
            filter(lambda tup: tup[1].is_copy_of, enumerate(self)),
        ):
            for mimic, master in filter(
                lambda item: item[1] in orig.names,
                dest.is_copy_of.items(),
            ):
                if idx_o == idx_d:
                    continue
                # Checks for velocity profile copies
                self.graph[idx_o].add(idx_d)
                orig.copies_to[master].append((idx_d, mimic))

        return True

    def getFittingSequence(self) -> list[int]:
        if self.graph is None:
            success = self.checkVelocityProfileCopies()
            if not success:
                return list(range(len(self)))
        return self.graph.expand(inplace=True).createChain()

    def getModel(
        self,
    ) -> Optional[LineModel]:
        """
        Retrieves and combines each 'LWindow's current fit/model, combining them
        into a single model.
        """
        models = []
        for lwindow in self:
            model = lwindow.getModel()
            if model is not None:
                models.append(model)

        return sum(models[1:], start=models[0]) if models else None

    @pydantic.validate_call
    def adoptFit(
        self,
        fit: LineModel,
        *,
        fit_info: OptimizeResultType | None = None,
        update_emission: bool = False,
    ) -> Self:
        from quasar_models.utils.astropy import _get_free_params
        submodels = (fit,) if (fit.n_submodels == 1) else fit

        count: int = 0
        for lwindow in self:
            fs = list(
                filter(lambda f: f.pure_name in lwindow.names, submodels)
            )
            if not fs:
                continue

            model = sum(fs[1:], start=fs[0])
            if fit_info is None:
                finfo = None
            else:
                n_free = sum(_get_free_params(model).values())
                sel = slice(count, count + n_free)
                finfo = OptimizeResult(
                    x=fit_info.x[sel],
                    success=fit_info.success,
                    message=fit_info.message,
                    status=fit_info.status,
                    fun=fit_info.fun,
                    jac=fit_info.jac[sel, sel],
                    nfev=fit_info.nfev,
                    njev=fit_info.njev,
                    nit=fit_info.get("nit", 0),
                    maxcv=fit_info.get("maxcv", 0),
                )
                count += n_free

            lwindow.adoptFit.__wrapped__(
                lwindow,
                model,
                fit_info=finfo,
                update_emission=update_emission,
            )

        return self

    @pydantic.validate_call
    def applyFit(
        self,
        fit: LineModel,
        *,
        fit_info: OptimizeResultType | None = None,
        update_emission: bool = False,
    ) -> Self:
        """
        Applies the 'fit' and 'fit_info' values to all 'LWindow' classes.
        """
        from quasar_models.utils.astropy import _get_free_params

        submodels = fit if (fit.n_submodels > 1) else [fit]

        count: int = 0
        for lwindow in self:
            fs = list(
                filter(lambda f: f.pure_name in lwindow.names, submodels)
            )
            if not fs:
                continue

            model = sum(fs[1:], start=fs[0])
            if fit_info is None:
                finfo = None
            else:
                n_free = sum(_get_free_params(model).values())
                sel = slice(count, count + n_free)
                finfo = OptimizeResult(
                    x=fit_info.x[sel],
                    success=fit_info.success,
                    message=fit_info.message,
                    status=fit_info.status,
                    fun=fit_info.fun,
                    jac=fit_info.jac[sel, sel],
                    nfev=fit_info.nfev,
                    njev=fit_info.njev,
                    nit=fit_info.get("nit", 0),
                    maxcv=fit_info.get("maxcv", 0),
                )
                count += n_free

            lwindow.applyFit.__wrapped__(
                lwindow,
                model,
                fit_info=finfo,
                update_emission=update_emission,
            )

        return self
