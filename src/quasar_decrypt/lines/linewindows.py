__all__ = ["LineWindows"]

from dataclasses import dataclass, field
from itertools import product
from logging import getLogger
from pathlib import Path
from typing import ClassVar, Literal, Optional, Self, Union

from numpy import zeros_like
from quasar_errors.model_samples import GaussianSampleList
from quasar_models.line import GaussianModel, _VProfileCopy
from quasar_models.utils.astropy import get_free_params
from quasar_typing.astropy import CompoundModel_, FitInfo
from quasar_typing.misc import BackgroundFlux
from quasar_typing.numpy import BoolVector, FloatVector
from quasar_typing.pathlib import AbsoluteFilePath
from quasar_utils.decorators import (
    validate_call,
    validated_apply_info_to_method,
)
from quasar_utils.pipeline import LineList
from quasar_utils.setup import FitterKwargs
from scipy.optimize import OptimizeResult

from ..utils import (
    ContiguousMaskedCoords,
    MaskedCoords,
    ReadOnlyMaskedCoords,
    _SpecWindowList,
)
from .graph_utils import Graph
from .lwindow import LWindow

logger = getLogger(__name__)


@dataclass(init=False)
class LineWindows(_SpecWindowList[LWindow]):
    graph: Graph | None = field(default=None, init=False)

    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({"all", "em"})

    @property
    def sample(self) -> GaussianSampleList | None:
        if (model := self.getModel()) is None:
            return None
        return GaussianSampleList.fromGaussianModels(model)

    @validated_apply_info_to_method(
        subjects=("lines",), specific_kwargs={"v_sep"}
    )
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

    @validated_apply_info_to_method(subjects=("loading", "lines", "nonlinear"))
    def __call__(
        self,
        linelist: AbsoluteFilePath | LineList,
        *,
        template_model: Optional[
            Union[GaussianModel, CompoundModel_[GaussianModel]]
        ] = None,
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
        make_copies: bool | None = None,
        fitter_kwargs: FitterKwargs | None = None,
    ) -> bool:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        logger.debug(f"Starting pipeline for {self.__str__(True)}")
        self.updateLinesEmission()

        ### 'applyLineList'
        msg = "Applying line list: "
        success = self.applyLineList.__wrapped__(
            self,
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
            logger.warning(msg + "failed!")
            return False

        if template_model is not None:
            logger.debug("Applying template model.")
            self.applyFit.__wrapped__(
                self,
                template_model,
                update_emission=True,
            )
            for lwindow in self:
                lwindow.fitModel.__wrapped__(
                    lwindow,
                    update_flux=True,
                    bg_flux=bg_flux,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                    with_neighbours=with_neighbours,
                    fitter_kwargs=fitter_kwargs,
                )
        else:
            ### 'instantiateModels'
            msg = "Instantiating models: "
            for i, lwindow in enumerate(self):
                success = lwindow.instantiateModels.__wrapped__(
                    lwindow,
                    bg_flux=bg_flux,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                    with_neighbours=with_neighbours,
                )
                if not success:
                    logger.warning(msg + f"failed on LWindow no. {i}!")
                    return False
            logger.debug(msg + "success!")

            ### 'getFittingSequence'
            msg = "Getting fitting sequence: "
            fitting_sequence = self.getFittingSequence()
            try:
                logger.debug(msg + "success!")
            except Exception as _:
                logger.warning(msg + "failed!")
                return False

            if make_copies:
                msg = "Applying velocity profile copies: "
                if self.graph.is_circular:
                    logger.warning(
                        msg + "Circular graph detected -> continuing."
                    )
                    make_copies = False
                else:
                    logger.debug(msg + f"success: {fitting_sequence=}!")

            msg = "Fitting Lwindows: "
            for idx in fitting_sequence:
                lwindow = self[idx]

                success = lwindow.makeInitialFit.__wrapped__(
                    lwindow,
                    bg_flux=bg_flux,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                    with_neighbours=with_neighbours,
                    evaluate_initial=evaluate_initial,
                    fitter_kwargs=fitter_kwargs,
                )
                if not success:
                    msg += (
                        f"failed during 'makeInitialFit' on LWindow no. {idx}!"
                    )
                    logger.warning(msg)
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
                    fitter_kwargs=fitter_kwargs,
                )
                if not success:
                    msg += (
                        f"failed during 'makeFinalFit' on LWindow no. {idx}!"
                    )
                    logger.warning(msg)
                    return False

                if make_copies:
                    lwindow.applyMyself.__wrapped__(lwindow)
            logger.debug(msg + "success!")

        return True

    @validated_apply_info_to_method(
        subjects=("lines",), specific_kwargs={"v_sep"}
    )
    def getMaskedCoords(
        self,
        *,
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
    ) -> MaskedCoords:
        if bg_flux is None:
            bg_flux = self.default_bg

        mask = self.getMask.__wrapped__(
            self,
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
        match mode:
            case "c":
                return ContiguousMaskedCoords(self, mask, bg_flux=bg_flux)
            case "r":
                return ReadOnlyMaskedCoords(self, mask, bg_flux=bg_flux)
            case None:
                return MaskedCoords(self, mask, bg_flux=bg_flux)

    @validated_apply_info_to_method(subjects=("loading", "lines"))
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
        kwargs = {"x_bounds": self.x_bounds}
        if self.spectrum is None:
            kwargs["x"] = self._x
            kwargs["y"] = self._y
            kwargs["dy"] = self._dy
            kwargs["dx"] = self._dx

            kwargs["y_smooth"] = self._y_smooth
            kwargs["y_pl"] = self._y_pl
            kwargs["y_fe"] = self._y_fe
            kwargs["y_ba"] = self._y_ba
            kwargs["y_hg"] = self._y_hg
            kwargs["y_em"] = self._y_em

            kwargs["rejected_pixels"] = self._rejected_pixels
            kwargs["absorbed_pixels"] = self._absorbed_pixels
            kwargs["valid_pixels"] = self._valid_pixels
            kwargs["log_valid_pixels"] = self._log_valid_pixels
            kwargs["p_absorbed"] = self._p_absorbed

            kwargs["x0"] = self.x0
            kwargs["y0"] = self.y0
            kwargs["x_log"] = self._x_log
            kwargs["y_log"] = self._y_log
            kwargs["dy_log"] = self._dy_log

            kwargs["info"] = self.info
            kwargs["get_mask"] = self.get_mask
        else:
            kwargs["spectrum"] = self.spectrum

        if isinstance(linelist, Path):
            linelist = LineList.read_csv.__wrapped__(
                LineList,
                path=linelist,
                info=self.info,
            )

        df = linelist.sort_values("line", inplace=False)
        if self.x_bounds[0] is not None:
            df.drop(df[df["line"] < self.x_bounds[0]].index, inplace=True)
        if self.x_bounds[1] is not None:
            df.drop(df[df["line"] >= self.x_bounds[1]].index, inplace=True)
        df = df.reset_index(drop=True)

        prev_line: float = None
        for idx, row in df.iterrows():
            line = row["line"]

            cond = idx == 0
            if not cond:
                _line = prev_line

                cond = (
                    (_line < forced_splits) & (forced_splits < line)
                ).any() or (line - _line) > (_line + line) * v_sep

            if cond:
                lwindow = LWindow.create.__wrapped__(
                    LWindow,
                    spectrum=self.spectrum,
                    x_bounds=self.x_bounds,
                    get_mask=None,
                )
                self.append(lwindow)

            _ = self[-1].add.__wrapped__(
                self[-1],
                row["name"],
                row["complex"],
                row["line"],
                row["n_max"],
                needs_line=row["needs_line"],
                is_copy_of=row["is_copy_of"],
                strength_bounds=(row["strength_lower"], row["strength_upper"]),
                v_off_bounds=(row["v_off_lower"], row["v_off_upper"]),
                fwhm_v_bounds=(row["fwhm_v_lower"], row["fwhm_v_upper"]),
                scale_init=row["scale_init"],
                scale_bounds=(row["scale_lower"], row["scale_upper"]),
                scale_fixed=row["scale_fixed"],
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
            if self.spectrum is not None and self.spectrum.pl is not None:
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
        if isinstance(linelist, Path):
            linelist = LineList.read_csv.__wrapped__(
                LineList,
                path=linelist,
                info=self.info,
            )

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

                ser = linelist[linelist["name"] == needed_line]
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

                def get_bounds(s: str, row=row) -> tuple[float, float]:
                    return (row[f"{s}_lower"], row[f"{s}_upper"])

                lwindow.add.__wrapped__(
                    lwindow,
                    row["name"],
                    row["complex"],
                    row["line"],
                    row["n_max"],
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
    ) -> Optional[Union[GaussianModel, CompoundModel_[GaussianModel]]]:
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
    def adoptFit(
        self,
        fit: Union[
            GaussianModel,
            _VProfileCopy,
            CompoundModel_[Union[GaussianModel, _VProfileCopy]],
        ],
        *,
        fit_info: FitInfo | None = None,
        update_emission: bool = False,
    ) -> Self:
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
                n_free = sum(get_free_params(model).values())
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

    @validate_call
    def applyFit(
        self,
        fit: Union[
            GaussianModel,
            _VProfileCopy,
            CompoundModel_[Union[GaussianModel, _VProfileCopy]],
        ],
        *,
        fit_info: FitInfo | None = None,
        freeze: bool = False,
        update_emission: bool = False,
    ) -> Self:
        """
        Applies the 'fit' and 'fit_info' values to all 'LWindow' classes.

        Notes
        -----
        The keyword argument 'freeze' is set to True when applying models,
        meaning that copies of models are applied, and that any models copying
        other models' velocity profiles will have their parameters frozen and
        their 'tie' attributes disabled.
        """
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
                n_free = sum(get_free_params(model).values())
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
                freeze=freeze,
                update_emission=update_emission,
            )

        return self
