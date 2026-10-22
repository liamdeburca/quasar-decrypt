__all__ = ["LWindow"]

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import field
from itertools import pairwise
from logging import getLogger
from math import floor, prod
from re import Pattern
from typing import Any, ClassVar, Literal, Optional, Self, Union

import numpy as np
import pydantic
from astropy.constants import c
from quasar_models import (
    BalmerModel,
    EmModel,
    GaussianModel,
    HostGalaxyModel,
    IronModel,
    PowerLawModel,
)
from quasar_models.line import (
    VProfileCopy,
    VProfileCopy1G,
    _VProfileCopy,
    create_vprofilecopy,
    rename_gaussians,
)
from quasar_models.modeling import Fitter, PrepareModel
from quasar_models.utils.astropy import (
    _apply_bounds,
    get_largest_possible_bounds,
)
from quasar_typing.astropy import CompoundModelType
from quasar_typing.bounds import AstropyBounds
from quasar_typing.literals import BootstrapType, Scale, Variant
from quasar_typing.multiprocessing import PoolType
from quasar_typing.numpy import BoolVector, FloatVector, RandomStateType
from quasar_typing.scipy import OptimizeResultType
from quasar_typing.string_selections import (
    BackgroundFlux,
    ModelTypes,
    VaryLines,
)
from quasar_utils.continuum_fit_result import ContinuumFitResult
from quasar_utils.decorators import validated_apply_info_to_method
from quasar_utils.setup import FitterKwargs

from quasar_decrypt.utils import (
    ContiguousMaskedCoords,
    MaskedCoords,
    ReadOnlyMaskedCoords,
    _SpecWindow,
    stopwatch,
)

logger = getLogger(__name__)
C_KMS: float = c.to("km/s").value

type LineModel = Union[EmModel, CompoundModelType[EmModel]]


@pydantic.dataclasses.dataclass(kw_only=True)
class LWindow(_SpecWindow):
    names: set[str] = field(default_factory=set, kw_only=True)

    linetypes: dict[str, Literal["n", "b"]] = field(default_factory=dict, kw_only=True)
    complexes: dict[str, str] = field(default_factory=dict, kw_only=True)
    waves: dict[str, float] = field(default_factory=dict, kw_only=True)
    n_maxs: dict[str, int] = field(default_factory=dict, kw_only=True)
    needs_line: dict[str, str] = field(default_factory=dict, kw_only=True)

    strength_bounds: dict[str, tuple] = field(
        default_factory=dict, 
        kw_only=True,
    )
    fwhm_v_bounds: dict[str, tuple] = field(
        default_factory=dict, 
        kw_only=True,
    )
    v_off_bounds: dict[str, tuple] = field(
        default_factory=dict, 
        kw_only=True,
    )

    # Dictionary of string-string pairs, with the first string being the name of 
    # a line covered by this LWindow instance, and the second string being the 
    # name of the line to copy. 
    is_copy_of: dict[str, str] = field(default_factory=dict, kw_only=True)
    is_multiplet_of: dict[str, str] = field(default_factory=dict, kw_only=True)

    scale_init: dict[str, float] = field(default_factory=dict, kw_only=True)
    scale_bounds: dict[str, AstropyBounds] = field(default_factory=dict, kw_only=True)
    scale_fixed: dict[str, bool] = field(default_factory=dict, kw_only=True)

    copies_to: dict[str, list[tuple[int | None, str]]] = field(
        default_factory=lambda: defaultdict(list), 
        kw_only=True,
    )

    i_bounds: dict[str, tuple[float, float]] = field(
        default_factory=dict, kw_only=True
    )
    blacklist: dict[str, bool] = field(default_factory=dict, kw_only=True)
    _blacklist: dict[str, bool] = field(default_factory=dict, kw_only=True)

    neighbours: tuple[int | None, int | None] = field(
        default=(None, None), kw_only=True
    )
    # prev_model: LineModel | None = field(default=None, kw_only=True)
    model: LineModel | None = field(default=None, kw_only=True)
    fit: LineModel | None = field(default=None, kw_only=True)
    fit_info: OptimizeResultType | None = field(default=None, kw_only=True)

    fits: dict[int, LineModel] = field(default_factory=dict, kw_only=True)
    fit_infos: dict[int, OptimizeResultType] = field(default_factory=dict, kw_only=True)

    cropped: set[str] = field(default_factory=set, kw_only=True)

    _y_em_contrib: FloatVector | None = field(default=None, init=False)

    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({"all", "em"})

    def __post_init__(self):
        if self._y_em_contrib is None:
            self._y_em_contrib = np.zeros_like(self._x, dtype=np.float64)

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["names"] = self.names
        state["linetypes"] = self.linetypes
        state["complexes"] = self.complexes
        state["waves"] = self.waves
        state["n_maxs"] = self.n_maxs
        state["needs_line"] = self.needs_line
        state["strength_bounds"] = self.strength_bounds
        state["fwhm_v_bounds"] = self.fwhm_v_bounds
        state["v_off_bounds"] = self.v_off_bounds
        state["is_copy_of"] = self.is_copy_of
        state["is_multiplet_of"] = self.is_multiplet_of
        state["scale_init"] = self.scale_init
        state["scale_bounds"] = self.scale_bounds
        state["scale_fixed"] = self.scale_fixed
        state["copies_to"] = self.copies_to
        state["i_bounds"] = self.i_bounds
        state["blacklist"] = self.blacklist
        state["_blacklist"] = self._blacklist
        state["neighbours"] = self.neighbours
        state["model"] = self.model
        state["fit"] = self.fit
        state["fit_info"] = self.fit_info
        state["fits"] = self.fits
        state["fit_infos"] = self.fit_infos
        state["cropped"] = self.cropped
        state["_y_em_contrib"] = self._y_em_contrib
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.names = state.get("names")
        self.linetypes = state.get("linetypes")
        self.complexes = state.get("complexes")
        self.waves = state.get("waves")
        self.n_maxs = state.get("n_maxs")
        self.needs_line = state.get("needs_line")
        self.strength_bounds = state.get("strength_bounds")
        self.fwhm_v_bounds = state.get("fwhm_v_bounds")
        self.v_off_bounds = state.get("v_off_bounds")
        self.is_copy_of = state.get("is_copy_of")
        self.is_multiplet_of = state.get("is_multiplet_of")
        self.scale_init = state.get("scale_init")
        self.scale_bounds = state.get("scale_bounds")
        self.scale_fixed = state.get("scale_fixed")
        self.copies_to = state.get("copies_to")
        self.i_bounds = state.get("i_bounds")
        self.blacklist = state.get("blacklist")
        self._blacklist = state.get("_blacklist")
        self.neighbours = state.get("neighbours")
        self.model = state.get("model")
        self.fit = state.get("fit")
        self.fit_info = state.get("fit_info")
        self.fits = state.get("fits")
        self.fit_infos = state.get("fit_infos")
        self.cropped = state.get("cropped")
        self._y_em_contrib = state.get("_y_em_contrib")

    @property
    def sample(self):
        from quasar_errors.model_samples import GaussianSampleList
        if (model := self.getModel()) is None:
            return None
        return GaussianSampleList.fromGaussianModels(model)

    @validated_apply_info_to_method(subjects=("loading", "lines", "nonlinear"))
    def __call__(
        self,
        *,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        limited: bool = False,
        sigma_res: float | None = None,
        w: int | None = None,
        v_sep: float | None = None,
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

        assert len(self.waves) > 0

        logger.debug(f"Starting pipeline for {self.__str__(True)}")

        with stopwatch() as watch:
            self.prepareLines.__wrapped__(
                self,
                v_sep=v_sep,
                min_fittable_total=min_fittable_total,
                min_fittable_ratio=min_fittable_ratio,
            )
            if with_neighbours:
                success = self.prepareNeighbours.__wrapped__(
                    self,
                    sigma_res=sigma_res,
                )
                if not success:
                    logger.info(
                        ">>> Failed pipeline during 'prepareNeighbours'!"
                    )
                    return False

            success = self.instantiateModels.__wrapped__(
                self,
                bg_flux=bg_flux,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                with_neighbours=with_neighbours,
            )
            if not success:
                logger.info(
                    ">>> Failed pipeline during 'instantiateModels'!"
                )
                return False

            success = self.makeInitialFit.__wrapped__(
                self,
                bg_flux=bg_flux,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                with_neighbours=with_neighbours,
                evaluate_initial=evaluate_initial,
                fitter_kwargs=fitter_kwargs,
            )
            if not success:
                logger.info(">>> Failed pipeline during 'makeInitialFit'!")
                return False

            success = self.makeFinalFit.__wrapped__(
                self,
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
                logger.info(">>> Failed pipeline during 'makeFinalFit'!")
                return False

        logger.debug(f">>> Finished entire pipeline in {1e3 * watch.elapsed:.1f} ms.")
        return True

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
        line: str | float | None = None,
        complex_name: str | None = None,
        limited: bool = True,
        v_sep: float | None = None,
        as_tuple: bool = False,
    ) -> BoolVector | tuple[BoolVector, BoolVector]:
        original = self._update_original_flag(
            original=original,
            log=False,
        )
        suffix = "_original" if original else ""

        primary_mask, secondary_mask = super().getMask.__wrapped__(
            self, 
            original=original,
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=valid,
            log_valid=log_valid,
            as_tuple=True,
        )
        if with_neighbours and self.neighbours != (None, None):
            for idx in filter(lambda i: i is not None, self.neighbours):
                cwindow = self.spectrum.pl[idx]
                _p, _s = cwindow.getMask.__wrapped__(
                    cwindow,
                    original=original,
                    covered=covered,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                    valid=valid,
                    log_valid=log_valid,
                    as_tuple=as_tuple,
                )
                primary_mask |= _p
                secondary_mask |= _s

        bounds: tuple[float, float] | None = None
        if complex_name is not None:
            bounds = self.i_bounds[complex_name]
        elif line is not None:
            if isinstance(line, str):
                if limited:
                    # Based on 'i_bounds' -> find complex name
                    bounds = self.i_bounds[self.complexes[line]]
                else:
                    wave = self.waves[line]
                    bounds = (wave * (1 - v_sep), wave * (1 + v_sep))
            else:
                bounds = (line * (1 - v_sep), line * (1 + v_sep))

        if bounds is not None:
            x = getattr(self, "_x" + suffix)
            mask = (bounds[0] <= x) & (x < bounds[1])
            primary_mask &= mask
            secondary_mask &= mask

        if as_tuple:
            return primary_mask, secondary_mask
        return primary_mask & secondary_mask

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
        line: float | str | None = None,
        complex_name: str | None = None,
        limited: bool = True,
        v_sep: float | None = None,
        shrink: bool | None = None,
    ) -> MaskedCoords | ReadOnlyMaskedCoords | ContiguousMaskedCoords:
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
            complex_name=complex_name,
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

    @validated_apply_info_to_method(subjects=("lines",))
    def add(
        self,
        *,
        name: str,
        complex_name: str,
        wave: float, 
        linetype: Literal["n", "b"],
        n_max: int,
        needs_line: str | None = None,
        is_copy_of: str | None = None,
        strength_bounds: AstropyBounds | None = None,
        v_off_bounds: AstropyBounds | None = None,
        fwhm_v_bounds: AstropyBounds | None = None,
        scale_init: float | None = None,
        scale_bounds: AstropyBounds | None = None,
        scale_fixed: bool | None = None,
        force_add: bool = False,
    ) -> bool:
        """
        Adds a given line to the SubSlice under the condition that it falls
        within the covered wavelength range.
        """
        if (self.x_bounds[0] < wave < self.x_bounds[1]) or force_add:
            self.names.add(name)
            self.linetypes[name] = linetype
            self.complexes[name] = complex_name

            self.waves[name] = wave
            self.n_maxs[name] = n_max

            self.strength_bounds[name] = strength_bounds
            self.fwhm_v_bounds[name] = fwhm_v_bounds
            self.v_off_bounds[name] = v_off_bounds

            if bool(needs_line):
                self.needs_line[name] = needs_line

            if bool(is_copy_of):
                self.is_copy_of[name] = is_copy_of
                self.scale_init[name] = scale_init
                self.scale_bounds[name] = scale_bounds
                self.scale_fixed[name] = scale_fixed

            return True

        return False

    @validated_apply_info_to_method(subjects=("nonlinear",))
    def addCustomModel(
        self,
        new_model: EmModel,
        *,
        force_add: bool = False,
        refit: bool = False,
        update_flux: bool = False,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        fitter_kwargs: FitterKwargs | None = None,
    ) -> None:
        success = self.add.__wrapped__(
            self,
            name=new_model.pure_name,
            complex_name=new_model.pure_name,
            wave=new_model.wave,
            linetype=new_model.linetype,
            n_max=1,
            needs_line=None,
            is_copy_of=None,
            strength_bounds=new_model.strength.bounds,
            fwhm_v_bounds=new_model.fwhm_v.bounds,
            v_off_bounds=new_model.v_off.bounds,
            scale_init=None,
            scale_bounds=None,
            scale_fixed=None,
            force_add=force_add,
        )
        if not success:
            msg = f"Failed to add custom model to `Lwindow` with bounds {self.x_bounds}!"
            logger.info(msg)
            raise ValueError(msg)

        if refit:
            self.model = self.fit + new_model
            self.fitModel.__wrapped__(
                self,
                update_flux=update_flux,
                bg_flux=bg_flux,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                with_neighbours=with_neighbours,
                fitter_kwargs=fitter_kwargs,
            )
        else:
            self.applyFit.__wrapped__(
                self,
                self.fit + new_model,
                update_emission=update_flux,
            )

    @validated_apply_info_to_method(subjects=("lines",))
    def prepareLines(
        self,
        *,
        grade_lines: bool = True,
        v_sep: float | None = None,
        min_fittable_total: int | None = None,
        min_fittable_ratio: float | None = None,
    ) -> Self:
        """
        Performs the following steps:
        1.  Sorts the expected emission lines, and truncates the SubSlice if
            necessary.
        2.  Checks for heavy absorption.

        Parameters
        ----------
        v_sep : float
            Velocity separation (units of the speed of light) used to define the
            spectral region relevant for a given emission line. Default is 'self.default_kwargs['v_sep']'.
        crit_abs_ratio : float, optional
            Critical absorption ratio (# of absorbed pixels / # of pixels).
            Lines with absorption ratios exceeding this value are deemed as too
            absorbed, and are limited to a single profile function. If None, no
            emission lines are limited based on absorption. Default is 'self.default_kwargs['crit_abs_ratio'].
        """
        from .utils import common_middle, update_v_off_bounds
        # STEP 1

        logger.debug(
            f"Preparing lines: {grade_lines=}, {v_sep=}, {min_fittable_total=}, "
            f"{min_fittable_ratio=}"
        )

        logger.debug(">>> Clearing 'i_bounds'/'blacklist'/'_blacklist'")
        self.i_bounds.clear()
        self.blacklist.clear()
        self._blacklist.clear()

        if len(self.waves) == 0:
            logger.debug(">>> No lines to prepare!")
            return self
        elif len(self.waves) == 1:
            logger.debug(">>> Only 1 line to prepare!")
            name = next(iter(self.names))

            wave = self.waves[name]
            (lower, upper) = self.v_off_bounds[name]

            self.x_bounds = (
                max(self.x_bounds[0], wave * (1 - v_sep)),
                min(self.x_bounds[1], wave * (1 + v_sep)),
            )

            self.i_bounds[self.complexes[name]] = self.x_bounds
            self.v_off_bounds[name] = (
                max(lower, (self.x_bounds[0] / wave - 1) * C_KMS),
                min(upper, (self.x_bounds[1] / wave - 1) * C_KMS),
            )
        else:
            logger.debug(f">>> {len(self.waves)} lines to prepare!")

            # Create dictionary of complexes' bluest and reddest lines
            complex_to_lines: dict[str, set[str]] = defaultdict(set)
            for name, complex_name in self.complexes.items():
                complex_to_lines[complex_name].add(name)
            logger.debug(f">>> Using {complex_to_lines=}")

            complex_bounds: dict[str, list[float]] = {}
            for complex_name, names in complex_to_lines.items():
                waves = sorted(self.waves[name] for name in names)
                complex_bounds[complex_name] = [waves[0], waves[-1]]
            logger.debug(f">>> Using {complex_bounds=}")

            del complex_to_lines

            # Validate that complexes do not overlap
            complex_bounds = dict(sorted(
                complex_bounds.items(),
                key=lambda item: item[1][0],
            ))
            complex_lbs = np.fromiter(
                (val[0] for val in complex_bounds.values()),
                dtype=np.float64,
            )
            complex_ubs = np.fromiter(
                (val[1] for val in complex_bounds.values()),
                dtype=np.float64,
            )
            if not (complex_lbs[1:] >= complex_ubs[:-1]).all():
                msg = f"Complexes overlap: {complex_bounds=}"
                logger.critical(">>>" + msg)
                raise ValueError(msg)

            n_complexes = len(complex_bounds)
            _i_bounds = np.empty(n_complexes + 1)

            _i_bounds[0] = max(
                complex_lbs[0] * (1 - v_sep),
                self.x_bounds[0],
            )
            _i_bounds[-1] = min(
                complex_ubs[-1] * (1 + v_sep),
                self.x_bounds[1],
            )
            if n_complexes > 1:
                _i_bounds[1:-1] = [
                    common_middle(ub_left, lb_right, v_sep, v_sep)
                    for ub_left, lb_right in zip(
                        complex_ubs[:-1], complex_lbs[1:]
                    )
                ]

            self.i_bounds = i_bounds = dict(zip(
                complex_bounds.keys(), 
                pairwise(_i_bounds),
            ))
            self.x_bounds = x_bounds = (_i_bounds[0], _i_bounds[-1])
            logger.debug(f">>> Found:  {i_bounds=}, {x_bounds=}")

            # Update velocity offset bounds:
            # - Ignore multiplets / velocity profile copies. These are assumed 
            #   to be constrained enough to not need updating.

            _waves: dict[str, float] = {
                name: wave
                for name, wave in self.waves.items()
                if (name not in self.is_copy_of)     # Isn't a copy of ...
                    and (name not in self.copies_to) # Isn't used to copy to ...
            }
            narrow_waves: dict[str, float] = {
                name: wave
                for name, wave in _waves.items()
                if self.linetypes[name] == "n"
            }
            broad_waves: dict[str, float] = {
                name: wave
                for name, wave in _waves.items()
                if self.linetypes[name] == "b"
            }

            # Narrow
            if narrow_waves:
                logger.debug(
                    ">>> Updating velocity offset bounds for narrow lines: "
                    f"{narrow_waves}"
                )
                update_v_off_bounds(
                    v_off_bounds=self.v_off_bounds,
                    waves=narrow_waves,
                    x_bounds=self.x_bounds,
                )
            # Broad
            if broad_waves:
                logger.debug(
                    ">>> Updating velocity offset bounds for broad lines: "
                    f"{broad_waves}"
                )
                update_v_off_bounds(
                    v_off_bounds=self.v_off_bounds,
                    waves=broad_waves,
                    x_bounds=self.x_bounds,
                )

        # Check multiplets
        self.setMultiplets()

        # STEP 2
        msg = ">>> Populating blacklist: "

        for name in sorted(self.names, key=lambda n: self.waves[n]):
            # Blacklist lines which are:
            # - Limited to a single Gaussian, or
            # - Copies of other lines.
            msg += f"('{name}') "
            val = True
            if self.n_maxs[name] == 1:
                msg += "limited to 1 Gaussian "
            elif self.is_copy_of.get(name, None) in self.names:
                msg += "copy of another line "
            elif name in self.is_multiplet_of:
                msg += "multiplet of another line "
            else:
                msg += "not blacklisted "
                val = False

            self.blacklist[name] = self._blacklist[name] = val

        logger.debug(msg)
        logger.debug(f">>> Using blacklist={self.blacklist}")

        if grade_lines:
            _ = self.gradeLines.__wrapped__(
                self,
                with_neighbours=True,
                min_fittable_total=min_fittable_total,
                min_fittable_ratio=min_fittable_ratio,
                v_sep=v_sep,
            )

        return self

    def setMultiplets(self) -> None:
        """
        Simplify the `copies_to` dictionary for the special case when line A is 
        a velocity profile copy of an external line X, while line B is a 
        multiplet of line A. In this case, the `copies_to` dictionary can be 
        simplified s.t. line B is tied to line X directly, rather than through 
        line A.

        Notes
        -----
        For each line that is a velocity profile copy of an internal line, do 
        the following: 
        (A) if the master line is not a copy, do nothing; 
        (B) if the master line is a copy of an external line, update the 
        `copies_to` (and `is_multiplet_of`) dictionary accordingly;
        (C) if the master line is a copy of an internal line, repeat the same 
        logic recursively until an external or a non-copy internal line is 
        found.
        """
        logger.debug("Checking for multiplets: clearing 'is_multiplet_of'/'copies_to'")
        self.is_multiplet_of.clear()

        keys_to_remove: list[str] = []
        for name, tup in self.copies_to.items():
            if tup[0] is None:
                keys_to_remove.append(name)
        for key in keys_to_remove:
            self.copies_to.pop(key)

        for mimic, master in self.is_copy_of.items():
            if master in self.names:
                scale_inits = [self.scale_init[mimic]]
                scale_bounds = [self.scale_bounds[mimic]]
                scale_fixeds = [self.scale_fixed[mimic]]

                while (
                    master in self.is_copy_of
                    and self.is_copy_of[master] in self.names
                ):
                    scale_inits.append(self.scale_init[master])
                    scale_bounds.append(self.scale_bounds[master])
                    scale_fixeds.append(self.scale_fixed[master])

                    master = self.is_copy_of[master]

                self.is_copy_of[mimic] = master
                self.scale_init[mimic] = prod(scale_inits)
                self.scale_bounds[mimic] = get_largest_possible_bounds(
                    scale_bounds, 
                    min_val=self.info.lines.scale_bounds[0], 
                    max_val=self.info.lines.scale_bounds[1],
                )
                self.scale_fixed[mimic] = any(scale_fixeds)

                if master in self.names:
                    # Internal copy
                    logger.debug(f">>> Found internal copy: {mimic=}, {master=}")
                    self.copies_to[master].append((None, mimic))
                    self.is_multiplet_of[mimic] = master
                    continue

            logger.debug(f">>> Found external copy: {mimic=}, {master=}")

    @validated_apply_info_to_method(
        subjects=("loading",), specific_kwargs={"sigma_res"}
    )
    def prepareNeighbours(self, *, sigma_res: float | None = None) -> Self:
        from scipy.ndimage import binary_dilation

        msg = "Preparing neighbours for {}: ".format(
            self.__str__(simple=True).removesuffix(".")
        )
        if self.spectrum.pl is None:
            msg += "no available continuum windows!"
            logger.critical(msg)
            raise ValueError(msg)

        iterations = sigma_res
        if isinstance(sigma_res, float):
            iterations = floor(sigma_res / self.info.loading.sigma_res)

        vicinity = binary_dilation(self.mask, iterations=iterations)

        left = None
        right = None
        for i, cwindow in enumerate(self.spectrum.pl):
            if (left is not None) and (right is not None):
                break
            if not (cwindow.mask & vicinity).any():
                continue

            if cwindow.x_bounds[1] < self.x_bounds[0]:
                left = i
            elif self.x_bounds[1] < cwindow.x_bounds[0]:
                right = i

        msg += "no bluewise CWindow found, " \
            if left is None \
            else "found bluewise CWindow, "
        msg += "no redwise CWindow found." \
            if right is None \
            else "found redwise CWindow."

        self.neighbours = (left, right)

        return self

    @validated_apply_info_to_method(subjects=("lines",))
    def gradeLines(
        self,
        with_neighbours: bool = False,
        *,
        min_fittable_total: int | None = None,
        min_fittable_ratio: float | None = None,
        v_sep: float | None = None,
    ) -> Self:
        logger.debug(f"Grading lines in {self.__str__(simple=True)}:")

        lines_to_remove: set[str] = set()

        for name, wave in self.sortWaves():
            if (name in self.is_copy_of) or (name in self.is_multiplet_of):
                continue

            complex_name = self.complexes[name]
            mask = self.getMask.__wrapped__(
                self,
                with_neighbours=with_neighbours,
                covered=True,
                complex_name=complex_name,
                limited=True,
                v_sep=v_sep,
            )
            absorbed_pixels = self._absorbed_pixels[mask]
            valid_pixels = self._valid_pixels[mask]

            n = mask.sum()
            f = (valid_pixels & ~absorbed_pixels).sum()
            a = (valid_pixels & absorbed_pixels).sum()
            i = (~valid_pixels).sum()

            msg = (
                ">>> [{:.1f} <-> {:.1f}] w/ {}/{n} (fit.),"
                "{}/{n} (abs.), {}/{n} (inv.): ".format(
                    *self.i_bounds[complex_name], f, a, i, n=n
                )
            )

            if f == 0:
                msg += "No valid data -> "\
                    f"Removing line '{name}' at {wave:.1f}!"
                lines_to_remove.add(name)
            elif (f / n < min_fittable_ratio) or (f < min_fittable_total):
                msg += "Not enough valid data -> "\
                    f"Blacklisting line '{name}' at {wave:.1f}!"
                self.blacklist[name] = True
            else:
                msg += f"Enough valid data for line '{name}' at {wave:.1f}!"

            logger.debug(msg)

        for name in lines_to_remove:
            self.removeLine(name)

        return self

    @validated_apply_info_to_method(
        subjects=("lines",), specific_kwargs={"v_sep"}
    )
    def instantiateModels(
        self,
        *,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        v_sep: float | None = None,
    ) -> Self:
        """
        Quickly instantiates each submodel using the data in its nearest
        vicinity. The amgorithm is described in my thesis (steps 1 and 2 prefer
        smoothed flux density values, step 3 does not.):

        1.  The mean is calculated using the mean rest wavelength value for
            pixels whose flux density values exceed half the maximum value. The
            corresponding velocity offset is calculated using the theoretical
            rest wavelength. The velocity offset is adjusted to fit the accepted
            bounds.

            !!! Method is from lmfit.

        2.  The intrinsic velocity dispersion is inferred using the FWQM (full
            width at a quarter maximum) of the data, after correcting for a
            non-zero velocity resolution. The velocity dispersion is adjusted to
            fit the accepted bounds.

        3. The line strength is calculated as the discrete integral of the data.

        Lastly, the 'blacklist' (dict), 'fits' (dict) and 'fits_info' (dict)
        attributes are reset to their default values.

        Note
        ----
        Steps 1 and 2 prefer smoothed data. Step 3 does not.
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        logger.debug(
            "Instantiating models for {}:".format(
                self.__str__(True).removesuffix(".")
            )
        )

        models: dict[str, GaussianModel] = {}
        for name, wave in self.sortWaves():
            msg = f">>> Instantiating '{name}' at {wave:.1f}: "
            if name in self.cropped:
                msg += f"found in cropped lines ({self.cropped}) -> skipping!"
                logger.debug(msg)
                continue

            log = logger.debug
            if name in self.is_multiplet_of:
                master_name = self.is_multiplet_of[name]
                msg += f"line is a multiplet of '{master_name}', "
                if master_name in models:
                    logger.debug(msg + "instantiating as a copy!")
                    models[name] = model = VProfileCopy1G.from_model(
                        wave,
                        name,
                        self.linetypes[name],
                        models[self.is_multiplet_of[name]],
                        strength_scale_value=self.scale_init[name],
                        strength_scale_bounds=self.scale_bounds[name],
                        strength_scale_fixed=self.scale_fixed[name],
                    )
                    model.v_off_1.bounds = self.v_off_bounds[name]
                    model.v_off_1.value = _apply_bounds(
                        model.v_off_1.value, 
                        model.v_off_1.bounds,
                    )
                    continue

                log = logger.info
                msg += f"could not find '{master_name}', defaulting to "\
                    "standard instantiation: "

            masked_coords = self.getMaskedCoords.__wrapped__(
                self,
                original=False,
                mode="c",
                covered=True,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                with_neighbours=with_neighbours,
                valid=True,
                log_valid=False,
                bg_flux=bg_flux,
                complex_name=self.complexes[name],  # Integrate over the complex, not the individual line
                limited=True,
                v_sep=v_sep,
                shrink=None,
            )
            y_smooth = masked_coords.y_smooth
            if np.isnan(y_smooth).all():
                y_smooth = masked_coords.y

            log(msg + f"instantiating model using {masked_coords.size} data points!")

            models[name] = GaussianModel.instantiate.__wrapped__(
                GaussianModel,
                wave,
                masked_coords.x,
                masked_coords.y,
                y_smooth,
                sigma_res=self.info.loading.sigma_res,
                linetype=self.linetypes[name],
                name=name,
                strength_bounds=self.strength_bounds[name],
                v_off_bounds=self.v_off_bounds[name],
                fwhm_v_bounds=self.fwhm_v_bounds[name],
                logger=logger,
            )

        if len(models) == 0:
            logger.info("Did not instantiate a single model!")
            return False

        submodels = list(models.values())
        self.model = sum(submodels[1:], start=submodels[0])
        # Remove current fit if existant
        self.fit = None

        # Reset blacklist and previous fits
        self.blacklist = self._blacklist.copy()
        self.fits.clear()
        self.fit_infos.clear()

        return self

    @validated_apply_info_to_method(
        subjects=("nonlinear",), specific_kwargs={"fitter_kwargs"}
    )
    def fitModel(
        self,
        *,
        update_flux: bool = False,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        fitter_kwargs: FitterKwargs | None = None,
    ) -> bool:
        """
        Fits the available model using the specified non-linear fitting
        algorithm. The resultant fit is assigned to the 'fit' attribute, and
        saved in the 'fits' (dict) attribute whereafter in can be accessed using
        the corresponding fit complexity (number of profile functions). The fit
        info is saved in the 'fit_infos' (dict) attribute.
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        msg = "Fitting model for {}: ".format(
            self.__str__(simple=True).removesuffix(".")
        )

        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            original=False,
            mode="c",
            covered=True,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            with_neighbours=with_neighbours,
            valid=True,
            log_valid=False,
            bg_flux=bg_flux,
            shrink=None,
        )
        msg += f"no. of Gaussians: {self.model.n_submodels}, "
        msg += f"no. of data points: {masked_coords.size}. "

        preparer = PrepareModel.fromCoords(masked_coords, self.model)
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

        logger.debug(msg)

        self.applyFit.__wrapped__(
            self,
            fit,
            fit_info=fitter.fit_info,
            update_emission=update_flux,
        )
        return True

    @validated_apply_info_to_method(subjects=("lines", "nonlinear"))
    def makeInitialFit(
        self,
        *,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        evaluate_initial: float | None = None,
        fitter_kwargs: FitterKwargs | None = None,
    ) -> bool:
        """
        If necessary, instantiates the initial model (one profile function per
        known emission line), and fits the model.
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        if self.model is None:
            success = self.instantiateModels.__wrapped__(
                self,
                bg_flux=bg_flux,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                with_neighbours=with_neighbours,
            )
            if not success:
                return False

        success = self.fitModel.__wrapped__(
            self,
            bg_flux=bg_flux,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            with_neighbours=with_neighbours,
            fitter_kwargs=fitter_kwargs,
        )
        if not success:
            return False

        if isinstance(evaluate_initial, (int, float)):
            masked_coords = self.getMaskedCoords.__wrapped__(
                self,
                original=False,
                mode="c",
                covered=True,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                with_neighbours=with_neighbours,
                valid=True,
                log_valid=False,
                bg_flux=bg_flux,
                shrink=None,
            )

            fs = (self.fit,) if self.fit.n_submodels == 1 else self.fit
            for f in filter(
                lambda f: not self.blacklist[f.pure_name],
                fs,
            ):
                # Interpolate noise level and compare with peak flux density
                crit_val = evaluate_initial * np.interp(
                    f.mu,
                    masked_coords.x,
                    masked_coords.dy,
                )
                self.blacklist[f.pure_name] = f.peak < crit_val

        return True

    @validated_apply_info_to_method(subjects=("lines",), specific_kwargs={"w"})
    def addLine(
        self,
        *,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        w: int | None = None,  # ! Make 'lines'-specific window size?
    ) -> tuple[GaussianModel, bool]:
        if bg_flux is None:
            bg_flux = self.default_bg

        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            original=False,
            mode="c",
            covered=True,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            with_neighbours=with_neighbours,
            valid=True,
            log_valid=False,
            bg_flux=bg_flux,
            shrink=None,
        )
        if self.fit.n_submodels == 1:
            data = (
                masked_coords.x,
                masked_coords.dy,
                (masked_coords.y - self.fit(masked_coords.x))
                / masked_coords.dy,
            )
            submodel = self.fit
        else:
            fs = self.fit
            h = w // 2

            # Filter out blacklisted lines, including velocity profile copies.
            valid_lines = list(filter(
                lambda f: not self.blacklist[f.pure_name],
                fs,
            ))
            waves = [f.wave for f in valid_lines]

            z = abs(
                (masked_coords.y - self.fit(masked_coords.x))
                / masked_coords.dy
            )
            z_convolved = np.convolve(
                z,
                np.exp(-(np.linspace(-3, 3, w) ** 2)),
                mode="valid",
            )
            z_interp = np.interp(waves, masked_coords.x[h:-h], z_convolved)

            submodel: GaussianModel = valid_lines[np.argmax(z_interp).flatten()[0]]

            name = submodel.pure_name
            complex_name = self.complexes[name]

            (lb, ub) = self.i_bounds[complex_name]
            mask = (lb <= masked_coords.x) & (masked_coords.x < ub)
            data = (masked_coords.x[mask], masked_coords.dy[mask], z[mask])

        # Check if submodel is a 'master' model in a multiplet
        return (
            submodel.makeCopy(*data),
            submodel.pure_name in self.is_multiplet_of.values(),
        )

    @validated_apply_info_to_method(subjects=("lines", "nonlinear"))
    def makeFinalFit(
        self,
        *,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        limited: bool = False,
        w: int | None = None,
        aggressive: bool | None = None,
        crop: bool | None = None,
        measure: str | None = None,
        reverse: bool | None = None,
        evaluate_initial: float | None = None,
        v_sep: float | None = None,
        fitter_kwargs: FitterKwargs | None = None,
    ) -> Self:
        """
        !!! veto func?
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        # Instantiate if necessary
        if self.fit is None:
            msg = "Cannot finalises fit without an initial fit: call "\
                "`makeInitialFit` first!"
            logger.critical(msg)
            raise ValueError(msg)
        
        continue_fitting: bool = True

        # Check if any lines aren't already blacklisted
        if all(self.blacklist.values()):
            # No additional models necessary
            logger.debug(
                "All lines are already blacklisted -> no additional models "
                "necessary!"
            )
            continue_fitting = False
        elif self.fit.n_submodels <= 2:
            # Check if under-fitted
            logger.debug(
                "Only 1 or 2 submodels -> checking if they are under-fitted..."
            )
            continue_fitting = self.isUnderFitted.__wrapped__(
                self,
                self.fit.n_submodels,
                bg_flux=bg_flux,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                with_neighbours=with_neighbours,
                line=self.fit.wave if self.fit.n_submodels == 1 else None,
                limited=limited,
                v_sep=v_sep,
            )

        current_config = self.getConfiguration()
        all_configs = [tuple(current_config.values())]

        if continue_fitting:
            logger.debug("Starting component addition loop...")

        while continue_fitting:
            self.updateBlacklistFromConfiguration()
            if all(self.blacklist.values()):
                logger.debug("All lines are blacklisted -> breaking loop!")
                break

            # Add a line component
            new_model, check_multiplet = self.addLine.__wrapped__(
                self,
                bg_flux=bg_flux,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                with_neighbours=with_neighbours,
                w=w,
            )

            current_config[new_model.pure_name] += 1
            msg = f"Adding line '{new_model.name}': {current_config=}, "

            ms = list(self.fit + new_model)
            current_config[new_model.pure_name] += 1

            # Gaussians used to instantiate the new velocity profile copy/copies
            _ms: list[GaussianModel] = [
                m
                for m in ms
                if m.pure_name == new_model.pure_name
            ]
            master_model = sum(_ms[1:], start=_ms[0])
            del _ms
            
            if check_multiplet:
                # The new model is a master model in a multiplet: 
                # - Replace the existing velocity-profile copy with a new one 
                #   with one more component.
                for mimic in (
                    mimic
                    for window_idx, mimic in self.copies_to[new_model.pure_name]
                    if window_idx is None
                ):
                    # Velocity profile copy to replace
                    old_mimic: VProfileCopy = next(
                        m
                        for m in ms
                        if m.pure_name == mimic
                    )

                    assert old_mimic.n_profiles + 1 == master_model.n_submodels

                    new_mimic = create_vprofilecopy(
                        master_model,
                        mimic,
                        wave=self.waves[mimic],
                        strength_scale_value=self.scale_init[mimic],
                        strength_scale_bounds=self.scale_bounds[mimic],
                        strength_scale_fixed=self.scale_fixed[mimic],
                    )
                    # Replace the old vel.-profile copy with the new one
                    ms.remove(old_mimic)
                    ms.append(new_mimic)

            ms = sorted(ms, key=lambda m: m.sorting_key)
            self.model = sum(ms[1:], start=ms[0])
            rename_gaussians(self.model)

            n_prev: int = self.fit.n_submodels
            n_curr: int = self.model.n_submodels

            if n_curr <= n_prev:
                raise ValueError(
                    f"No. of components has not increased: {n_curr=}, {n_prev=}"
                )

            self.fitModel.__wrapped__(
                self,
                bg_flux=bg_flux,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                with_neighbours=with_neighbours,
                fitter_kwargs=fitter_kwargs,
            )

            # Check if current configuration has already been tried
            _config = tuple(current_config.values())
            if _config in all_configs:
                break

            all_configs.append(_config)

            # Check if model is over-fitted
            is_over, _ = self.isOverFitted.__wrapped__(
                self,
                n_prev,
                n_curr,
                bg_flux=bg_flux,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                with_neighbours=with_neighbours,
                line=new_model.wave,
                limited=limited,
                v_sep=v_sep,
            )
            if is_over:
                # Check whether models are saturated (blacklisted) or touching
                # their respective bounds.
                msg += "model is over-fitted, "
                fs = (self.fit,) if n_curr == 1 else self.fit
                is_redundant: list[bool] = [
                    f.isTouchingBounds() & self.blacklist[f.pure_name]
                        if isinstance(f, GaussianModel)
                        else False
                    for f in fs
                ]
                msg += f"{is_redundant=}, "

                if all(is_redundant):
                    msg += "all submodels are redundant -> stopping loop!"
                    continue_fitting = False
                elif any(is_redundant) and crop:
                    msg += f"some submodels are redundant AND {crop=}-> cropping fit!"
                    submodels = [
                        f
                        for (idx, f) in enumerate(fs)
                        if not is_redundant[idx]
                    ]

                    self.model = sum(submodels[1:], start=submodels[0])
                    self.fitModel.__wrapped__(
                        self,
                        update_flux=False,
                        bg_flux=bg_flux,
                        without_rejections=without_rejections,
                        without_absorption=without_absorption,
                        with_neighbours=with_neighbours,
                        fitter_kwargs=fitter_kwargs,
                    )
                else:
                    if any(is_redundant):
                        msg += f"some submodels are redundant BUT {crop=}"
                    else:
                        msg += "no submodels are redundant"
                    msg += " -> blacklisting the new model and stopping loop!"

                    continue_fitting = False
                    self.blacklist[new_model.pure_name] = True
                    if not aggressive:
                        self.model = self.fits[n_prev]
                        self.fit = self.fits[n_prev]

            logger.debug(msg)

        # Perform final model cropping
        if crop:
            self.cropFit.__wrapped__(
                self,
                bg_flux=bg_flux,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                with_neighbours=with_neighbours,
                limited=limited,
                v_sep=v_sep,
                measure=measure,
                reverse=reverse,
                fitter_kwargs=fitter_kwargs,
            )
        else:
            self.updateLinesEmission.__wrapped__(self, self.fit)

        # Update submodel names and order
        self.reformatFit()

        return self

    @validated_apply_info_to_method(subjects=("lines", "nonlinear"))
    def cropFit(
        self,
        *,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        limited: bool = False,
        v_sep: float | None = None,
        measure: str | None = None,
        reverse: bool | None = None,
        fitter_kwargs: FitterKwargs | None = None,
    ) -> bool:
        """
        ** PYDANTIC VALIDATED METHOD **

        Performs a single or multiple iterations of model cropping, i.e. identi-
        fies submodels with weak scores and evaluates whether those submodels
        are necessary. The algorithm works as follows:

        1.  All lines represented by a single submodel are identified and
            considered for cropping.
        2.  For each submodel (line), a quality measure is calculated. The
            quality measure serves to quantify the importance (or significance)
            of a specific submodel.
        3.  All submodels are ranked, and the weakest (worst quality) submodel
            is identified. A second model, identical to the initial model but
            excluding the weakest submodel, is created. The second model is then
            fitted.
        4.  If the first (advanced) model over-fits compared to the second
            (cropped) model, the submodel is justifiably removed, and the second
            model is accepted.
        5.  Steps 1-4 are repeated (recursively) until no submodel cropping is
            done.

        Parameters
        ----------
        measure : str
            The quality measure to use when quantifying the quality of a
            submodel. Options are 'getPeakSNR', 'getFluxSNR', 'getLineSNR', and
            'getWeightedAbsorption'. Default is {}.
        reverse : str
            Whether to reverse the order when ranking submodel qualities. If
            False, low-to-high quality score corresponds to bad-to-good. If
            True, high-to-low quality score corresponds to bad-to-good. Default
            is {}.

        Notes
        -----
        It does not crop line blends (multiple submodels representing a single
        emission line) as additional submodels were initially justified and
        added.
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        msg = "Cropping fit for {}: ".format(
            self.__str__(simple=True).removesuffix(".")
        )

        if self.fit is None:
            logger.debug(
                msg + "no 'fit' attribute found -> skipping cropping!"
            )
            return False

        if self.fit.n_submodels == 1:
            logger.debug(msg + "only a single line -> skipping cropping!")
            return True

        fit = self.fit
        fs = fit if fit.n_submodels > 1 else (fit,)
        n = fit.n_submodels
        fit_info = self.fit_infos[n]

        configuration = self.getConfiguration()

        single_lines: list[GaussianModel] = []
        multiple_lines: list[Union[GaussianModel, VProfileCopy]] = []
        for f in fs:
            if configuration[f.pure_name] == 1 and isinstance(
                f, GaussianModel
            ):
                single_lines.append(f)
            else:
                multiple_lines.append(f)

        msg += f"no. of single lines: {len(single_lines)}, no. of multiple lines: {n - len(single_lines)}. "

        if len(single_lines) == 0:
            logger.debug(msg + "No single lines -> skipping cropping!")
            return True
        elif (len(single_lines) == 1) and (n == 1):
            logger.debug(msg + "Only a single line -> skipping cropping!")
            return True
        else:
            single_lines = sorted(
                single_lines, 
                key=lambda f: getattr(f, measure)(self), 
                reverse=reverse,
            )

        removed_line = single_lines.pop(0)
        submodels = single_lines + multiple_lines
        self.model = sum(submodels[1:], start=submodels[0])

        success = self.fitModel.__wrapped__(
            self,
            update_flux=False,
            bg_flux=bg_flux,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            with_neighbours=with_neighbours,
            fitter_kwargs=fitter_kwargs,
        )
        if not success:
            msg += "Fitting the cropped model failed -> skipping cropping!"
            return False

        is_over, _ = self.isOverFitted.__wrapped__(
            self,
            self.fit.n_submodels,  # ? Cropped fit
            self.fit.n_submodels + 1,  # ? Previous fit
            bg_flux=bg_flux,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            with_neighbours=with_neighbours,
            line=removed_line.wave,
            limited=limited,
            v_sep=v_sep,
        )
        if is_over or False:  # ? 'False' is placeholder for VETO function
            # * Removing the single line had no significant impact
            # * on fit quality, i.e. using the more advanced model constitutes
            # * over-fitting.
            # * => Recursive call...
            self.cropped.add(removed_line.pure_name)
            return self.cropFit.__wrapped__(
                self,
                bg_flux=bg_flux,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                with_neighbours=with_neighbours,
                limited=limited,
                v_sep=v_sep,
                measure=measure,
                reverse=reverse,
                fitter_kwargs=fitter_kwargs,
            )
        else:
            # * Recursion ends -> update 'fit' and 'fit_info' attributes
            self.applyFit.__wrapped__(
                self,
                fit,
                fit_info=fit_info,
                update_emission=True,
            )
            return True

    def getModel(self) -> Optional[LineModel]:
        """
        Retrieves this 'LWindow's current fit/model if available.

        NOTES
        -----
        If 'thaw' is True, a copy of the model is retrieved. Fitting the
        retrieved model inplace will therefore NOT update the 'LWindow's model,
        and the fit will need to be applied using" 'ApplyFit'.
        """
        return self.fit or self.model
    
    @pydantic.validate_call
    def createVelocityProfileCopy(
        self,
        master: str,
        wave: float,
        mimic: str,
    ) -> VProfileCopy:
        """
        Create a copy of an emission line velocity profile. 

        Parameters
        ----------
        master : str
            The (pure) name of the emission line components to use to create 
            the velocity profile copy.
        wave : float
            The central wavelength of the velocity profile copy.
        mimic : str
            The name of the velocity profile copy. 
        """
        if master not in self.names:
            raise ValueError(f"Name '{master}' is not in {self.names}")


        model = self.getModel()
        ms = [
            m
            for m in ((model,) if model.n_submodels == 1 else model)
            if m.pure_name == master
        ]
        selected_model = sum(ms[1:], start=ms[0])

        return create_vprofilecopy(selected_model, mimic, wave=wave)

    @validated_apply_info_to_method(
        subjects=("lines",), specific_kwargs={"adapt_scale"}
    )
    def applyMyself(
        self,
        *,
        adapt_scale: bool | None = None,
    ) -> Self:
        """
        Apply this `Lwindow` instance's fit to other `LWindow` instances' 
        models.

        This method iterates over the `copies_to` attribute dictionary and 
        locates the `LWindow` instances containing the mimic lines.  

        Parameters
        ----------
        adapt_scale : bool

        Returns
        -------
        LWindow
            The current `LWindow` instance after applying the fit.

        Notes
        -----
        The `copies_to` dictionary has to element types: either a 2-tuple of 
        strings or a tuple of a string and None. The latter case indicates that 
        the corresponding line mimics a line covered by the same `LWindow` 
        instance.
        """
        for master, children in self.copies_to.items():
            for idx_d, mimic in filter(
                lambda item: item[0] is not None,
                children,
            ):
                lwindow: LWindow = self.spectrum.em[idx_d]

                # Check if master line has been cropped
                if master in self.cropped:
                    # Crop corresponding mimic line
                    lwindow.cropped.add(mimic)
                    continue

                vprofilecopy = self.createVelocityProfileCopy(
                    master,
                    lwindow.waves[mimic],
                    mimic,
                )
                lwindow.applyVelocityProfileCopy(
                    vprofilecopy,
                    adapt_scale=adapt_scale,
                )

                submimics: list[str] = [
                    submimic
                    for idx, submimic in lwindow.copies_to.get(mimic, [])
                    if idx is None
                ]
                for submimic in submimics:
                    # Create velocity profile copy of the 'master' velocity 
                    # profile copy
                    subcopy = create_vprofilecopy(
                        vprofilecopy,
                        submimic,
                        wave=lwindow.waves[submimic],
                        strength_scale_value=lwindow.scale_init[submimic],
                        strength_scale_bounds=lwindow.scale_bounds[submimic],
                        strength_scale_fixed=lwindow.scale_fixed[submimic],
                    )
                    lwindow.applyVelocityProfileCopy(
                        subcopy,
                        adapt_scale=adapt_scale,
                    )

        return self

    @validated_apply_info_to_method(
        subjects=("lines",), specific_kwargs={"adapt_scale"}
    )
    def applyVelocityProfileCopy(
        self,
        model: VProfileCopy,
        *,
        adapt_scale: bool | None = None,
    ) -> Self:
        """
        Apply a velocity profile copy to the current `LWindow` instance and 
        update the `strength_scale` parameter. 
        
        If `adapt_scale` is True and the `strength_scale` parameter is not 
        fixed, the strength scale will be adapted s.t. the integrated flux of 
        the velocity profile copy matches that of the replaced Gaussian 
        functions.  
        """
        mimic = model.pure_name
        assert mimic in self.names
        assert self.model is not None

        ms: Iterable[LineModel] = (
            self.model 
            if (self.model.n_submodels > 1) 
            else [self.model]
        )
        replaced_ms = [m for m in ms if m.pure_name == model.pure_name]
        other_ms = [m for m in ms if m.pure_name != model.pure_name]

        scale = model.strength_scale
        scale.value = _apply_bounds(
            self.scale_init[model.pure_name],
            self.scale_bounds[model.pure_name],
        )
        scale.bounds = self.scale_bounds[model.pure_name]
        scale.fixed = self.scale_fixed[model.pure_name] \
            or (scale.bounds[0] == scale.bounds[1])

        if adapt_scale and not model.strength_scale.fixed:
            model.adaptStrengthScale(replaced_ms)

        new_ms = sorted(
            sum(other_ms, start=model),
            key=lambda m: m.sorting_key,
        )
        self.model = sum(new_ms[1:], start=new_ms[0])

        # Remove current fit if existant
        self.fit = None

        # Reset blacklist and previous fits
        self._blacklist[model.pure_name] = True
        self.n_maxs[model.pure_name] = 1

        self.blacklist = self._blacklist.copy()
        self.fits.clear()
        self.fit_infos.clear()

        return self

    @pydantic.validate_call
    def adoptFit(
        self,
        fit: LineModel,
        *,
        fit_info: OptimizeResultType | None = None,
        update_emission: bool = False,
    ) -> Self:
        self.fits.clear()
        self.fit_infos.clear()

        n = fit.n_submodels
        self.fits[n] = self.fit = self.model = fit
        self.fit_infos[n] = self.fit_info = fit_info

        # Update parameter bounds
        for f in (fit,) if fit.n_submodels == 1 else fit:
            pure_name = f.pure_name

            if isinstance(f, GaussianModel):
                self.strength_bounds[pure_name] = f.strength.bounds
                self.fwhm_v_bounds[pure_name] = f.fwhm_v.bounds
                self.v_off_bounds[pure_name] = f.v_off.bounds

            if isinstance(f, _VProfileCopy):
                if pure_name not in self.is_copy_of:
                    msg = f"Found velocity profile copy, '{pure_name}', with no corresponding entry in 'is_copy_of' dict: {self.is_copy_of}"
                    raise ValueError(msg)

                self.scale_init[pure_name] = f.strength_scale.value
                self.scale_bounds[pure_name] = f.strength_scale.bounds
                self.scale_fixed[pure_name] = f.strength_scale.fixed

        # Update n_max values
        self.n_maxs.update(self.getConfiguration())

        # Update blacklist
        for name in self.names:
            self.blacklist[name] = self._blacklist[name] = True

        if update_emission:
            self.updateLinesEmission.__wrapped__(self, fit)

    @pydantic.validate_call
    def applyFit(
        self,
        fit: LineModel,
        *,
        fit_info: OptimizeResultType | None = None,
        update_emission: bool = False,
    ) -> Self:
        """
        Applies the 'fit' and 'fit_info' values to this 'LWindow' class.
        """
        n = fit.n_submodels
        self.fits[n] = self.fit = fit
        self.fit_infos[n] = self.fit_info = fit_info

        if update_emission:
            self.updateLinesEmission.__wrapped__(self, fit)

        return self

    @pydantic.validate_call
    def updateLinesEmission(
        self,
        model: Union[LineModel, None],
    ) -> Self:
        """
        Updates the '_y_em' attribute to account for a new emission line model,
        'model'. If no previous model is found, '_y_em' is updated directly. If
        a previous model is found, this model's '_y_em' contribution is first
        removed, and the new 'model's contribution is added.

        If 'model' is not given, i.e. None, the previous model's contribution
        is removed.
        """
        # Subtract previous contribution
        self._y_em[:] -= self._y_em_contrib

        # Calculate new contribution
        self._y_em_contrib[:] = 0.0
        if model is not None:
            eval_mask = np.isfinite(self._x)
            x = self._x[eval_mask]

            preparer = PrepareModel(
                x=self._x,
                model=model,
                eval_mask=eval_mask,
                comp_mask=eval_mask,
                kernels=self._kernels,
                is_logbinned=True,
            )
            with preparer as (seq_model, _):
                f = seq_model(x)

            # Store the new emission line contribution
            self._y_em_contrib[eval_mask] = f
            # Add current contribution
            self._y_em[eval_mask] += f

        return self

    def removeLine(
        self,
        name: str,
    ) -> Self:
        if name not in self.names:
            raise ValueError(f"line '{name}' not in 'self.names'!")

        self.names.remove(name)
        self.linetypes.pop(name)
        self.complexes.pop(name)

        del self.waves[name]
        del self.n_maxs[name]

        del self.strength_bounds[name]
        del self.fwhm_v_bounds[name]
        del self.v_off_bounds[name]

        if name in self.i_bounds:
            del self.i_bounds[name]
        if name in self.blacklist:
            del self.blacklist[name]
        if name in self._blacklist:
            del self._blacklist[name]

        return self

    def sortWaves(self) -> list[tuple[str, float]]:
        def sorting_key(item: tuple[str, float]) -> tuple[float, float]:
            return (
                0.0 if item[0] in self.is_multiplet_of.values() else 1.0,
                item[1],
            )

        return sorted(self.waves.items(), key=sorting_key)

    def getConfiguration(
        self,
        *,
        model: Union[LineModel, None] = None,
    ) -> dict[str, int]:
        """
        Counts the number of submodels with the same `pure_name` attribute in
        the output of `self.getModel()`.
        """
        if model is None:
            model = self.getModel()

        if model is None:
            return {}

        ms = (model,) if model.n_submodels == 1 else model
        return Counter(m.pure_name for m in ms)

    def updateBlacklistFromConfiguration(self) -> Self:
        configuration = self.getConfiguration()

        for name in filter(lambda name: not self.blacklist[name], self.names):
            self.blacklist[name] = self.n_maxs[name] == configuration[name]

        return self

    def reformatFit(self) -> Self:
        if self.model is not None:
            rename_gaussians(self.model)
        if self.fit is not None:
            rename_gaussians(self.fit)
        return self

    @validated_apply_info_to_method(
        subjects=("lines",), specific_kwargs={"v_sep"}
    )
    def isUnderFitted(
        self,
        n: int,
        *,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        line: str | float | None = None,
        complex_name: str | None = None,
        limited: bool = False,
        v_sep: float | None = None,
    ) -> tuple[bool, FloatVector]:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        from ..ml import Under
        if bg_flux is None:
            bg_flux = self.default_bg

        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            original=False,
            mode=None,
            covered=True,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            with_neighbours=with_neighbours,
            valid=True,
            log_valid=False,
            bg_flux=bg_flux,
            line=line,
            complex_name=complex_name,
            limited=limited,
            v_sep=v_sep,
            shrink=None,
        )
        under = Under(
            masked_coords.x,
            masked_coords.y,
            masked_coords.dy,
            fit=self.fits[n],
            snr=self.info.lines.snr,  # ! Changing snr has not effect on model!
        )
        return under(), under.getFeatures(as_dict=True)

    @validated_apply_info_to_method(
        subjects=("lines",), specific_kwargs={"v_sep"}
    )
    def isOverFitted(
        self,
        n: int,
        m: int,
        *,
        bg_flux: BackgroundFlux | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        with_neighbours: bool = False,
        line: str | float | None = None,
        complex_name: str | None = None,
        limited: bool = False,
        v_sep: float | None = None,
    ) -> tuple[bool, FloatVector]:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        from ..ml import Over

        if bg_flux is None:
            bg_flux = self.default_bg

        assert n < m

        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            original=False,
            mode=None,
            covered=True,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            with_neighbours=with_neighbours,
            valid=True,
            log_valid=False,
            bg_flux=bg_flux,
            line=line,
            complex_name=complex_name,
            limited=limited,
            v_sep=v_sep,
            shrink=None,
        )
        over = Over(
            masked_coords.x,
            masked_coords.y,
            masked_coords.dy,
            fit_initial=self.fits[n],
            fit_final=self.fits[m],
            snr=self.info.lines.snr,  # ! Changing snr has not effect on model!
        )
        return over(), over.getFeatures(as_dict=True)

    ### Error estimation

    @validated_apply_info_to_method(subjects=("error", "nonlinear", "random"))
    def instantiateBootstrapper(
        self,
        *,
        pl: Optional[PowerLawModel] = None,
        fe: Optional[IronModel] = None,
        ba: Optional[BalmerModel] = None,
        hg: Optional[HostGalaxyModel] = None,
        model_types: ModelTypes | None = None,
        without_rejections: bool = False,
        without_absorption: bool = False,
        pool: PoolType | None = None,
        scale: Scale | None = None,
        variant: Variant | None = None,
        bootstrap_type: BootstrapType | None = None,
        n_sigmas: float | None = None,
        vary_lines: VaryLines | None = None,
        cfit: ContinuumFitResult | None = None,
        iterations: int | None = None,
        replace_missing: bool | None = None,
        tqdm_disable: bool | None = None,
        tqdm_leave: bool | None = None,
        fitter_kwargs: FitterKwargs | None = None,
        random_state: RandomStateType | None = None,
    ):
        from quasar_errors.spectrum_utils.format_bootstrapping_kwargs_for_lwindow import (
            format_bootstrapping_kwargs_for_lwindow,
        )
        assert scale == "local" or scale == "semilocal"

        if self.bootstrapper is not None:
            logger.debug("Existing 'bootstrapper' will be overwritten.")

        cls, args, kwargs = format_bootstrapping_kwargs_for_lwindow(
            self,
            fitter_kwargs=fitter_kwargs,
            pl=pl,
            fe=fe,
            ba=ba,
            hg=hg,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            with_neighbours=(scale == "semilocal"),
            variant=variant,
            model_types=model_types,
            bootstrap_type=bootstrap_type,
            n_sigmas=n_sigmas,
            vary_lines=vary_lines,
            pool=pool,
            cfit=cfit,
            iterations=iterations,
            random_state=random_state,
            replace_missing=replace_missing,
            tqdm_disable=tqdm_disable,
            tqdm_leave=tqdm_leave,
            logger=logger,
        )
        return cls(*args, **kwargs)
    

    def componentsplot(
        self,
        pattern: str | Pattern[str],
        dv: float,
        mode: Literal["simple", "difference", "residual"],
        model: Union[LineModel, None] = None,
        lazy: bool = False,
        **plotter_kwargs,
    ):
        from quasar_decrypt.plotting.diagnostics import (
            EmissionLineComponentsWithDifferences,
            EmissionLineComponentsWithResiduals,
            SimpleEmissionLineComponents,
        )
        if model is None:
            model = self.getModel()
        
        args = (self, model, pattern, dv)
        if mode == "simple":
            plotter_cls = SimpleEmissionLineComponents
        elif mode == "difference":
            plotter_cls = EmissionLineComponentsWithDifferences
        elif mode == "residual":
            plotter_cls = EmissionLineComponentsWithResiduals
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return self._plot_helper(
            plotter_cls=plotter_cls,
            lazy=lazy,
            args=args,
            plotter_kwargs=plotter_kwargs,
        )
