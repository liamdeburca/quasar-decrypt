__all__ = ["LWindow"]

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from logging import getLogger
from typing import ClassVar, Literal, Optional, Self, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import (
    argmax,
    convolve,
    empty,
    exp,
    float64,
    fromiter,
    interp,
    invert,
    isfinite,
    linspace,
    zeros_like,
)
from quasar_errors.bootstrapping import BaseBootstrapper
from quasar_errors.error_result import ErrorResult
from quasar_errors.model_samples import GaussianSampleList
from quasar_errors.spectrum_utils.format_bootstrapping_kwargs_for_lwindow import (
    format_bootstrapping_kwargs_for_lwindow,
)
from quasar_models import (
    BalmerModel,
    GaussianModel,
    HostGalaxyModel,
    IronModel,
    PowerLawModel,
)
from quasar_models.line import VProfileCopy1G, VProfileCopyDict, _VProfileCopy
from quasar_models.modeling import Fitter, PrepareModel
from quasar_models.utils.astropy import apply_bounds, order_submodels
from quasar_plotting import absorptionplot, fitplot, quickplot
from quasar_plotting.colors import DEFAULT_COLORS
from quasar_plotting.utils import get_coords
from quasar_typing.astropy import CompoundModel_, FitInfo
from quasar_typing.bounds import AstropyBounds
from quasar_typing.misc import (
    BackgroundFlux,
    BootstrapType,
    ModelTypes,
    Scale,
    Variant,
    VaryLines,
)
from quasar_typing.misc.pool import Pool_
from quasar_typing.numpy import BoolVector, FloatVector, RandomState_
from quasar_utils.continuum_fit_result import ContinuumFitResult
from quasar_utils.decorators import (
    validate_call,
    validated_apply_info_to_method,
)
from quasar_utils.setup import FitterKwargs
from scipy.ndimage import binary_dilation

from ..ml import Over, Under
from ..utils import (
    ContiguousMaskedCoords,
    MaskedCoords,
    ReadOnlyMaskedCoords,
    _SpecWindow,
    get_bounds_indices,
    stopwatch,
)
from .utils import common_middle, update_v_off_bounds

logger = getLogger(__name__)

type LineModel = Union[GaussianModel, CompoundModel_[GaussianModel]]


@dataclass(kw_only=True)
class LWindow(_SpecWindow):
    names: set[str] = field(default_factory=set, kw_only=True)

    complexes: dict[str, str] = field(default_factory=dict, kw_only=True)

    lines: dict[str, float] = field(default_factory=dict, kw_only=True)
    n_maxs: dict[str, int] = field(default_factory=dict, kw_only=True)
    needs_line: dict[str, str] = field(default_factory=dict, kw_only=True)

    strength_bounds: dict[str, tuple] = field(
        default_factory=dict, kw_only=True
    )
    fwhm_v_bounds: dict[str, tuple] = field(default_factory=dict, kw_only=True)
    v_off_bounds: dict[str, tuple] = field(default_factory=dict, kw_only=True)

    is_copy_of: dict[str, str] = field(default_factory=dict, kw_only=True)
    is_multiplet_of: dict[str, str] = field(default_factory=dict, kw_only=True)
    scale_init: dict[str, float] = field(default_factory=dict, kw_only=True)
    scale_bounds: dict[str, tuple] = field(default_factory=dict, kw_only=True)
    scale_fixed: dict[str, bool] = field(default_factory=dict, kw_only=True)

    copies_to: dict[str, list[tuple[int | None, str]]] = field(
        default_factory=lambda: defaultdict(list), kw_only=True
    )
    i_bounds: dict[str, tuple[float, float]] = field(
        default_factory=dict, kw_only=True
    )
    blacklist: dict[str, bool] = field(default_factory=dict, kw_only=True)
    _blacklist: dict[str, bool] = field(default_factory=dict, kw_only=True)

    neighbours: tuple[_SpecWindow | None, _SpecWindow | None] = field(
        default=(None, None), kw_only=True
    )
    # prev_model: LineModel | None = field(default=None, kw_only=True)
    model: LineModel | None = field(default=None, kw_only=True)
    fit: LineModel | None = field(default=None, kw_only=True)
    fit_info: FitInfo | None = field(default=None, kw_only=True)

    fits: dict[int, LineModel] = field(default_factory=dict, kw_only=True)
    fit_infos: dict[int, FitInfo] = field(default_factory=dict, kw_only=True)

    cropped: set[str] = field(default_factory=set, kw_only=True)

    bootstrapper: BaseBootstrapper | None = field(default=None, init=False)
    error_result: ErrorResult | None = field(default=None, init=False)

    _y_em_contrib: FloatVector | None = field(default=None, init=False)

    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({"all", "em"})

    def __post_init__(self):
        if self._y_em_contrib is None:
            self._y_em_contrib = zeros_like(self._x, dtype=float64)

    @property
    def sample(self) -> GaussianSampleList | None:
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

        assert len(self.lines) > 0

        logger.debug(f"Starting pipeline for {self.__str__(True)}")

        with stopwatch() as watch:
            try:
                self.prepareLines.__wrapped__(
                    self,
                    v_sep=v_sep,
                    min_fittable_total=min_fittable_total,
                    min_fittable_ratio=min_fittable_ratio,
                )
            except Exception as e:
                msg = f"Failed pipeline during `prepareLines` due to: {e}"
                logger.warning(msg)
                return False

            if with_neighbours:
                success = self.prepareNeighbours.__wrapped__(
                    self,
                    sigma_res=sigma_res,
                )
                if not success:
                    logger.warning(
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
                logger.warning(
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
                logger.warning(">>> Failed pipeline during 'makeInitialFit'!")
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
                logger.warning(">>> Failed pipeline during 'makeFinalFit'!")
                return False

        logger.debug(f">>> Finished entire pipeline in {1e3 * watch.elapsed:.1f} ms.")
        return True

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
        line: str | float | None = None,
        complex_name: str | None = None,
        limited: bool = True,
        v_sep: float | None = None,
    ) -> BoolVector:
        mask = super().getMask.__wrapped__(
            self,
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=valid,
            log_valid=log_valid,
        )
        if with_neighbours and self.neighbours != (None, None):
            for cwindow in filter(lambda w: w is not None, self.neighbours):
                mask |= cwindow.__wrapped__.getMask(
                    cwindow,
                    covered=covered,
                    without_rejections=without_rejections,
                    without_absorption=without_absorption,
                    valid=valid,
                    log_valid=log_valid,
                )

        if complex_name is not None:
            bounds = self.i_bounds[complex_name]

        elif line is not None:
            if isinstance(line, str):
                if limited:
                    # Based on 'i_bounds' -> find complex name
                    bounds = self.i_bounds[self.complexes[line]]
                else:
                    _line = self.lines[line]
                    bounds = (_line * (1 - v_sep), _line * (1 + v_sep))
            else:
                bounds = (line * (1 - v_sep), line * (1 + v_sep))

        if "bounds" in locals():
            idx_left, idx_right = get_bounds_indices(self._x, bounds)

            lmask = mask.copy()
            lmask[:] = False
            lmask[idx_left : idx_right + 1] = True

            mask &= lmask

        return mask

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
        line: float | str | None = None,
        complex_name: str | None = None,
        limited: bool = True,
        v_sep: float | None = None,
    ) -> MaskedCoords | ReadOnlyMaskedCoords | ContiguousMaskedCoords:
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
            complex_name=complex_name,
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

    @validated_apply_info_to_method(subjects=("lines",))
    def add(
        self,
        name: str,
        complex_name: str,
        line: float,
        n_max: int,
        needs_line: str | None = None,
        is_copy_of: str | None = None,
        *,
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

        Parameters
        ----------
        line : float
            Unitless rest wavelength of the emission line.
        v_off : float
            Maximum absolute velocity offset of the emission line in units of
            the speed of light.
        n_max : int
            Maximum allowed number of profile functions to model the emission
            line.
        name : str
            Name of the emission line model(s).
        """
        if (
            self.x_bounds[0] < line < self.x_bounds[1]
        ) or force_add:
            self.names.add(name)
            self.complexes[name] = complex_name

            self.lines[name] = line
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
        new_model: GaussianModel,
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
            new_model.pure_name,
            new_model.pure_name,
            new_model.wave,
            1,
            None,
            None,
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
            logger.warning(msg)
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
        # STEP 1

        self.i_bounds.clear()
        self.blacklist.clear()
        self._blacklist.clear()

        if len(self.lines) == 0:
            # No lines to consider.
            return self

        elif len(self.lines) == 1:
            # Only one line to consider.
            name = next(iter(self.names))

            line = self.lines[name]
            (lower, upper) = self.v_off_bounds[name]

            self.x_bounds = (
                max(self.x_bounds[0], line * (1 - v_sep)),
                min(self.x_bounds[1], line * (1 + v_sep)),
            )

            self.i_bounds[self.complexes[name]] = self.x_bounds
            self.v_off_bounds[name] = (
                max(lower, self.x_bounds[0] / line - 1),
                min(upper, self.x_bounds[1] / line - 1),
            )

        else:
            # Create dictionary of complexes' bluest and reddest lines

            complex_to_lines: dict[str, set[str]] = defaultdict(set)
            for name, complex_name in self.complexes.items():
                complex_to_lines[complex_name].add(name)

            complex_bounds: dict[str, list[float]] = {}
            for complex_name, names in complex_to_lines.items():
                lines = sorted(self.lines[name] for name in names)
                complex_bounds[complex_name] = [lines[0], lines[-1]]

            del complex_to_lines

            # Validate that complexes do not overlap
            complex_bounds = dict(
                sorted(
                    complex_bounds.items(),
                    key=lambda item: item[1][0],
                )
            )
            complex_lbs = fromiter(
                (val[0] for val in complex_bounds.values()),
                float64,
            )
            complex_ubs = fromiter(
                (val[1] for val in complex_bounds.values()),
                float64,
            )
            if not (complex_lbs[1:] >= complex_ubs[:-1]).all():
                msg = f"Complexes overlap: {complex_bounds=}"
                logger.critical(msg)
                raise ValueError(msg)

            n_complexes = len(complex_bounds)
            _i_bounds = empty(n_complexes + 1)

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

            self.i_bounds = dict(zip(complex_bounds.keys(), pairwise(_i_bounds)))
            self.x_bounds = (_i_bounds[0], _i_bounds[-1])

            # Update velocity offset bounds

            narrow_lines: dict[str, float] = {}
            broad_lines: dict[str, float] = {}
            for name, line in self.lines.items():
                d = narrow_lines if name.startswith("n") else broad_lines
                d[name] = line

            # Narrow
            if narrow_lines:
                update_v_off_bounds(
                    v_off_bounds=self.v_off_bounds,
                    lines=narrow_lines,
                    x_bounds=self.x_bounds,
                )
            # Broad
            if broad_lines:
                update_v_off_bounds(
                    v_off_bounds=self.v_off_bounds,
                    lines=broad_lines,
                    x_bounds=self.x_bounds,
                )

        # Check multiplets
        self.is_multiplet_of.clear()
        self.copies_to.clear()
        for mimic, master in filter(
            lambda item: item[1] in self.names,
            self.is_copy_of.items(),
        ):
            self.is_multiplet_of[mimic] = master
            self.copies_to[master].append((None, mimic))

        # STEP 2
        for name in self.names:
            # Blacklist lines which are:
            # - Limited to a single Gaussian, or
            # - Copies of other lines.
            self.blacklist[name] = self._blacklist[name] = any(
                [
                    self.n_maxs[name] == 1,
                    name in self.is_copy_of,
                    name in self.is_multiplet_of,
                ]
            )

        if grade_lines:
            _ = self.gradeLines.__wrapped__(
                self,
                with_neighbours=True,
                min_fittable_total=min_fittable_total,
                min_fittable_ratio=min_fittable_ratio,
                v_sep=v_sep,
            )

        return self

    @validated_apply_info_to_method(
        subjects=("loading",), specific_kwargs={"sigma_res"}
    )
    def prepareNeighbours(
        self,
        *,
        sigma_res: float | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if self.spectrum is None:
            msg = "LWindow must have a `spectrum` attribute to prepare \
                neighbours!"
            logger.critical(msg)
            raise ValueError(msg)
        elif self.spectrum.pl is None:
            msg = "LWindow's `spectrum` must have a `continuum_windows` \
                attribute to prepare neighbours!"
            logger.critical(msg)
            raise ValueError(msg)

        if isinstance(sigma_res, float):
            iterations = int(sigma_res // self.info.loading["sigma_res"])
        else:
            iterations = sigma_res

        vicinity = binary_dilation(self.mask, iterations=iterations)

        left = None
        right = None
        for cwindow in self.spectrum.continuum_windows:
            if not (cwindow.mask & vicinity).any():
                continue

            if cwindow.x_bounds[1] < self.x_bounds[0]:
                left = cwindow
            elif self.x_bounds[1] < cwindow.x_bounds[0]:
                right = cwindow

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

        for name, line in self.sortLines():
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
            f = (valid_pixels & invert(absorbed_pixels)).sum()
            a = (valid_pixels & absorbed_pixels).sum()
            i = invert(valid_pixels).sum()

            msg = (
                ">>> [{:.1f} <-> {:.1f}] w/ {}/{n} (fit.),"
                "{}/{n} (abs.), {}/{n} (inv.): ".format(
                    *self.i_bounds[complex_name], f, a, i, n=n
                )
            )

            if f == 0:
                msg += f"No valid data -> removing line '{name}' at {line:.1f}!"
                lines_to_remove.add(name)
            elif (f / n < min_fittable_ratio) or (f < min_fittable_total):
                msg += f"Not enough valid data -> blacklisting line '{name}' at {line:.1f}!"
                self.blacklist[name] = True
            else:
                msg += f"Enough valid data for line '{name}' at {line:.1f}!"

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
        for name, line in self.sortLines():
            if name in self.cropped:
                continue

            if name in self.is_multiplet_of:
                models[name] = VProfileCopy1G.from_model(
                    line,
                    name,
                    models[self.is_multiplet_of[name]],
                    strength_scale_value=self.scale_init[name],
                    strength_scale_bounds=self.scale_bounds[name],
                    strength_scale_fixed=self.scale_fixed[name],
                    freeze=False,
                )
                continue

            masked_coords = self.getMaskedCoords.__wrapped__(
                self,
                mode="c",
                covered=True,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                with_neighbours=with_neighbours,
                valid=True,
                log_valid=False,
                bg_flux=bg_flux,
                complex_name=self.complexes[
                    name
                ],  # Integrate over the complex, not the individual line
                limited=True,
                v_sep=v_sep,
            )
            try:
                models[name] = GaussianModel.instantiate(
                    line,
                    masked_coords.x,
                    masked_coords.y,
                    masked_coords.y_smooth,
                    name=name,
                    strength_bounds=self.strength_bounds[name],
                    v_off_bounds=self.v_off_bounds[name],
                    fwhm_v_bounds=self.fwhm_v_bounds[name],
                    sigma_res=self.info.loading["sigma_res"],
                    logger=logger,
                )
                msg = f">>> Successfully instantiated line '{name}' at {line:.1f}."
                logger.debug(msg)
            except Exception as e:
                msg = f">>> Failed instantiating line '{name}' at {line:.1f} due to: {e}"
                logger.warning(msg)
                continue

        if len(models) == 0:
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
            mode="c",
            covered=True,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            with_neighbours=with_neighbours,
            valid=True,
            log_valid=False,
            bg_flux=bg_flux,
        )
        msg += f"no. of Gaussians: {self.model.n_submodels}, "
        msg += f"no. of data points: {masked_coords.size}. "

        with (
            stopwatch() as watch,
            PrepareModel(x=masked_coords.x, model=self.model),
        ):
            fitter = Fitter()
            fit = fitter(
                self.model,
                masked_coords.x,
                masked_coords.y,
                dy=masked_coords.dy,
                get_model=True,
                inplace=False,
                **fitter_kwargs,
            )

        msg += f"Finished fitting in {1e3 * watch.elapsed:.1f} ms."
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
                mode="c",
                covered=True,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                with_neighbours=with_neighbours,
                valid=True,
                log_valid=False,
                bg_flux=bg_flux,
            )

            fs = (self.fit,) if self.fit.n_submodels == 1 else self.fit
            for f in filter(
                lambda f: not self.blacklist[f.pure_name],
                fs,
            ):
                # Interpolate noise level and compare with peak flux density
                crit_val = evaluate_initial * interp(
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
            mode="c",
            covered=True,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            with_neighbours=with_neighbours,
            valid=True,
            log_valid=False,
            bg_flux=bg_flux,
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
            valid_lines = list(
                filter(
                    lambda f: not self.blacklist[f.pure_name],
                    fs,
                )
            )
            waves = [f.wave for f in valid_lines]

            z = abs(
                (masked_coords.y - self.fit(masked_coords.x))
                / masked_coords.dy
            )
            z_convolved = convolve(
                z,
                exp(-(linspace(-3, 3, w) ** 2)),
                mode="valid",
            )
            z_interp = interp(waves, masked_coords.x[h:-h], z_convolved)

            submodel: GaussianModel = valid_lines[
                argmax(z_interp).flatten()[0]
            ]

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
            self.makeInitialFit.__wrapped__(
                self,
                bg_flux=bg_flux,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                with_neighbours=with_neighbours,
                evaluate_initial=evaluate_initial,
                fitter_kwargs=fitter_kwargs,
            )

        continue_fitting: bool = True

        # Check if any lines aren't already blacklisted
        if all(self.blacklist.values()):
            # No additional models necessary
            continue_fitting = False

        elif self.fit.n_submodels <= 2:
            # Check if under-fitted
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

        while continue_fitting:
            self.updateBlacklistFromConfiguration()
            if all(self.blacklist.values()):
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

            self.model = self.fit + new_model
            if check_multiplet:
                # The new model is a master model in a multiplet. Add a Gaussian
                # component to each mimic line.
                for mimic in (
                    mimic
                    for window_idx, mimic in self.copies_to[
                        new_model.pure_name
                    ]
                    if window_idx is None
                ):
                    mimic_model = VProfileCopy1G.from_model(
                        self.lines[mimic],
                        mimic,
                        new_model,
                        strength_scale_value=self.scale_init[mimic],
                        strength_scale_bounds=self.scale_bounds[mimic],
                        strength_scale_fixed=self.scale_fixed[mimic],
                        adapt=True,
                        freeze=False,
                    )

                    self.model += mimic_model
                    current_config[mimic] += 1

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
                self.fit.n_submodels - 1,
                self.fit.n_submodels,
                bg_flux=bg_flux,
                without_rejections=without_rejections,
                without_absorption=without_absorption,
                with_neighbours=with_neighbours,
                line=new_model.wave,
                limited=limited,
                v_sep=v_sep,
            )
            if is_over or False:  # ? 'False' is placeholder for VETO function
                # Check whether models are saturated (blacklisted) or touching
                # their respective bounds.
                fs = self.fit if (self.fit.n_submodels > 1) else [self.fit]
                is_redundant: list[bool] = [
                    f.isTouchingBounds() & self.blacklist[f.pure_name]
                    if isinstance(f, GaussianModel)
                    else False
                    for f in fs
                ]

                if all(is_redundant):
                    continue_fitting = False

                elif any(is_redundant) and crop:
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
                    continue_fitting = False
                    self.blacklist[new_model.pure_name] = True

                    if not aggressive:
                        self.fit = self.fits[self.fit.n_submodels - 1]

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
        multiple_lines: list[GaussianModel | _VProfileCopy] = []
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

    @validate_call
    def getModel(self, thaw: bool = False) -> Optional[CompoundModel_]:
        """
        Retrieves this 'LWindow's current fit/model if available.

        NOTES
        -----
        If 'thaw' is True, a copy of the model is retrieved. Fitting the
        retrieved model inplace will therefore NOT update the 'LWindow's model,
        and the fit will need to be applied using" 'ApplyFit'.
        """
        fit = self.fit or self.model
        if fit is None:
            return None

        if thaw:
            fit = fit.copy()
            fs = (fit,) if fit.n_submodels == 1 else fit

            def filter_func(f):
                # True if submodel is a velocity profile copy, but not a part of
                # a multiplet.
                return (
                    isinstance(f, _VProfileCopy)
                    and f.pure_name not in self.is_multiplet_of
                )

            for f in filter(filter_func, fs):
                f._thaw_velocity_profile(inplace=True)
                f._remember_ties(inplace=True)

        return fit

    @validate_call
    def createVelocityProfileCopy(
        self,
        master: str,
        wave: float,
        mimic: str,
        freeze: bool = False,
        model_kwargs: dict | None = None,
    ) -> _VProfileCopy:
        """
        ** PYDANTIC VALIDATED METHOD **

        Creates a copy of an emission lines velocity profile, with the centre
        'wave' (float) and gives it the name 'name' (str). All submodels with
        a 'pure_name' equal to 'master' are used for the velocity profile.
        """
        if master not in self.names:
            msg = f"Line '{master}' not found in {self.__str__(simple=True).removesuffix('.')}!  Available names are: {self.names}"
            raise ValueError(msg)
        
        model = self.getModel.__wrapped__(self, thaw=False)
        ms = (model,) if model.n_submodels == 1 else model
        names = {m.pure_name for m in ms}

        if master not in names:
            raise ValueError(
                f"Line '{master}' not found in the current model of "
                f"{self.__str__(simple=True).removesuffix('.')}! Available "
                f"lines are: {names}"
            )

        submodels = [m for m in ms if m.pure_name == master]

        return VProfileCopyDict[len(submodels)].from_model(
            wave,
            mimic,
            sum(submodels[1:], start=submodels[0]),
            freeze=freeze,
            **(model_kwargs or {}),
        )

    @validated_apply_info_to_method(
        subjects=("lines",), specific_kwargs={"adapt_scale"}
    )
    def applyMyself(
        self,
        line_windows: list[_SpecWindow] | None = None,
        *,
        adapt_scale: bool | None = None,
    ) -> Self:
        """
        Applies this 'LWindow's velocity profile copies to other 'LWindow's.

        NOTES
        -----
        When creating the velocity-profile copy, the keyword argument 'freeze'
        is set to True.

        This method does not use Pydantic type validation due to the use of:
        'list[Self]' in the type annotation of 'line_windows'.
        """
        if line_windows is None:
            assert self.spectrum is not None
            line_windows = self.spectrum.em

        for master, children in self.copies_to.items():
            for idx_d, mimic in filter(
                lambda item: item[0] is not None,
                children,
            ):
                lwindow = line_windows[idx_d]

                # Check if master line has been cropped
                if master in self.cropped:
                    # Crop corresponding mimic line
                    lwindow.cropped.add(mimic)
                    continue

                lwindow.applyVelocityProfileCopy.__wrapped__(
                    lwindow,
                    self.createVelocityProfileCopy(
                        master,
                        lwindow.lines[mimic],
                        mimic,
                        freeze=True,
                    ),
                    adapt_scale=adapt_scale,
                )

        return self

    @validated_apply_info_to_method(
        subjects=("lines",), specific_kwargs={"adapt_scale"}
    )
    def applyVelocityProfileCopy(
        self,
        master_model: _VProfileCopy,
        *,
        adapt_scale: bool | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **

        Applies a velocity profile copy, 'master_model', to this 'LWindow'. This
        velocity profile copy replaces the existant submodels with same
        'pure_name' attribute.

        Successfully applying the velocity profile copy resets 'blacklist',
        'fits' and 'fit_infos' attributes, and sets the default blacklist
        value of the velocity profile to 'True' and 'n_max' to 1, i.e. no
        additional Gaussians will be placed on top of the velocity profile copy.
        """
        mimic = master_model.pure_name
        assert mimic in self.names
        assert self.model is not None

        ms = self.model if (self.model.n_submodels > 1) else [self.model]
        replaced_ms = list(
            filter(
                lambda m: m.pure_name == mimic,
                ms,
            )
        )
        other_ms = list(
            filter(
                lambda m: m.pure_name != mimic,
                ms,
            )
        )
        val = apply_bounds(
            self.scale_init[mimic],
            bounds := self.scale_bounds[mimic],
        )
        master_model.strength_scale.value = val
        master_model.strength_scale.bounds = bounds
        master_model.strength_scale.fixed = bool(val == bounds[0] == bounds[1])

        if adapt_scale and not master_model.strength_scale.fixed:
            master_model.adaptStrengthScale(replaced_ms)

        new_ms = sorted(
            other_ms + [master_model],
            key=lambda m: m.sorting_key,
        )

        self.model = (
            new_ms[0]
            if (len(new_ms) == 1)
            else sum(new_ms[1:], start=new_ms[0])
        )

        # Remove current fit if existant
        self.fit = None

        # Reset blacklist and previous fits
        self._blacklist[master_model.pure_name] = True
        self.n_maxs[master_model.pure_name] = 1

        self.blacklist = self._blacklist.copy()
        self.fits.clear()
        self.fit_infos.clear()

        return self

    @validate_call
    def adoptFit(
        self,
        fit: Union[
            GaussianModel,
            type[_VProfileCopy],
            CompoundModel_[Union[GaussianModel, _VProfileCopy]],
        ],
        *,
        fit_info: FitInfo | None = None,
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

    @validate_call
    def applyFit(
        self,
        fit: Union[
            GaussianModel,
            type[_VProfileCopy],
            CompoundModel_[Union[GaussianModel, _VProfileCopy]],
        ],
        *,
        fit_info: FitInfo | None = None,
        freeze: bool = False,
        update_emission: bool = False,
    ) -> Self:
        """
        Applies the 'fit' and 'fit_info' values to this 'LWindow' class.

        NOTES
        -----
        If 'freeze' is True, a copy of the 'fit' value is applied.
        """
        if freeze:
            fit = fit.copy()
            fs = (fit,) if fit.n_submodels == 1 else fit
            for f in (f for f in fs if isinstance(f, _VProfileCopy)):
                f._freeze_velocity_profile(inplace=True)
                f._forget_ties(inplace=True)

        n = fit.n_submodels
        self.fits[n] = self.fit = fit
        self.fit_infos[n] = self.fit_info = fit_info

        if update_emission:
            self.updateLinesEmission.__wrapped__(self, fit)

        return self

    @validate_call
    def updateLinesEmission(
        self,
        model: Optional[Union[GaussianModel, CompoundModel_[GaussianModel]]],
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **

        Updates the '_y_em' attribute to account for a new emission line model,
        'model'. If no previous model is found, '_y_em' is updated directly. If
        a previous model is found, this model's '_y_em' contribution is first
        removed, and the new 'model's contribution is added.

        If 'model' is not given, i.e. None, the previous model's contribution
        is removed.
        """
        # Subtract previous contribution
        self._y_em -= self._y_em_contrib

        # Calculate new contribution
        self._y_em_contrib[:] = 0

        if model is not None:
            mask = isfinite(self._x)
            self._y_em_contrib[mask] = model(self._x[mask])

            # Add current contribution
            self._y_em[mask] += self._y_em_contrib[mask]

        return self

    def removeLine(
        self,
        name: str,
    ) -> Self:
        if name not in self.names:
            raise ValueError(f"line '{name}' not in 'self.names'!")

        self.names.remove(name)
        self.complexes.pop(name)

        del self.lines[name]
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

    def sortLines(self) -> list[tuple[str, float]]:
        def sorting_key(item: tuple[str, float]) -> tuple[float, float]:
            return (
                0.0 if item[0] in self.is_multiplet_of.values() else 1.0,
                item[1],
            )

        return sorted(self.lines.items(), key=sorting_key)

    def getConfiguration(
        self,
        *,
        model: Optional[
            Union[GaussianModel, CompoundModel_[GaussianModel]]
        ] = None,
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
        # ! Make this method inplace, i.e. no final 'sum'?
        assert self.fit is not None

        if self.fit.n_submodels == 1:
            self.fit.name = self.fit.pure_name
        else:
            fs = order_submodels(self.fit, combine=False)
            current_count = defaultdict(lambda: 1)
            max_count = self.getConfiguration()

            for f in filter(lambda f: max_count[f.pure_name] > 1, fs):
                f.name = f"{f.pure_name}#{current_count[f.pure_name]}"
                current_count[f.pure_name] += 1

            self.fit = sum(fs[1:], start=fs[0])

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
        if bg_flux is None:
            bg_flux = self.default_bg

        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
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
        if bg_flux is None:
            bg_flux = self.default_bg

        assert n < m

        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
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

    @validated_apply_info_to_method(subjects=("error", "nonlinear"))
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
        pool: Pool_ | None = None,
        scale: Scale | None = None,
        variant: Variant | None = None,
        bootstrap_type: BootstrapType | None = None,
        n_sigmas: float | None = None,
        vary_lines: VaryLines | None = None,
        cfit: ContinuumFitResult | None = None,
        iterations: int | None = None,
        random_state: RandomState_ | None = None,
        renew_rng: bool | None = None,
        replace_missing: bool | None = None,
        tqdm_disable: bool | None = None,
        tqdm_leave: bool | None = None,
        fitter_kwargs: FitterKwargs | None = None,
    ) -> BaseBootstrapper:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
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
            renew_rng=renew_rng,
            replace_missing=replace_missing,
            tqdm_disable=tqdm_disable,
            tqdm_leave=tqdm_leave,
            logger=logger,
        )
        self.bootstrapper = cls(*args, **kwargs)
        return self.bootstrapper

    @validate_call
    def runBootstrapper(self) -> ErrorResult:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if self.bootstrapper is None:
            msg = "No bootstrapper instance found! \
                Run 'instantiateBootstrapper()' method first!"
            logger.critical(msg)
            raise RuntimeError(msg)
        if self.error_result is not None:
            logger.debug("Existing 'error_result' will be overwritten.")

        out = self.bootstrapper.run()
        self.error_result = self.bootstrapper.toErrorResult(out)
        return self.error_result

    def quickplot(
        self,
        *,
        figure: tuple[Figure, Axes] | None = None,
        figsize: tuple[float, float] = (8, 6),
        dpi: int = 300,
        title: str | None = None,
        n_sigma: float = 2.0,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        pl_color: str | None = DEFAULT_COLORS["pl"],
        fe_color: str | None = DEFAULT_COLORS["fe"],
        ba_color: str | None = DEFAULT_COLORS["ba"],
        hg_color: str | None = DEFAULT_COLORS["hg"],
        em_color: str | None = DEFAULT_COLORS["em"],
        sm_color: str | None = DEFAULT_COLORS["sm"],
        ab_color: str | None = DEFAULT_COLORS["ab"],
        xticks: tuple[float, float] | None = None,
        yticks: tuple[float, float] | None = None,
        logx: bool = False,
        logy: bool = False,
    ) -> tuple[Figure, Axes]:
        """
        Basic plotting routing for 'Spectrum' classes.

        NOTES
        -----
        This method overwrites but still calls the inherited '_quickplot'
        method, setting 'title' equal to 'self.title' by default.
        """
        xlim = xlim or self.x_bounds
        return quickplot(
            get_coords(self, x_bounds=xlim, replace_with_nan=False),
            self.info,
            figure=figure,
            figsize=figsize,
            dpi=dpi,
            title=title or self.__str__(simple=True).removesuffix("."),
            n_sigma=n_sigma,
            xlabel=xlabel or "auto",
            ylabel=ylabel or "auto",
            xlim=xlim,
            ylim=ylim or "auto",
            pl_color=pl_color,
            fe_color=fe_color,
            ba_color=ba_color,
            hg_color=hg_color,
            em_color=em_color,
            sm_color=sm_color,
            ab_color=ab_color,
            xticks=xticks or "auto",
            yticks=yticks or "auto",
            logx=logx,
            logy=logy,
        )

    def absorptionplot(
        self,
        *,
        figure: tuple[Figure, list[Axes, Axes, Axes]] | None = None,
        figsize: tuple[float, float] = (8, 6),
        dpi: int = 300,
        title: str | None = None,
        height_ratio: float = 3.0,
        n_sigma: float = 2.0,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
        plabel: str | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        zlim: tuple[float, float] | None = (-5, 5),
        plim: tuple[float, float] | None = (-5, 0),
        pl_color: str | None = DEFAULT_COLORS["pl"],
        fe_color: str | None = DEFAULT_COLORS["fe"],
        ba_color: str | None = DEFAULT_COLORS["ba"],
        hg_color: str | None = DEFAULT_COLORS["hg"],
        em_color: str | None = DEFAULT_COLORS["em"],
        sm_color: str | None = DEFAULT_COLORS["sm"],
        ab_color: str | None = DEFAULT_COLORS["ab"],
        xticks: tuple[float, float] | None = None,
        yticks: tuple[float, float] | None = None,
        zticks: tuple[float, float] | None = None,
        pticks: tuple[float, float] | None = None,
        logx: bool = False,
        logy: bool = False,
        logp: bool = True,
    ) -> tuple[Figure, list[Axes]]:
        xlim = xlim or self.x_bounds
        return absorptionplot(
            get_coords(self, x_bounds=xlim, replace_with_nan=False),
            self.info,
            figure=figure,
            figsize=figsize,
            dpi=dpi,
            title=title or self.__str__(simple=True).removesuffix("."),
            height_ratio=height_ratio,
            n_sigma=n_sigma,
            z_crit=self.info.absorption.z_crit,
            p_crit=self.info.absorption.p_crit,
            xlabel=xlabel or "auto",
            ylabel=ylabel or "auto",
            zlabel=zlabel or "auto",
            plabel=plabel or "auto",
            xlim=xlim or "auto",
            ylim=ylim or "auto",
            zlim=zlim or "auto",
            plim=plim or "auto",
            pl_color=pl_color,
            fe_color=fe_color,
            ba_color=ba_color,
            hg_color=hg_color,
            em_color=em_color,
            sm_color=sm_color,
            ab_color=ab_color,
            xticks=xticks or "auto",
            yticks=yticks or "auto",
            zticks=zticks or "auto",
            pticks=pticks or "auto",
            logx=logx,
            logy=logy,
            logp=logp,
        )

    def fitplot(
        self,
        *,
        plot_components: bool = False,
        plot_type: Literal["difference", "residual"] = "difference",
        figure: tuple[Figure, list[Axes, Axes]] | None = None,
        figsize: tuple[float, float] = (8, 6),
        dpi: int = 300,
        title: str | None = None,
        height_ratio: float = 3.0,
        n_sigma: float = 2.0,
        xlabel: str | None = None,
        ylabel: str | None = None,
        dlabel: str | None = None,
        zlabel: str | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        dlim: tuple[float, float] | None = None,
        zlim: tuple[float, float] | None = (-5, 5),
        pl_color: str | None = DEFAULT_COLORS["pl"],
        fe_color: str | None = DEFAULT_COLORS["fe"],
        ba_color: str | None = DEFAULT_COLORS["ba"],
        hg_color: str | None = DEFAULT_COLORS["hg"],
        em_color: str | None = DEFAULT_COLORS["em"],
        sm_color: str | None = DEFAULT_COLORS["sm"],
        ab_color: str | None = DEFAULT_COLORS["ab"],
        xticks: tuple[float, float] | None = None,
        yticks: tuple[float, float] | None = None,
        dticks: tuple[float, float] | None = None,
        zticks: tuple[float, float] | None = None,
        logx: bool = False,
        logy: bool = False,
        cmap_name: str = "tab20",
        distinguish_narrow: bool = True,
    ) -> tuple[Figure, list[Axes, Axes]]:
        xlim = xlim or self.x_bounds
        model = (self.spectrum or self).getModel() if plot_components else None
        coords = get_coords(self, x_bounds=xlim, replace_with_nan=False)
        if model is not None:
            ms = (model,) if model.n_submodels == 1 else model

            def func(m):
                return m.model_type in ("pl", "fe", "ba", "hg") \
                    or m.pure_name in self.names

            submodels = list(filter(func, ms))
            if submodels:
                model = sum(submodels[1:], start=submodels[0])
            else:
                model = None

        kwargs = {
            'plot_type': plot_type,
            'figure': figure,
            'figsize': figsize,
            'dpi': dpi,
            'title': title or self.__str__(),
            'height_ratio': height_ratio,
            'n_sigma': n_sigma,
            'xlabel': xlabel or "auto",
            'ylabel': ylabel or "auto",
            'dlabel': dlabel or "auto",
            'zlabel': zlabel or "auto",
            'xlim': xlim or "auto",
            'ylim': ylim or "auto",
            'dlim': dlim or "auto",
            'zlim': zlim or "auto",
            'pl_color': pl_color,
            'fe_color': fe_color,
            'ba_color': ba_color,
            'hg_color': hg_color,
            'em_color': em_color,
            'sm_color': sm_color,
            'ab_color': ab_color,
            'xticks': xticks or "auto",
            'yticks': yticks or "auto",
            'dticks': dticks or "auto",
            'zticks': zticks or "auto",
            'logx': logx,
            'logy': logy,
            'cmap_name': cmap_name,
            'distinguish_narrow': distinguish_narrow,
        }
        if plot_components:
            model = (self.spectrum or self).getModel()
            submodels = [
                m
                for m in ((model,) if model.n_submodels == 1 else model)
                if (m.model_type in {"pl", "fe", "ba", "hg"})
                or (m.pure_name in self.names)
            ]
            model = (
                sum(submodels[1:], start=submodels[0]) if submodels else None
            )
            with PrepareModel(x=coords.x, model=model):
                return fitplot(
                    coords,
                    self.info,
                    model=model,
                    **kwargs,
                )

        return fitplot(
            coords,
            self.info,
            model=None,
            **kwargs,
        )
