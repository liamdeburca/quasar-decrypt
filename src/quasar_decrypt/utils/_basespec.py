__all__ = ["_BaseSpec"]

from abc import abstractmethod
from dataclasses import field
from functools import wraps
from logging import getLogger
from typing import Any, Literal, Optional, Self, Union

import numpy as np
import pydantic
from quasar_models import (
    BalmerModel,
    EmModel,
    HostGalaxyModel,
    IronModel,
    PowerLawModel,
)
from quasar_typing.astropy import CompoundModelType
from quasar_typing.bounds import CoordBounds
from quasar_typing.numpy import BoolVector, FloatMatrix, FloatVector
from quasar_typing.scipy import CSRMatrix
from quasar_typing.string_selections import BackgroundFlux

from .utils import CachedGetMask

logger = getLogger(__name__)


@pydantic.dataclasses.dataclass
class _BaseSpec:
    """
    Base class for '_Spec', '_SpecWindow', and '_SpecWindowList' classes.
    
    Stores very basic attributes and essential methods. 
    """
    x_bounds: CoordBounds = field(kw_only=True)
    get_mask: CachedGetMask = field(kw_only=True)           # Not pickled
    get_mask_original: CachedGetMask = field(kw_only=True)  # Not pickled

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("get_mask", None)
        state.pop("get_mask_original", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.x_bounds = state["x_bounds"]
        self._no_original = state["_no_original"]

    def __str__(self, simple: bool = False) -> str:
        s = f"<{self.__class__.__name__} object at {hex(id(self))}> "
        s += f"x_bounds={self.x_bounds}"
        if not simple:
            n = self.size
            s += " w/ "
            s += f"{self.n_rej}/{n} (rej.) "
            s += f"{self.n_abs}/{n} (abs.) "
            s += f"{self.n_val}/{n} (val.) "
            s += f"{self.n_logval}/{n} (log-val.)"
        return s

    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs) -> Self:
        pass

    # Mask covering the 'x_bounds' of the spectrum

    @property
    def mask(self) -> BoolVector:
        return self.get_mask(*self.x_bounds)

    @property
    def mask_original(self) -> BoolVector:
        return self.get_mask_original(*self.x_bounds)

    # The size (no. of pixels) of the mask -- cannot be modified directly

    @property
    def size(self) -> int:
        return self.mask.astype(int).sum()

    @property
    def size_original(self) -> int:
        return self.mask_original.astype(int).sum()

    # The masked wavelength array

    @property
    def x(self) -> FloatVector:
        """Masked vector of rest wavelength values."""
        return self._x[self.mask]

    @property
    def x_original(self) -> FloatVector:
        """Original, masked vector of rest wavelength values."""
        return self._x_original[self.mask_original]

    # The masked flux density array

    @property
    def y(self) -> FloatVector:
        """Masked vector of flux density values."""
        return self._y[self.mask]

    @property
    def y_original(self) -> FloatVector:
        """Original, masked vector of flux density values."""
        return self._y_original[self.mask_original]

    # The masked flux density uncertainty array

    @property
    def dy(self) -> FloatVector:
        """Masked vector of flux density uncertainty values."""
        return self._dy[self.mask]

    @property
    def dy_original(self) -> FloatVector:
        """Original, masked vector of flux density uncertainty values."""
        return self._dy_original[self.mask_original]

    # The masked wavelength bin size array

    @property
    def dx(self) -> FloatVector:
        """Masked vector of wavelength bin size values."""
        return self._dx[self.mask]

    @property
    def dx_original(self) -> FloatVector:
        """Original, masked vector of wavelength bin size values."""
        return self._dx_original[self.mask_original]

    # The matrix of resolution kernels (corrects for instrumental broadening)

    @property
    def kernels(self) -> FloatMatrix | CSRMatrix:
        """Masked matrix of resolution kernels."""
        from quasar_utils.resolving import mask_kernels
        return mask_kernels(
            self._kernels,
            self.mask,
        )

    @property
    def kernels_original(self) -> FloatMatrix | CSRMatrix:
        """Masked matrix of original resolution kernels."""
        from quasar_utils.resolving import mask_kernels
        return mask_kernels(
            self._kernels_original,
            self.mask_original,
        )

    # The masked log-wavelength array

    @property
    def x_log(self) -> FloatVector:
        """Masked vector of log-wavelength values."""
        return self._x_log[self.mask]

    @property
    def x_log_original(self) -> FloatVector:
        """Original, masked vector of log-wavelength values."""
        return self._x_log_original[self.mask_original]

    # The masked log-flux density array

    @property
    def y_log(self) -> FloatVector:
        """Masked vector of log-flux density values."""
        return self._y_log[self.mask]

    @property
    def y_log_original(self) -> FloatVector:
        """Original, masked vector of log-flux density values."""
        return self._y_log_original[self.mask_original]

    # The masked log-flux density uncertainty array

    @property
    def dy_log(self) -> FloatVector:
        """Masked vector of log-flux density uncertainty values."""
        return self._dy_log[self.mask]

    @property
    def dy_log_original(self) -> FloatVector:
        """Original, masked vector of log-flux density uncertainty values."""
        return self._dy_log_original[self.mask_original]

    # The masked smoothed flux density array -- can only be modified inplace

    @property
    def y_smooth(self) -> FloatVector:
        """Masked vector of smoothed flux density values."""
        return self._y_smooth[self.mask]

    @y_smooth.setter
    def y_smooth(self, value: FloatVector) -> None:
        self._y_smooth[self.mask] = value

    @property
    def y_smooth_original(self) -> FloatVector:
        """Original, masked vector of smoothed flux density values."""
        return self._y_smooth_original[self.mask_original]

    @y_smooth_original.setter
    def y_smooth_original(self, value: FloatVector) -> None:
        self._y_smooth_original[self.mask_original] = value

    # The masked continuum flux density array -- can only be modified inplace

    @property
    def y_pl(self) -> FloatVector:
        """Masked vector of power law continuum flux density values."""
        return self._y_pl[self.mask]

    @y_pl.setter
    def y_pl(self, value: FloatVector) -> None:
        self._y_pl[self.mask] = value

    @property
    def y_pl_original(self) -> FloatVector:
        """Original, masked vector of power law continuum flux density values."""
        return self._y_pl_original[self.mask_original]

    @y_pl_original.setter
    def y_pl_original(self, value: FloatVector) -> None:
        self._y_pl_original[self.mask_original] = value

    # The masked iron flux density array -- can only be modified inplace

    @property
    def y_fe(self) -> FloatVector:
        """Masked vector of iron pseudo-continuum flux density values."""
        return self._y_fe[self.mask]

    @y_fe.setter
    def y_fe(self, value: FloatVector) -> None:
        self._y_fe[self.mask] = value

    @property
    def y_fe_original(self) -> FloatVector:
        """Original, masked vector of iron pseudo-continuum flux density values."""
        return self._y_fe_original[self.mask_original]

    @y_fe_original.setter
    def y_fe_original(self, value: FloatVector) -> None:
        self._y_fe_original[self.mask_original] = value

    # The masked Balmer flux density array -- can only be modified inplace

    @property
    def y_ba(self) -> FloatVector:
        """Masked vector of Balmer pseudo-continuum flux density values."""
        return self._y_ba[self.mask]

    @y_ba.setter
    def y_ba(self, value: FloatVector) -> None:
        self._y_ba[self.mask] = value

    @property
    def y_ba_original(self) -> FloatVector:
        """Original, masked vector of Balmer pseudo-continuum flux density values."""
        return self._y_ba_original[self.mask_original]

    @y_ba_original.setter
    def y_ba_original(self, value: FloatVector) -> None:
        self._y_ba_original[self.mask_original] = value

    # The masked host galaxy flux density array -- can only be modified inplace

    @property
    def y_hg(self) -> FloatVector:
        """Masked vector of host galaxy flux density values."""
        return self._y_hg[self.mask]

    @y_hg.setter
    def y_hg(self, value: FloatVector) -> None:
        self._y_hg[self.mask] = value

    @property
    def y_hg_original(self) -> FloatVector:
        """Original, masked vector of host galaxy flux density values."""
        return self._y_hg_original[self.mask_original]

    @y_hg_original.setter
    def y_hg_original(self, value: FloatVector) -> None:
        self._y_hg_original[self.mask_original] = value

    # The masked emission lines flux density array -- can only be modified inplace

    @property
    def y_em(self) -> FloatVector:
        """Masked vector of emission line flux density values."""
        return self._y_em[self.mask]

    @y_em.setter
    def y_em(self, value: FloatVector) -> None:
        self._y_em[self.mask] = value

    @property
    def y_em_original(self) -> FloatVector:
        """Original, masked vector of emission line flux density values."""
        return self._y_em_original[self.mask_original]

    @y_em_original.setter
    def y_em_original(self, value: FloatVector) -> None:
        self._y_em_original[self.mask_original] = value

    # The masked boolean mask of rejected pixels -- can only be modified inplace -- deleting resets the mask

    @property
    def rejected_pixels(self) -> BoolVector:
        """Masked boolean mask of rejected pixels."""
        return self._rejected_pixels[self.mask]

    @rejected_pixels.setter
    def rejected_pixels(self, value: BoolVector) -> None:
        self.applyRejections.__wrapped__(self, value, enforce=True)

    @rejected_pixels.deleter
    def rejected_pixels(self) -> None:
        self.resetRejections()

    @property
    def rejected_pixels_original(self) -> BoolVector:
        """Original, masked boolean mask of rejected pixels."""
        return self._rejected_pixels_original[self.mask_original]

    @rejected_pixels_original.setter
    def rejected_pixels_original(self, value: BoolVector) -> None:
        self._rejected_pixels_original[self.mask_original] = value

    @rejected_pixels_original.deleter
    def rejected_pixels_original(self) -> None:
        self.resetRejections()

    # The masked boolean mask of absorbed pixels -- can only be modified inplace -- deleting resets the mask

    @property
    def absorbed_pixels(self) -> BoolVector:
        """Masked boolean mask of absorbed pixels."""
        return self._absorbed_pixels[self.mask]

    @absorbed_pixels.setter
    def absorbed_pixels(self, value: BoolVector) -> None:
        self.applyAbsorption.__wrapped__(self, value, enforce=True)

    @absorbed_pixels.deleter
    def absorbed_pixels(self) -> None:
        self.resetAbsorption()

    @property
    def absorbed_pixels_original(self) -> BoolVector:
        """Original, masked boolean mask of absorbed pixels."""
        return self._absorbed_pixels_original[self.mask_original]

    @absorbed_pixels_original.setter
    def absorbed_pixels_original(self, value: BoolVector) -> None:
        self._absorbed_pixels_original[self.mask_original] = value

    @absorbed_pixels_original.deleter
    def absorbed_pixels_original(self) -> None:
        self.resetAbsorption()

    # The masked boolean mask of valid pixels -- can only be modified inplace

    @property
    def valid_pixels(self) -> BoolVector:
        """Masked boolean mask of valid pixels."""
        return self._valid_pixels[self.mask]

    @valid_pixels.setter
    def valid_pixels(self, value: BoolVector) -> None:
        self._valid_pixels[self.mask] = value

    @property
    def valid_pixels_original(self) -> BoolVector:
        """Original, masked boolean mask of valid pixels."""
        return self._valid_pixels_original[self.mask_original]

    @valid_pixels_original.setter
    def valid_pixels_original(self, value: BoolVector) -> None:
        self._valid_pixels_original[self.mask_original] = value

    # The masked boolean mask of valid pixels in log-space -- can only be modified inplace

    @property
    def log_valid_pixels(self) -> BoolVector:
        """Masked boolean mask of valid pixels in log-space."""
        return self._log_valid_pixels[self.mask]

    @log_valid_pixels.setter
    def log_valid_pixels(self, value: BoolVector) -> None:
        self._log_valid_pixels[self.mask] = value

    @property
    def log_valid_pixels_original(self) -> BoolVector:
        """Original, masked boolean mask of valid pixels in log-space."""
        return self._log_valid_pixels_original[self.mask_original]

    @log_valid_pixels_original.setter
    def log_valid_pixels_original(self, value: BoolVector) -> None:
        self._log_valid_pixels_original[self.mask_original] = value

    # The masked absorbed pixel p-values -- can only be modified inplace

    @property
    def p_absorbed(self) -> FloatVector:
        """Masked vector of absorbed pixel p-values."""
        return self._p_absorbed[self.mask]

    @p_absorbed.setter
    def p_absorbed(self, value: FloatVector) -> None:
        self._p_absorbed[self.mask] = value

    @property
    def p_absorbed_original(self) -> FloatVector:
        """Original, masked vector of absorbed pixel p-values."""
        return self._p_absorbed_original[self.mask_original]

    @p_absorbed_original.setter
    def p_absorbed_original(self, value: FloatVector) -> None:
        self._p_absorbed_original[self.mask_original] = value

    # The number of rejected pixels in the mask -- cannot be modified directly

    @property
    def n_rej(self) -> int:
        """Number of rejected pixels in the mask."""
        return self.rejected_pixels.sum()

    @property
    def n_rej_original(self) -> int:
        """Number of rejected pixels (original) in the mask."""
        return self.rejected_pixels_original.sum()

    # The number of absorbed pixels in the mask -- cannot be modified directly

    @property
    def n_abs(self) -> int:
        """Number of absorbed pixels in the mask."""
        return self.absorbed_pixels.sum()

    @property
    def n_abs_original(self) -> int:
        """Number of absorbed pixels (original) in the mask."""
        return self.absorbed_pixels_original.sum()

    # The number of valid pixels in the mask -- cannot be modified directly

    @property
    def n_val(self) -> int:
        """Number of valid pixels in the mask."""
        return self.valid_pixels.sum()

    @property
    def n_val_original(self) -> int:
        """Number of valid pixels (original) in the mask."""
        return self.valid_pixels_original.sum()

    # The number of valid pixels in log-space in the mask -- cannot be modified directly

    @property
    def n_logval(self) -> int:
        """Number of valid pixels in log-space in the mask."""
        return self.log_valid_pixels.sum()

    @property
    def n_logval_original(self) -> int:
        """Number of valid pixels (original) in log-space in the mask."""
        return self.log_valid_pixels_original.sum()

    def _update_original_flag(
        self, 
        original: bool, 
        log: bool,
        prefix: str = "",
        suffix: str = "",
    ) -> bool:
        if original and self._no_original:
            if log:
                msg = "Original data is not available -> "\
                    "setting `original` to False."
                logger.debug(prefix + msg + suffix)
            return False
        return bool(original)

    ### getMask

    @pydantic.validate_call
    def getMask(
        self,
        *,
        original: bool = False,
        covered: bool = True,
        without_rejections: bool = False,
        without_absorption: bool = False,
        valid: bool = False,
        log_valid: bool = False,
        as_tuple: bool = False,
    ) -> BoolVector | tuple[BoolVector, BoolVector]:
        """
        Get the total mask for the current object, corresponding to the window 
        mask with additional filtering options.
        """
        return self._getMask(
            original=original,
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=valid,
            log_valid=log_valid,
            as_tuple=as_tuple,
        )

    @wraps(getMask)
    def _getMask(
        self,
        *,
        original: bool = False,
        covered: bool = True,
        without_rejections: bool = False,
        without_absorption: bool = False,
        valid: bool = False,
        log_valid: bool = False,
        as_tuple: bool = False,
    ) -> BoolVector | tuple[BoolVector, BoolVector]:
        original = self._update_original_flag(
            original=original,
            log=True,
        )
        suffix = "_original" if original else ""
        
        primary_mask = getattr(self, "mask" + suffix).copy(order="C")
        if not covered:
            primary_mask[:] = True
    
        secondary_mask = primary_mask.copy(order="C")
        if without_rejections:
            secondary_mask &= ~getattr(self, "_rejected_pixels" + suffix)
        if without_absorption:
            secondary_mask &= ~getattr(self, "_absorbed_pixels" + suffix)

        if log_valid:
            secondary_mask &= getattr(self, "_log_valid_pixels" + suffix)
        elif valid:
            secondary_mask &= getattr(self, "_valid_pixels" + suffix)

        if as_tuple:
            return primary_mask, secondary_mask
        return primary_mask & secondary_mask

    ### getMaskedCoords

    def _get_masked_coords_helper(
        self,
        *,
        original: bool,
        mode: Literal["r", "c"] | None,
        primary_mask: BoolVector,
        secondary_mask: BoolVector,
        log_valid: bool,
        bg_flux: BackgroundFlux | None,
        shrink: bool | None,
    ):
        from .masked_coords import (
            ContiguousMaskedCoords,
            MaskedCoords,
            OriginalContiguousMaskedCoords,
            OriginalMaskedCoords,
            OriginalReadOnlyMaskedCoords,
            ReadOnlyMaskedCoords,
        )
        original = self._update_original_flag(
            original=original,
            log=False,
        )
        match original, mode:
            case True, "r":
                cls = OriginalReadOnlyMaskedCoords
            case True, "c":
                cls = OriginalContiguousMaskedCoords
            case True, _:
                cls = OriginalMaskedCoords
            case False, "r":
                cls = ReadOnlyMaskedCoords
            case False, "c":
                cls = ContiguousMaskedCoords
            case False, _:
                cls = MaskedCoords

        coords = cls(
            self, primary_mask, secondary_mask, 
            bg_flux=bg_flux, 
        )
        if log_valid:
            coords = coords.ensureLogValid()
        if shrink is not None:
            coords = coords.limitLoss(shrink=shrink)
        return coords


    @pydantic.validate_call
    def getMaskedCoords(
        self,
        *,
        original: bool = False,
        mode: Literal["r", "c"] | None = None,
        covered: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        valid: bool = False,
        log_valid: bool = False,
        bg_flux: BackgroundFlux | None = None,
        shrink: bool | None = None,
    ):
        return self._getMaskedCoords(
            original=original,
            mode=mode,
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=valid,
            log_valid=log_valid,
            bg_flux=bg_flux,
            shrink=shrink,
        )


    @wraps(getMaskedCoords)
    def _getMaskedCoords(
        self,
        *,
        original: bool = False,
        mode: Literal["r", "c"] | None = None,
        covered: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        valid: bool = False,
        log_valid: bool = False,
        bg_flux: BackgroundFlux | None = None,
        shrink: bool | None = None,
    ):
        original = self._update_original_flag(
            original=original,
            log=False,
        )
        primary_mask, secondary_mask = self.getMask.__wrapped__(
            self,
            original=original,
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=valid,
            log_valid=log_valid,
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

    ###

    def resetRejections(self) -> None:
        msg = "Resetting rejection mask for "
        msg += self.__str__(simple=True).removesuffix(".")
        msg += ": "

        n = self.size
        r = self.n_rej
        msg += f"{r}/{n} -> {0}/{n}"

        if self._no_original:
            msg += " (no original)"
            self._rejected_pixels[:] = False
        else:
            n = self.size_original
            r = self.n_rej_original
            msg += f", (original) {r}/{n} -> {0}/{n}"
            self._rejected_pixels[:] = False
            self._rejected_pixels_original[:] = False

        logger.debug(msg)

    def resetAbsorption(self) -> None:
        msg = "Resetting absorption mask for "
        msg += self.__str__(simple=True).removesuffix(".")
        msg += ": "

        n = self.size
        a = self.n_abs
        msg += f"{a}/{n} -> {0}/{n}"

        if self._no_original:
            msg += " (no original)"
            self._absorbed_pixels[:] = False
        else:
            n = self.size_original
            a = self.n_abs_original
            msg += f", (original) {a}/{n} -> {0}/{n}"
            self._absorbed_pixels[:] = False
            self._absorbed_pixels_original[:] = False

        logger.debug(msg)

    ### applyRejections

    @pydantic.validate_call
    def applyRejections(
        self,
        mask: BoolVector,
        enforce: bool = True,
        original: bool = False,
    ) -> Self:
        """
        Update the rejection mask with the provided boolean mask.
        """
        return self._applyRejections(
            mask,
            enforce=enforce,
            original=original,
        )

    @wraps(applyRejections)
    def _applyRejections(
        self,
        mask: BoolVector,
        enforce: bool = True,
        original: bool = False,
    ) -> Self:
        suffix = "_original" if original else ""

        msg = "[ORIGINAL] " if original else ""
        msg += "Applying rejection mask to "
        msg += self.__str__(simple=True).removesuffix(".")
        msg += ": "

        m = mask.size
        _n = getattr(self, "_x" + suffix).size
        n = getattr(self, "size" + suffix)
        if m not in {n, _n}:
            msg += f"Improper mask size, {m}. Should be either {n} or {_n}."
            logger.critical(msg)
            raise ValueError(msg)

        r1 = getattr(self, "n_rej" + suffix)
        if m == _n:
            if enforce:
                getattr(self, "_rejected_pixels" + suffix)[:] = mask
            else:
                getattr(self, "_rejected_pixels" + suffix)[:] |= mask
        else:
            if enforce:
                getattr(self, "_rejected_pixels" + suffix)[self.mask] = mask
            else:
                getattr(self, "_rejected_pixels" + suffix)[self.mask] |= mask

        r2 = getattr(self, "n_rej" + suffix)
        logger.debug(msg + f"{r1}/{n} -> {r2}/{n} (rej.).")

        return self

    ### applyAbsorption

    @pydantic.validate_call
    def applyAbsorption(
        self,
        mask: BoolVector,
        y_smooth: FloatVector | None = None,
        enforce: bool = True,
        original: bool = False,
    ) -> Self:
        """
        Update the absorption mask with the provided boolean mask, and 
        optionally the array of smoothed flux density values.
        """
        return self._applyAbsorption(
            mask,
            y_smooth=y_smooth,
            enforce=enforce,
            original=original,
        )

    @wraps(applyAbsorption)
    def _applyAbsorption(
        self,
        mask: BoolVector,
        y_smooth: FloatVector | None = None,
        enforce: bool = True,
        original: bool = False,
    ) -> Self:
        suffix = "_original" if original else ""

        msg = "[ORIGINAL] " if original else ""
        msg += "Applying absorption mask to "
        msg += self.__str__(simple=True).removesuffix(".")
        msg += ": "

        m = mask.size
        _n = getattr(self, "_x" + suffix).size
        n = getattr(self, "size" + suffix)

        if m not in {n, _n}:
            msg += f"Improper mask size, {m}. Should be either {n} or {_n}."
            logger.critical(msg)
            raise ValueError(msg)

        a1 = getattr(self, "n_abs" + suffix)
        if m == _n:
            if enforce:
                getattr(self, "_absorbed_pixels" + suffix)[:] = mask
            else:
                getattr(self, "_absorbed_pixels" + suffix)[:] |= mask

            if y_smooth is not None:
                getattr(self, "_y_smooth" + suffix)[:] = y_smooth
        else:
            if enforce:
                getattr(self, "_absorbed_pixels" + suffix)[self.mask] = mask
            else:
                getattr(self, "_absorbed_pixels" + suffix)[self.mask] |= mask

            if y_smooth is not None:
                getattr(self, "_y_smooth" + suffix)[self.mask] = y_smooth

        a2 = getattr(self, "n_abs" + suffix)
        logger.debug(msg + f"{a1}/{n} -> {a2}/{n} (abs.).")

        return self

    def _update_emission_helper(
        self,
        model: object | None,
        original: bool,
        data_type: str,
    ) -> Self:
        from quasar_models.modeling import PrepareModel

        original = self._update_original_flag(original, False)

        arr: FloatVector = getattr(self, f"_y_{data_type}")
        arr[:] = 0.0
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
                arr[eval_mask] = seq_model(x)

        if original:
            arr_original: FloatVector = getattr(self, f"_y_{data_type}_original")
            arr_original[:] = 0.0

            if model is not None:
                eval_mask = np.isfinite(self._x_original)
                x = self._x_original[eval_mask]

                preparer = PrepareModel(
                    x=self._x_original,
                    model=model,
                    eval_mask=eval_mask,
                    comp_mask=eval_mask,
                    kernels=self._kernels_original,
                    is_logbinned=False,
                )
                with preparer as (seq_model, _):
                    arr_original[eval_mask] = seq_model(x)

        return self

    ### updateContinuumEmission

    @pydantic.validate_call
    def updateContinuumEmission(
        self,
        model: Optional[PowerLawModel] = None,
        original: bool = False,
    ) -> Self:
        return self._updateContinuumEmission(model=model, original=original)

    @wraps(updateContinuumEmission)
    def _updateContinuumEmission(
        self,
        model: Optional[PowerLawModel] = None,
        original: bool = False,
    ) -> Self:
        return self._update_emission_helper(model, original, "pl")

    ### updateIronEmission

    @pydantic.validate_call
    def updateIronEmission(
        self,
        model: Union[IronModel, CompoundModelType[IronModel], None] = None,
        original: bool = False,
    ) -> Self:
        return self._updateIronEmission(model=model, original=original)

    @wraps(updateIronEmission)
    def _updateIronEmission(
        self,
        model: Union[IronModel, CompoundModelType[IronModel], None] = None,
        original: bool = False,
    ) -> Self:
        return self._update_emission_helper(model, original, "fe")

    ### updateBalmerEmission

    @pydantic.validate_call
    def updateBalmerEmission(
        self,
        model: Optional[BalmerModel] = None,
        original: bool = False,
    ) -> Self:
        return self._updateBalmerEmission(model=model, original=original)

    @wraps(updateBalmerEmission)
    def _updateBalmerEmission(
        self,
        model: Optional[BalmerModel] = None,
        original: bool = False,
    ) -> Self:
        return self._update_emission_helper(model, original, "ba")

    ### updateHostGalaxyEmission

    @pydantic.validate_call
    def updateHostGalaxyEmission(
        self,
        model: Optional[HostGalaxyModel] = None,
        original: bool = False,
    ) -> Self:
        return self._updateHostGalaxyEmission(model=model, original=original)

    @wraps(updateHostGalaxyEmission)
    def _updateHostGalaxyEmission(
        self,
        model: Optional[HostGalaxyModel] = None,
        original: bool = False,
    ) -> Self:
        return self._update_emission_helper(model, original, "hg")
    
    ### updateLinesEmission

    @pydantic.validate_call
    def updateLinesEmission(
        self,
        model: Union[EmModel, CompoundModelType[EmModel], None] = None,
        original: bool = False,
    ) -> Self:
        return self._updateLinesEmission(model=model, original=original)

    @wraps(updateLinesEmission)
    def _updateLinesEmission(
        self,
        model: Union[EmModel, CompoundModelType[EmModel], None] = None,
        original: bool = False,
    ) -> Self:
        return self._update_emission_helper(model, original, "em")