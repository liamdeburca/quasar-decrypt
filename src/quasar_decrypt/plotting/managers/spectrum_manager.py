from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from numpy import nan, where
from quasar_typing.numpy import Array, BoolVector, FloatVector
from quasar_utils.setup import Info

from ...utils._spec import _Spec
from ...utils._specwindow import _SpecWindow


@dataclass
class SpectrumManager:
    """
    Dataclass for managing a Spectrum. 
    """
    spectrum: _Spec | _SpecWindow
    mask: BoolVector | None = field(default=None, kw_only=True)

    original: bool = field(default=False, kw_only=True)

    def _apply_user_mask(self, arr: Array, fill_value: Any = nan) -> Array:
        if self.mask is None:
            return arr
        return where(self.mask, arr, fill_value)

    def _get_arr_from_spectrum(self, arr_name: str) -> Array:
        if self.original and not self.spectrum._no_original:
            arr_name = f"_{arr_name}_original"
        return getattr(self.spectrum, arr_name)

    @cached_property
    def _x(self) -> FloatVector:
        arr = self._get_arr_from_spectrum("_x")
        return self._apply_user_mask(arr)

    @cached_property
    def _y(self) -> FloatVector:
        arr = self._get_arr_from_spectrum("_y")
        return self._apply_user_mask(arr)

    @cached_property
    def _dy(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_dy")
        return self._apply_user_mask(arr)

    @cached_property
    def _dx(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_dx")
        return self._apply_user_mask(arr)

    @cached_property
    def _y_smooth(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_y_smooth")
        return self._apply_user_mask(arr)

    @cached_property
    def _y_pl(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_y_pl")
        return self._apply_user_mask(arr)

    @cached_property
    def _y_fe(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_y_fe")
        return self._apply_user_mask(arr)

    @cached_property
    def _y_ba(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_y_ba")
        return self._apply_user_mask(arr)

    @cached_property
    def _y_hg(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_y_hg")
        return self._apply_user_mask(arr)

    @cached_property
    def _y_em(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_y_em")
        return self._apply_user_mask(arr)

    @cached_property
    def _f(self) -> FloatVector: 
        return self._y_pl + self._y_fe + self._y_ba + self._y_hg + self._y_em

    @cached_property
    def _rejected_pixels(self) -> BoolVector:
        arr = self._get_arr_from_spectrum("_rejected_pixels")
        return self._apply_user_mask(arr, False)

    @cached_property
    def _absorbed_pixels(self) -> BoolVector: 
        arr = self._get_arr_from_spectrum("_absorbed_pixels")
        return self._apply_user_mask(arr, False)

    @cached_property
    def _valid_pixels(self) -> BoolVector: 
        arr = self._get_arr_from_spectrum("_valid_pixels")
        return self._apply_user_mask(arr, False)

    @cached_property
    def _log_valid_pixels(self) -> BoolVector: 
        arr = self._get_arr_from_spectrum("_log_valid_pixels")
        return self._apply_user_mask(arr, False)

    @cached_property
    def _p_absorbed(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_p_absorbed")
        return self._apply_user_mask(arr)

    @cached_property
    def x0(self) -> float: 
        return self.spectrum.x0

    @cached_property
    def y0(self) -> float: 
        return self.spectrum.y0

    @cached_property
    def _x_log(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_x_log")
        return self._apply_user_mask(arr)

    @cached_property
    def _y_log(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_y_log")
        return self._apply_user_mask(arr)

    @cached_property
    def _dy_log(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_dy_log")
        return self._apply_user_mask(arr)

    @cached_property
    def info(self) -> Info: 
        return self.spectrum.info

    @cached_property
    def _n_rej(self) -> int: 
        return self._rejected_pixels.sum()

    @cached_property
    def _n_abs(self) -> int: 
        return self._absorbed_pixels.sum()

    @cached_property
    def _n_val(self) -> int: 
        return self._valid_pixels.sum()

    @cached_property
    def _n_logval(self) -> int: 
        return self._log_valid_pixels.sum()

    ### Masked properties

    def _apply_user_and_spectrum_mask(self, arr: Array, fill_value: Any = nan) -> Array:
        if self.mask is None:
            return where(self.spectrum.mask, arr, fill_value)
        return where(self.mask & self.spectrum.mask, arr, fill_value)

    @cached_property
    def x(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_x")
        return self._apply_user_and_spectrum_mask(arr)

    @cached_property
    def y(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_y")
        return self._apply_user_and_spectrum_mask(arr)

    @cached_property
    def dy(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_dy")
        return self._apply_user_and_spectrum_mask(arr)

    @cached_property
    def dx(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_dx")
        return self._apply_user_and_spectrum_mask(arr)

    @cached_property
    def y_smooth(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_y_smooth")
        return self._apply_user_and_spectrum_mask(arr)

    @cached_property
    def y_pl(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_y_pl")
        return self._apply_user_and_spectrum_mask(arr)

    @cached_property
    def y_fe(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_y_fe")
        return self._apply_user_and_spectrum_mask(arr)

    @cached_property
    def y_ba(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_y_ba")
        return self._apply_user_and_spectrum_mask(arr)

    @cached_property
    def y_hg(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_y_hg")
        return self._apply_user_and_spectrum_mask(arr)

    @cached_property
    def y_em(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_y_em")
        return self._apply_user_and_spectrum_mask(arr)

    @cached_property
    def f(self) -> FloatVector: 
        return self.y_pl + self.y_fe + self.y_ba + self.y_hg + self.y_em

    @cached_property
    def rejected_pixels(self) -> BoolVector: 
        arr = self._get_arr_from_spectrum("_rejected_pixels")
        return self._apply_user_and_spectrum_mask(arr, False)

    @cached_property
    def absorbed_pixels(self) -> BoolVector: 
        arr = self._get_arr_from_spectrum("_absorbed_pixels")
        return self._apply_user_and_spectrum_mask(arr, False)

    @cached_property
    def valid_pixels(self) -> BoolVector: 
        arr = self._get_arr_from_spectrum("_valid_pixels")
        return self._apply_user_and_spectrum_mask(arr, False)

    @cached_property
    def log_valid_pixels(self) -> BoolVector: 
        arr = self._get_arr_from_spectrum("_log_valid_pixels")
        return self._apply_user_and_spectrum_mask(arr, False)

    @cached_property
    def p_absorbed(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_p_absorbed")
        return self._apply_user_and_spectrum_mask(arr)

    @cached_property
    def x_log(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_x_log")
        return self._apply_user_and_spectrum_mask(arr)

    @cached_property
    def y_log(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_y_log")
        return self._apply_user_and_spectrum_mask(arr)

    @cached_property
    def dy_log(self) -> FloatVector: 
        arr = self._get_arr_from_spectrum("_dy_log")
        return self._apply_user_and_spectrum_mask(arr)

    @cached_property
    def n_rej(self) -> int: 
        return self.rejected_pixels.sum()

    @cached_property
    def n_abs(self) -> int: 
        return self.absorbed_pixels.sum()

    @cached_property
    def n_val(self) -> int: 
        return self.valid_pixels.sum()

    @cached_property
    def n_logval(self) -> int: 
        return self.log_valid_pixels.sum()