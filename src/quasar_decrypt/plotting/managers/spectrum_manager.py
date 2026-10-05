from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from numpy import nan, where
from quasar_typing.numpy import Array_, BoolVector, FloatVector
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

    def _apply_user_mask(self, arr: Array_, fill_value: Any = nan) -> Array_:
        return arr if self.mask is None else where(self.mask, arr, fill_value)

    @cached_property
    def _x(self) -> FloatVector: 
        return self._apply_user_mask(self.spectrum._x)

    @cached_property
    def _y(self) -> FloatVector: 
        return self._apply_user_mask(self.spectrum._y)

    @cached_property
    def _dy(self) -> FloatVector: 
        return self._apply_user_mask(self.spectrum._dy)

    @cached_property
    def _dx(self) -> FloatVector: 
        return self._apply_user_mask(self.spectrum._dx)

    @cached_property
    def _y_smooth(self) -> FloatVector: 
        return self._apply_user_mask(self.spectrum._y_smooth)

    @cached_property
    def _y_pl(self) -> FloatVector: 
        return self._apply_user_mask(self.spectrum._y_pl)

    @cached_property
    def _y_fe(self) -> FloatVector: 
        return self._apply_user_mask(self.spectrum._y_fe)

    @cached_property
    def _y_ba(self) -> FloatVector: 
        return self._apply_user_mask(self.spectrum._y_ba)

    @cached_property
    def _y_hg(self) -> FloatVector: 
        return self._apply_user_mask(self.spectrum._y_hg)

    @cached_property
    def _y_em(self) -> FloatVector: 
        return self._apply_user_mask(self.spectrum._y_em)

    @cached_property
    def _f(self) -> FloatVector: 
        return self._y_pl + self._y_fe + self._y_ba + self._y_hg + self._y_em

    @cached_property
    def _rejected_pixels(self) -> BoolVector:
        return self._apply_user_mask(self.spectrum._rejected_pixels, False)

    @cached_property
    def _absorbed_pixels(self) -> BoolVector: 
        return self._apply_user_mask(self.spectrum._absorbed_pixels, False)

    @cached_property
    def _valid_pixels(self) -> BoolVector: 
        return self._apply_user_mask(self.spectrum._valid_pixels, False)

    @cached_property
    def _log_valid_pixels(self) -> BoolVector: 
        return self._apply_user_mask(self.spectrum._log_valid_pixels, False)

    @cached_property
    def _p_absorbed(self) -> FloatVector: 
        return self._apply_user_mask(self.spectrum._p_absorbed)

    @cached_property
    def x0(self) -> float: 
        return self.spectrum.x0

    @cached_property
    def y0(self) -> float: 
        return self.spectrum.y0

    @cached_property
    def _x_log(self) -> FloatVector: 
        return self._apply_user_mask(self.spectrum._x_log)

    @cached_property
    def _y_log(self) -> FloatVector: 
        return self._apply_user_mask(self.spectrum._y_log)

    @cached_property
    def _dy_log(self) -> FloatVector: 
        return self._apply_user_mask(self.spectrum._dy_log)

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

    def _apply_user_and_spectrum_mask(self, arr: Array_, fill_value: Any = nan) -> Array_:
        if self.mask is None:
            return where(self.spectrum.mask, arr, fill_value)
        return where(self.mask & self.spectrum.mask, arr, fill_value)

    @cached_property
    def x(self) -> FloatVector: 
        return self._apply_user_and_spectrum_mask(self.spectrum._x)

    @cached_property
    def y(self) -> FloatVector: 
        return self._apply_user_and_spectrum_mask(self.spectrum._y)

    @cached_property
    def dy(self) -> FloatVector: 
        return self._apply_user_and_spectrum_mask(self.spectrum._dy)

    @cached_property
    def dx(self) -> FloatVector: 
        return self._apply_user_and_spectrum_mask(self.spectrum._dx)

    @cached_property
    def y_smooth(self) -> FloatVector: 
        return self._apply_user_and_spectrum_mask(self.spectrum._y_smooth)

    @cached_property
    def y_pl(self) -> FloatVector: 
        return self._apply_user_and_spectrum_mask(self.spectrum._y_pl)

    @cached_property
    def y_fe(self) -> FloatVector: 
        return self._apply_user_and_spectrum_mask(self.spectrum._y_fe)

    @cached_property
    def y_ba(self) -> FloatVector: 
        return self._apply_user_and_spectrum_mask(self.spectrum._y_ba)

    @cached_property
    def y_hg(self) -> FloatVector: 
        return self._apply_user_and_spectrum_mask(self.spectrum._y_hg)

    @cached_property
    def y_em(self) -> FloatVector: 
        return self._apply_user_and_spectrum_mask(self.spectrum._y_em)

    @cached_property
    def f(self) -> FloatVector: 
        return self.y_pl + self.y_fe + self.y_ba + self.y_hg + self.y_em

    @cached_property
    def rejected_pixels(self) -> BoolVector: 
        return self._apply_user_and_spectrum_mask(self.spectrum._rejected_pixels, False)

    @cached_property
    def absorbed_pixels(self) -> BoolVector: 
        return self._apply_user_and_spectrum_mask(self.spectrum._absorbed_pixels, False)

    @cached_property
    def valid_pixels(self) -> BoolVector: 
        return self._apply_user_and_spectrum_mask(self.spectrum._valid_pixels, False)

    @cached_property
    def log_valid_pixels(self) -> BoolVector: 
        return self._apply_user_and_spectrum_mask(self.spectrum._log_valid_pixels, False)

    @cached_property
    def p_absorbed(self) -> FloatVector: 
        return self._apply_user_and_spectrum_mask(self.spectrum._p_absorbed)

    @cached_property
    def x_log(self) -> FloatVector: 
        return self._apply_user_and_spectrum_mask(self.spectrum._x_log)

    @cached_property
    def y_log(self) -> FloatVector: 
        return self._apply_user_and_spectrum_mask(self.spectrum._y_log)

    @cached_property
    def dy_log(self) -> FloatVector: 
        return self._apply_user_and_spectrum_mask(self.spectrum._dy_log)

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