__all__ = ["_SpecWindow"]

from collections.abc import Callable
from dataclasses import field
from logging import getLogger
from typing import ClassVar, Self

from pydantic.dataclasses import dataclass
from quasar_typing.bounds import CoordBounds
from quasar_typing.misc import BackgroundFlux
from quasar_typing.numpy import BoolVector, FloatVector
from quasar_utils.decorators import validate_call
from quasar_utils.setup import Info

from ._basespec import _BaseSpec
from ._spec import _Spec
from .utils import create_cached_get_mask

logger = getLogger(__name__)        

@dataclass
class _SpecWindow(_BaseSpec):
    """
    Class for storing spectral windows. 

    Has no inherent attributes, but points to a '_Spectrum' object.
    """
    spectrum: _Spec = field(kw_only=True)
    x_bounds: CoordBounds = field(kw_only=True)
    get_mask: Callable[[float, float], BoolVector] | None = field(kw_only=True)

    default_bg: ClassVar[BackgroundFlux]

    @classmethod
    @validate_call
    def create(
        cls,
        *,
        spectrum: _Spec,
        x_bounds: CoordBounds,
        get_mask: Callable[[float, float], BoolVector] | None = None,
    ) -> Self:
        """
        Creates a '_SpecWindow' object from a '_Spec' object and x_bounds.
        """
        if get_mask is None:
            get_mask = create_cached_get_mask(spectrum._x)

        return cls(
            spectrum=spectrum,
            x_bounds=x_bounds,
            get_mask=get_mask,
        )

    @property
    def _x(self) -> FloatVector: 
        return self.spectrum._x

    @property
    def _y(self) -> FloatVector: 
        return self.spectrum._y

    @property
    def _dy(self) -> FloatVector: 
        return self.spectrum._dy

    @property
    def _dx(self) -> FloatVector: 
        return self.spectrum._dx

    @property
    def _y_smooth(self) -> FloatVector: 
        return self.spectrum._y_smooth

    @property
    def _y_pl(self) -> FloatVector: 
        return self.spectrum._y_pl

    @property
    def _y_fe(self) -> FloatVector: 
        return self.spectrum._y_fe

    @property
    def _y_ba(self) -> FloatVector: 
        return self.spectrum._y_ba

    @property
    def _y_hg(self) -> FloatVector: 
        return self.spectrum._y_hg

    @property
    def _y_em(self) -> FloatVector: 
        return self.spectrum._y_em

    @property
    def _rejected_pixels(self) -> BoolVector: 
        return self.spectrum._rejected_pixels

    @property
    def _absorbed_pixels(self) -> BoolVector: 
        return self.spectrum._absorbed_pixels

    @property
    def _valid_pixels(self) -> BoolVector: 
        return self.spectrum._valid_pixels

    @property
    def _log_valid_pixels(self) -> BoolVector: 
        return self.spectrum._log_valid_pixels

    @property
    def _p_absorbed(self) -> FloatVector: 
        return self.spectrum._p_absorbed

    @property
    def x0(self) -> float: 
        return self.spectrum.x0

    @property
    def y0(self) -> float: 
        return self.spectrum.y0

    @property
    def _x_log(self) -> FloatVector: 
        return self.spectrum._x_log

    @property
    def _y_log(self) -> FloatVector: 
        return self.spectrum._y_log

    @property
    def _dy_log(self) -> FloatVector: 
        return self.spectrum._dy_log

    @property
    def info(self) -> Info: 
        return self.spectrum.info