from logging import getLogger
from typing import Iterable, Self, Callable
from numpy import zeros_like, bool_
from dataclasses import dataclass
from pydantic import validate_call

from quasar_typing.numpy import FloatVector, BoolVector
from quasar_typing.bounds import CoordBounds
from quasar_typing.misc.coords_tuple import CoordsTuple

from quasar_utils.setup import Info

from .specdata import SpecData
from ._specdata import _SpecData

logger = getLogger(__name__)
logger.disabled = not getLogger().hasHandlers()

@dataclass
class SpecList(SpecData, list[SpecData]):
    @validate_call(validate_return=False)
    def __init__(
        self,
        coords_or_spectrum: CoordsTuple | _SpecData,
        *,
        windows: Iterable[CoordBounds] | None = None,
        x_bounds: CoordBounds | None = None,
        info: Info = None,
        y_smooth: FloatVector | None = None,
        y_pl: FloatVector | None = None,
        y_fe: FloatVector | None = None,
        y_ba: FloatVector | None = None,
        y_hg: FloatVector | None = None,
        y_em: FloatVector | None = None,
        rejected_pixels: BoolVector | None = None,
        absorbed_pixels: BoolVector | None = None,
        valid_pixels: BoolVector | None = None,
        log_valid_pixels: BoolVector | None = None,
        p_absorbed: FloatVector | None = None,
        x0: float | None = None,
        y0: float | None = None,
        x_log: FloatVector | None = None,
        y_log: FloatVector | None = None,
        dy_log: FloatVector | None = None,
        get_mask: Callable[[float, float], BoolVector] | None = None,
    ):
        super().__init__.__wrapped__(
            self,
            coords_or_spectrum,
            x_bounds=x_bounds,
            info=info,
            y_smooth=y_smooth,
            y_pl=y_pl,
            y_fe=y_fe,
            y_ba=y_ba,
            y_hg=y_hg,
            y_em=y_em,
            rejected_pixels=rejected_pixels,
            absorbed_pixels=absorbed_pixels,
            valid_pixels=valid_pixels,
            log_valid_pixels=log_valid_pixels,
            p_absorbed=p_absorbed,
            x0=x0,
            y0=y0,
            x_log=x_log,
            y_log=y_log,
            dy_log=dy_log,
            get_mask=get_mask,
        )
        self.populate.__wrapped__(self, windows=windows)

    @validate_call(validate_return=False)
    def populate(
        self,
        *,
        windows: Iterable[CoordBounds] | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **

        This method does nothing.
        """
        return self

    def __str__(self, simple: bool = False) -> str:

        s = f"'{self.__class__.__name__}' class [" \
            + ' | '.join([
                "{:.1f} <-> {:.1f}".format(*window.x_bounds) \
                for window in self
            ]) \
            + ']'
        
        if not simple:
            s += " w/ {}/{n} (rej.) {}/{n} (abs.) {}/{n} " \
                "(val.) {}/{n} (log-val.)" \
                    .format(
                        self.rejected_pixels.sum(), 
                        self.absorbed_pixels.sum(), 
                        self.valid_pixels.sum(), 
                        self.log_valid_pixels.sum(), 
                        n = self.size,
                    )

        return s + '.'
    
    @property
    def is_empty(self) -> bool:
        return len(self) == 0
    
    @property
    def mask(self) -> BoolVector:
        mask = zeros_like(self._x, dtype=bool_)
        for window in self:
            mask |= window.mask
        return mask
    
    @property
    def window_bounds(self) -> list[CoordBounds]:
        return [window.x_bounds for window in self]
    
    @property
    def window_sizes(self) -> list[int]:
        return [window.size for window in self]

