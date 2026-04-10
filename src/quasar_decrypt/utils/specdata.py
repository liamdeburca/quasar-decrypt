from logging import getLogger
from typing import Callable
from dataclasses import dataclass, field
from pydantic import validate_call

from quasar_typing.numpy import FloatVector, BoolVector
from quasar_typing.bounds import CoordBounds
from quasar_typing.misc.coords_tuple import CoordsTuple

from quasar_utils.setup import Info

from ._specdata import _SpecData

logger = getLogger(__name__)
logger.disabled = not getLogger().hasHandlers()

@dataclass
class SpecData(_SpecData):
    """
    Parent class used for inheriting properties and methods. Designed for 
    inputting 'Spectrum' objects. 
    """
    spectrum: _SpecData | None = field(default=None, init=False)

    @validate_call(validate_return=False)
    def __init__(
        self,
        coords_or_spectrum: CoordsTuple | _SpecData,
        *,
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
        if isinstance(coords_or_spectrum, _SpecData):
            self.spectrum = spectrum = coords_or_spectrum
            super().__init__.__wrapped__(
                self,
                spectrum._coords,
                x_bounds=x_bounds,
                info=spectrum.info,
                y_smooth=spectrum._y_smooth,
                y_pl=spectrum._y_pl,
                y_fe=spectrum._y_fe,
                y_ba=spectrum._y_ba,
                y_hg=spectrum._y_hg,
                y_em=spectrum._y_em,
                rejected_pixels=spectrum._rejected_pixels,
                absorbed_pixels=spectrum._absorbed_pixels,
                valid_pixels=spectrum._valid_pixels,
                log_valid_pixels=spectrum._log_valid_pixels,
                p_absorbed=spectrum._p_absorbed,
                x0=spectrum.x0,
                y0=spectrum.y0,
                x_log=spectrum._x_log,
                y_log=spectrum._y_log,
                dy_log=spectrum._dy_log,
                get_mask=spectrum.get_mask,
            )
        else:
            coords = coords_or_spectrum
            super().__init__.__wrapped__(
                self,
                coords,
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
            