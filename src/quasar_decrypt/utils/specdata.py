from collections.abc import Callable
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Self

from quasar_typing.bounds import CoordBounds
from quasar_typing.numpy import BoolVector, FloatVector
from quasar_utils.decorators import validate_call
from quasar_utils.setup import Info

from ._specdata import _SpecData

logger = getLogger(__name__)


@dataclass
class SpecData(_SpecData):
    """
    Parent class used for inheriting properties and methods. Designed for
    inputting 'Spectrum' objects.
    """

    spectrum: _SpecData | None = field(default=None, kw_only=True)

    @classmethod
    @validate_call
    def create(
        cls,
        *,
        spectrum: _SpecData | None = None,
        x: FloatVector | None = None,
        y: FloatVector | None = None,
        dy: FloatVector | None = None,
        dx: FloatVector | None = None,
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
        x_bounds: CoordBounds | None = None,
        info: Info = None,
        get_mask: Callable[[float, float], BoolVector] | None = None,
    ) -> Self:
        kwargs = cls.get_kwargs.__wrapped__(
            cls,
            spectrum=spectrum,
            x=x,
            y=y,
            dy=dy,
            dx=dx,
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
            x_bounds=x_bounds,
            info=info,
            get_mask=get_mask,
        )
        return cls(**kwargs)

    @classmethod
    @validate_call
    def get_kwargs(
        cls,
        *,
        spectrum: _SpecData | None = None,
        x: FloatVector | None = None,
        y: FloatVector | None = None,
        dy: FloatVector | None = None,
        dx: FloatVector | None = None,
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
        x_bounds: CoordBounds | None = None,
        info: Info | None = None,
        get_mask: Callable[[float, float], BoolVector] | None = None,
    ) -> dict[str, Any]:
        kwargs = {}

        if spectrum is None:
            if any(arr is None for arr in (x, y, dy)):
                raise ValueError(
                    "All arrays ('x', 'y', 'dy') must be provided if "
                    "'spectrum' is not given."
                )
            if info is None:
                raise ValueError(
                    "The 'info' argument must be provided if 'spectrum' "
                    "is not given."
                )

            kwargs["spectrum"] = None
            kwargs.update(super().get_kwargs.__wrapped__(
                cls,
                x=x,
                y=y,
                dy=dy,
                dx=dx,
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
                x_bounds=x_bounds,
                info=info,
                get_mask=get_mask,
            ))
        else:
            kwargs["spectrum"] = spectrum
            kwargs.update(super().get_kwargs_from_specdata.__wrapped__(
                cls,
                spectrum,
                x_bounds=x_bounds,
                info=info,
            ))
        return kwargs
