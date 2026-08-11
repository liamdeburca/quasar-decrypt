__all__ = ["_Spec"]

from collections.abc import Callable
from dataclasses import field
from logging import getLogger
from typing import Any, Self

from numpy import (
    ascontiguousarray,
    bool_,
    float64,
    inf,
    isfinite,
    ones_like,
    zeros_like,
)
from pydantic.dataclasses import dataclass
from quasar_typing.bounds import CoordBounds
from quasar_typing.numpy import BoolVector, FloatVector
from quasar_typing.pathlib import AbsoluteFilePath
from quasar_utils.decorators import validate_call
from quasar_utils.setup import Info

from ._basespec import _BaseSpec
from .utils import create_cached_get_mask, get_log

logger = getLogger(__name__)        

@dataclass
class _Spec(_BaseSpec):
    """
    Parent class used for inheriting properties and methods. Designed for
    inputting arrays.
    """

    _x: FloatVector = field(kw_only=True)
    _y: FloatVector = field(kw_only=True)
    _dy: FloatVector = field(kw_only=True)

    _dx: FloatVector = field(kw_only=True)

    _y_smooth: FloatVector = field(kw_only=True)
    _y_pl: FloatVector = field(kw_only=True)
    _y_fe: FloatVector = field(kw_only=True)
    _y_ba: FloatVector = field(kw_only=True)
    _y_hg: FloatVector = field(kw_only=True)
    _y_em: FloatVector = field(kw_only=True)

    _rejected_pixels: BoolVector = field(kw_only=True)
    _absorbed_pixels: BoolVector = field(kw_only=True)
    _valid_pixels: BoolVector = field(kw_only=True)
    _log_valid_pixels: BoolVector = field(kw_only=True)

    _p_absorbed: FloatVector = field(kw_only=True)

    x0: float = field(kw_only=True)
    y0: float = field(kw_only=True)

    _x_log: FloatVector = field(kw_only=True)
    _y_log: FloatVector = field(kw_only=True)
    _dy_log: FloatVector = field(kw_only=True)

    info: Info = field(kw_only=True)

    @classmethod
    @validate_call
    def create(
        cls,
        *,
        path: AbsoluteFilePath,
        title: str,
        x: FloatVector,
        y: FloatVector,
        dy: FloatVector,
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
        info: Info = None,
        x_bounds: CoordBounds | None = None,
        get_mask: Callable[[float, float], BoolVector] | None = None,
    ) -> Self:
        kwargs = cls.get_kwargs.__wrapped__(
            cls,
            path=path,
            title=title,
            x=x,
            y=y,
            dy=dy,
            dx=dx,
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
        return cls(**kwargs)

    @classmethod
    @validate_call
    def get_kwargs(
        cls,
        *,
        path: AbsoluteFilePath,
        title: str,
        x: FloatVector,
        y: FloatVector,
        dy: FloatVector,
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
        info: Info = None,
        x_bounds: CoordBounds | None = None,
        get_mask: Callable[[float, float], BoolVector] | None = None,
    ) -> dict[str, Any]:
        """
        Creates the keyword arguments for instantiating a _SpecData object. 

        Notes
        -----
        This method ensures that all output arrays are C-contiguous and of the 
        correct data type, and it therefore creates copies of the input arrays.
        """
        kwargs = {
            "path": path,
            "title": title,
            "_x": ascontiguousarray(x, dtype=float64),
            "_y": ascontiguousarray(y, dtype=float64),
            "_dy": ascontiguousarray(dy, dtype=float64),
        }

        kwargs['_dx'] = ascontiguousarray(
            x * info.loading.sigma_res if dx is None else dx,
            dtype=float64,
        )
        kwargs['_y_smooth'] = ascontiguousarray(
            y if y_smooth is None else y_smooth,
            dtype=float64,
        )

        kwargs["_y_pl"] = (
            zeros_like(kwargs["_x"], dtype=float64, order="C")
            if y_pl is None
            else ascontiguousarray(y_pl, dtype=float64)
        )
        kwargs["_y_fe"] = (
            zeros_like(kwargs["_x"], dtype=float64, order="C")
            if y_fe is None
            else ascontiguousarray(y_fe, dtype=float64)
        )
        kwargs["_y_ba"] = (
            zeros_like(kwargs["_x"], dtype=float64, order="C")
            if y_ba is None
            else ascontiguousarray(y_ba, dtype=float64)
        )
        kwargs["_y_hg"] = (
            zeros_like(kwargs["_x"], dtype=float64, order="C")
            if y_hg is None
            else ascontiguousarray(y_hg, dtype=float64)
        )
        kwargs["_y_em"] = (
            zeros_like(kwargs["_x"], dtype=float64, order="C")
            if y_em is None
            else ascontiguousarray(y_em, dtype=float64)
        )

        kwargs["_rejected_pixels"] = (
            zeros_like(kwargs["_x"], dtype=bool_, order="C")
            if rejected_pixels is None
            else ascontiguousarray(rejected_pixels, dtype=bool_)
        )
        kwargs["_absorbed_pixels"] = (
            zeros_like(kwargs["_x"], dtype=bool_, order="C")
            if absorbed_pixels is None
            else ascontiguousarray(absorbed_pixels, dtype=bool_)
        )
        kwargs["_valid_pixels"] = ascontiguousarray(
            isfinite(kwargs["_x"])
            & isfinite(kwargs["_y"])
            & isfinite(kwargs["_dy"])
            & (kwargs["_dy"] > 0)
            if valid_pixels is None
            else valid_pixels,
            dtype=bool_,
        )
        kwargs["_log_valid_pixels"] = ascontiguousarray(
            kwargs["_valid_pixels"] & (kwargs["_y"] > 0)
            if log_valid_pixels is None
            else log_valid_pixels,
            dtype=bool_,
        )

        kwargs["_p_absorbed"] = (
            ones_like(kwargs["_x"], dtype=float64, order="C")
            if p_absorbed is None
            else ascontiguousarray(p_absorbed, dtype=float64)
        )

        if x_bounds is None:
            mask = kwargs["_valid_pixels"]
            n_valid = mask.sum()
            if n_valid < 2:
                msg = "No. of valid pixels is less than 2!"
                logger.critical(msg)
                x_bounds = (0, inf)
            else:
                x = kwargs["_x"][mask]
                x_bounds = (
                    x[0] * (1 - info.loading.sigma_res / 2),
                    x[-1] * (1 + info.loading.sigma_res / 2),
                )

        kwargs["x_bounds"] = x_bounds

        kwargs["x0"] = x0 or info.continuum.x0
        kwargs["y0"] = y0 or info.continuum.y0

        kwargs["_x_log"] = ascontiguousarray(
            get_log(kwargs["_x"], kwargs["x0"], kwargs["_log_valid_pixels"])
            if x_log is None
            else x_log,
            dtype=float64,
        )
        kwargs["_y_log"] = ascontiguousarray(
            get_log(kwargs["_y"], kwargs["y0"], kwargs["_log_valid_pixels"])
            if y_log is None
            else y_log,
            dtype=float64,
        )
        kwargs["_dy_log"] = ascontiguousarray(
            get_log(kwargs["_dy"], kwargs["_y"], kwargs["_log_valid_pixels"])
            if dy_log is None
            else dy_log,
            dtype=float64,
        )

        kwargs["info"] = info

        kwargs["get_mask"] = (
            create_cached_get_mask(kwargs["_x"], maxsize=1)
            if get_mask is None
            else get_mask
        )

        return kwargs