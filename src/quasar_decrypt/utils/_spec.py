__all__ = ["_Spec"]

from dataclasses import field
from logging import getLogger
from typing import Any, Self, TypedDict, TypeVar

from numpy import (
    array_equal,
    bool_,
    float64,
    full,
    inf,
    isfinite,
    nan,
    ones,
    zeros,
)
from pydantic.dataclasses import dataclass
from quasar_typing.bounds import CoordBounds
from quasar_typing.numpy import (
    Array_,
    BoolArray,
    BoolVector,
    FloatArray,
    FloatMatrix,
    FloatVector,
)
from quasar_typing.pathlib import AbsoluteFilePath
from quasar_typing.scipy import csr_matrix_
from quasar_utils.decorators import validate_call
from quasar_utils.setup import Info

from ._basespec import _BaseSpec
from .utils import CachedGetMask, get_log

logger = getLogger(__name__)     


def _asreadonlyarray(arr: FloatArray) -> FloatArray:
    """
    Convert a numpy array to a contiguous, read-only contiguous array.
    """
    arr = arr.astype(float64, order="C", copy=True)
    arr.setflags(write=False)
    return arr

def _asfcontig(arr: Array_) -> FloatArray:
    return arr.astype(float64, order="C", copy=True)

def _asbcontig(arr: Array_) -> BoolArray:
    return arr.astype(bool_, order="C", copy=True)


N = TypeVar("N", bound=int)
O = TypeVar("O", bound=int)

class _SpecKwargs[N, O](TypedDict):
    _x: FloatVector[N]
    _y: FloatVector[N]
    _dy: FloatVector[N]
    _dx: FloatVector[N]
    _kernels: FloatMatrix | csr_matrix_

    _x_original: FloatVector[O]
    _y_original: FloatVector[O]
    _dy_original: FloatVector[O]
    _dx_original: FloatVector[O]
    _kernels_original: FloatMatrix | csr_matrix_

    _y_smooth: FloatVector[N]
    _y_pl: FloatVector[N]
    _y_fe: FloatVector[N]
    _y_ba: FloatVector[N]
    _y_hg: FloatVector[N]
    _y_em: FloatVector[N]

    _y_smooth_original: FloatVector[O]
    _y_pl_original: FloatVector[O]
    _y_fe_original: FloatVector[O]
    _y_ba_original: FloatVector[O]
    _y_hg_original: FloatVector[O]
    _y_em_original: FloatVector[O]

    _rejected_pixels: BoolVector[N]
    _absorbed_pixels: BoolVector[N]
    _valid_pixels: BoolVector[N]
    _log_valid_pixels: BoolVector[N]
    _p_absorbed: FloatVector[N]

    _rejected_pixels_original: BoolVector[O]
    _absorbed_pixels_original: BoolVector[O]
    _valid_pixels_original: BoolVector[O]
    _log_valid_pixels_original: BoolVector[O]
    _p_absorbed_original: FloatVector[O]

    x0: float
    y0: float

    _x_log: FloatVector[N]
    _y_log: FloatVector[N]
    _dy_log: FloatVector[N]

    _x_log_original: FloatVector[O]
    _y_log_original: FloatVector[O]
    _dy_log_original: FloatVector[O]

    info: Info
    get_mask: CachedGetMask
    get_mask_original: CachedGetMask


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
    _kernels: FloatMatrix | csr_matrix_ = field(kw_only=True)

    _x_original: FloatVector = field(kw_only=True)
    _y_original: FloatVector = field(kw_only=True)
    _dy_original: FloatVector = field(kw_only=True)
    _dx_original: FloatVector = field(kw_only=True)
    _kernels_original: FloatMatrix | csr_matrix_ = field(kw_only=True)

    _y_smooth: FloatVector = field(kw_only=True)
    _y_pl: FloatVector = field(kw_only=True)
    _y_fe: FloatVector = field(kw_only=True)
    _y_ba: FloatVector = field(kw_only=True)
    _y_hg: FloatVector = field(kw_only=True)
    _y_em: FloatVector = field(kw_only=True)

    _y_smooth_original: FloatVector = field(kw_only=True)
    _y_pl_original: FloatVector = field(kw_only=True)
    _y_fe_original: FloatVector = field(kw_only=True)
    _y_ba_original: FloatVector = field(kw_only=True)
    _y_hg_original: FloatVector = field(kw_only=True)
    _y_em_original: FloatVector = field(kw_only=True)

    _rejected_pixels: BoolVector = field(kw_only=True)
    _absorbed_pixels: BoolVector = field(kw_only=True)
    _valid_pixels: BoolVector = field(kw_only=True)
    _log_valid_pixels: BoolVector = field(kw_only=True)
    _p_absorbed: FloatVector = field(kw_only=True)

    _rejected_pixels_original: BoolVector = field(kw_only=True)
    _absorbed_pixels_original: BoolVector = field(kw_only=True)
    _valid_pixels_original: BoolVector = field(kw_only=True)
    _log_valid_pixels_original: BoolVector = field(kw_only=True)
    _p_absorbed_original: FloatVector = field(kw_only=True)

    x0: float = field(kw_only=True)
    y0: float = field(kw_only=True)

    _x_log: FloatVector = field(kw_only=True)
    _y_log: FloatVector = field(kw_only=True)
    _dy_log: FloatVector = field(kw_only=True)

    _x_log_original: FloatVector = field(kw_only=True)
    _y_log_original: FloatVector = field(kw_only=True)
    _dy_log_original: FloatVector = field(kw_only=True)

    info: Info = field(kw_only=True)
    get_mask: CachedGetMask = field(kw_only=True)
    get_mask_original: CachedGetMask = field(kw_only=True)
    _no_original: bool = field(kw_only=True)

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state['_x'] = self._x
        state['_y'] = self._y
        state['_dy'] = self._dy
        state['_dx'] = self._dx
        state['_kernels'] = self._kernels

        state['_y_smooth'] = self._y_smooth
        state['_y_pl'] = self._y_pl
        state['_y_fe'] = self._y_fe
        state['_y_ba'] = self._y_ba
        state['_y_hg'] = self._y_hg
        state['_y_em'] = self._y_em

        state['_rejected_pixels'] = self._rejected_pixels
        state['_absorbed_pixels'] = self._absorbed_pixels
        state['_valid_pixels'] = self._valid_pixels
        state['_log_valid_pixels'] = self._log_valid_pixels
        state['_p_absorbed'] = self._p_absorbed

        state['x0'] = self.x0
        state['y0'] = self.y0

        state['_x_log'] = self._x_log
        state['_y_log'] = self._y_log
        state['_dy_log'] = self._dy_log

        state['info'] = self.info
        state['_no_original'] = self._no_original
        
        if not self._no_original:
            state['_x_original'] = self._x_original
            state['_y_original'] = self._y_original
            state['_dy_original'] = self._dy_original
            state['_dx_original'] = self._dx_original
            state['_kernels_original'] = self._kernels_original

            state['_y_smooth_original'] = self._y_smooth_original
            state['_y_pl_original'] = self._y_pl_original
            state['_y_fe_original'] = self._y_fe_original
            state['_y_ba_original'] = self._y_ba_original
            state['_y_hg_original'] = self._y_hg_original
            state['_y_em_original'] = self._y_em_original

            state['_rejected_pixels_original'] = self._rejected_pixels_original
            state['_absorbed_pixels_original'] = self._absorbed_pixels_original
            state['_valid_pixels_original'] = self._valid_pixels_original
            state['_log_valid_pixels_original'] = self._log_valid_pixels_original
            state['_p_absorbed_original'] = self._p_absorbed_original

            state['_x_log_original'] = self._x_log_original
            state['_y_log_original'] = self._y_log_original
            state['_dy_log_original'] = self._dy_log_original

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)

        self._no_original = state.pop("_no_original")

        self._x = state.pop("_x")
        self._y = state.pop("_y")
        self._dy = state.pop("_dy")
        self._dx = state.pop("_dx")
        self._kernels = state.pop("_kernels")

        self._y_smooth = state.pop("_y_smooth")
        self._y_pl = state.pop("_y_pl")
        self._y_fe = state.pop("_y_fe")
        self._y_ba = state.pop("_y_ba")
        self._y_hg = state.pop("_y_hg")
        self._y_em = state.pop("_y_em")
        
        self._rejected_pixels = state.pop("_rejected_pixels")
        self._absorbed_pixels = state.pop("_absorbed_pixels")
        self._valid_pixels = state.pop("_valid_pixels")
        self._log_valid_pixels = state.pop("_log_valid_pixels")
        self._p_absorbed = state.pop("_p_absorbed")

        self.x0 = state.pop("x0")
        self.y0 = state.pop("y0")

        self._x_log = state.pop("_x_log")
        self._y_log = state.pop("_y_log")
        self._dy_log = state.pop("_dy_log")
        
        self.info = state.pop('info')

        if self._no_original:
            self._x_original = self._x
            self._y_original = self._y
            self._dy_original = self._dy
            self._dx_original = self._dx
            self._kernels_original = self._kernels
            
            self._y_smooth_original = self._y_smooth
            self._y_pl_original = self._y_pl
            self._y_fe_original = self._y_fe
            self._y_ba_original = self._y_ba
            self._y_hg_original = self._y_hg
            self._y_em_original = self._y_em
            
            self._rejected_pixels_original = self._rejected_pixels
            self._absorbed_pixels_original = self._absorbed_pixels
            self._valid_pixels_original = self._valid_pixels
            self._log_valid_pixels_original = self._log_valid_pixels
            self._p_absorbed_original = self._p_absorbed
            
            self._x_log_original = self._x_log
            self._y_log_original = self._y_log
            self._dy_log_original = self._dy_log
        else:
            self._x_original = state.pop("_x_original")
            self._y_original = state.pop("_y_original")
            self._dy_original = state.pop("_dy_original")
            self._dx_original = state.pop("_dx_original")
            self._kernels_original = state.pop("_kernels_original")

            self._y_smooth_original = state.pop("_y_smooth_original", None)
            self._y_pl_original = state.pop("_y_pl_original", None)
            self._y_fe_original = state.pop("_y_fe_original", None)
            self._y_ba_original = state.pop("_y_ba_original", None)
            self._y_hg_original = state.pop("_y_hg_original", None)
            self._y_em_original = state.pop("_y_em_original", None)

            self._rejected_pixels_original = state.pop("_rejected_pixels_original", None)
            self._absorbed_pixels_original = state.pop("_absorbed_pixels_original", None)
            self._valid_pixels_original = state.pop("_valid_pixels_original", None)
            self._log_valid_pixels_original = state.pop("_log_valid_pixels_original", None)
            self._p_absorbed_original = state.pop("_p_absorbed_original", None)

            self._x_log_original = state.pop("_x_log_original")
            self._y_log_original = state.pop("_y_log_original")
            self._dy_log_original = state.pop("_dy_log_original")

    def _finalize_unpickling(self) -> None:
        """
        Does nothing (for now).
        """
        if getattr(self, "get_mask", None) is None:
            self.get_mask = CachedGetMask.create(self._x)

        if self._no_original:
            self.get_mask_original = self.get_mask
        elif getattr(self, "get_mask_original", None) is None:
            self.get_mask_original = CachedGetMask.create(self._x_original)

        self._validate_fields()

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
        dx: FloatVector,

        x_original: FloatVector,
        y_original: FloatVector,
        dy_original: FloatVector,
        dx_original: FloatVector,

        kernels: FloatMatrix | csr_matrix_,
        kernels_original: FloatMatrix | csr_matrix_,

        y_smooth: FloatVector | None = None,
        y_pl: FloatVector | None = None,
        y_fe: FloatVector | None = None,
        y_ba: FloatVector | None = None,
        y_hg: FloatVector | None = None,
        y_em: FloatVector | None = None,

        y_smooth_original: FloatVector | None = None,
        y_pl_original: FloatVector | None = None,
        y_fe_original: FloatVector | None = None,
        y_ba_original: FloatVector | None = None,
        y_hg_original: FloatVector | None = None,
        y_em_original: FloatVector | None = None,

        rejected_pixels: BoolVector | None = None,
        absorbed_pixels: BoolVector | None = None,
        valid_pixels: BoolVector | None = None,
        log_valid_pixels: BoolVector | None = None,
        p_absorbed: FloatVector | None = None,

        rejected_pixels_original: BoolVector | None = None,
        absorbed_pixels_original: BoolVector | None = None,
        valid_pixels_original: BoolVector | None = None,
        log_valid_pixels_original: BoolVector | None = None,
        p_absorbed_original: FloatVector | None = None,

        x0: float | None = None,
        y0: float | None = None,

        x_log: FloatVector | None = None,
        y_log: FloatVector | None = None,
        dy_log: FloatVector | None = None,

        x_log_original: FloatVector | None = None,
        y_log_original: FloatVector | None = None,
        dy_log_original: FloatVector | None = None,

        info: Info = None,
        x_bounds: CoordBounds | None = None,
        get_mask: CachedGetMask | None = None,
        get_mask_original: CachedGetMask | None = None,
    ) -> Self:
        if array_equal(x, x_original):
            kwargs = cls.get_kwargs_simple.__wrapped__(
                cls,
                path=path,
                title=title,

                x=x,
                y=y,
                dy=dy,
                dx=dx,
                kernels=kernels,

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
            kwargs["_no_original"] = True
        else:
            kwargs = cls.get_kwargs.__wrapped__(
                cls,
                path=path,
                title=title,

                x=x,
                y=y,
                dy=dy,
                dx=dx,
                x_original=x_original,
                y_original=y_original,
                dy_original=dy_original,
                dx_original=dx_original,

                kernels=kernels,
                kernels_original=kernels_original,

                y_smooth=y_smooth,
                y_pl=y_pl,
                y_fe=y_fe,
                y_ba=y_ba,
                y_hg=y_hg,
                y_em=y_em,

                y_smooth_original=y_smooth_original,
                y_pl_original=y_pl_original,
                y_fe_original=y_fe_original,
                y_ba_original=y_ba_original,
                y_hg_original=y_hg_original,
                y_em_original=y_em_original,

                rejected_pixels=rejected_pixels,
                absorbed_pixels=absorbed_pixels,
                valid_pixels=valid_pixels,
                log_valid_pixels=log_valid_pixels,
                p_absorbed=p_absorbed,

                rejected_pixels_original=rejected_pixels_original,
                absorbed_pixels_original=absorbed_pixels_original,
                valid_pixels_original=valid_pixels_original,
                log_valid_pixels_original=log_valid_pixels_original,
                p_absorbed_original=p_absorbed_original,

                x0=x0,
                y0=y0,

                x_log=x_log,
                y_log=y_log,
                dy_log=dy_log,

                x_log_original=x_log_original,
                y_log_original=y_log_original,
                dy_log_original=dy_log_original,

                x_bounds=x_bounds,
                info=info,
                get_mask=get_mask,
                get_mask_original=get_mask_original,
            )
            kwargs["_no_original"] = False
            
        spec = cls(**kwargs)
        spec._validate_fields()
        return spec

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
        dx: FloatVector,
        x_original: FloatVector,
        y_original: FloatVector,
        dy_original: FloatVector,
        dx_original: FloatVector,

        kernels: FloatMatrix | csr_matrix_,
        kernels_original: FloatMatrix | csr_matrix_,

        y_smooth: FloatVector | None = None,
        y_pl: FloatVector | None = None,
        y_fe: FloatVector | None = None,
        y_ba: FloatVector | None = None,
        y_hg: FloatVector | None = None,
        y_em: FloatVector | None = None,

        y_smooth_original: FloatVector | None = None,
        y_pl_original: FloatVector | None = None,
        y_fe_original: FloatVector | None = None,
        y_ba_original: FloatVector | None = None,
        y_hg_original: FloatVector | None = None,
        y_em_original: FloatVector | None = None,

        rejected_pixels: BoolVector | None = None,
        absorbed_pixels: BoolVector | None = None,
        valid_pixels: BoolVector | None = None,
        log_valid_pixels: BoolVector | None = None,
        p_absorbed: FloatVector | None = None,

        rejected_pixels_original: BoolVector | None = None,
        absorbed_pixels_original: BoolVector | None = None,
        valid_pixels_original: BoolVector | None = None,
        log_valid_pixels_original: BoolVector | None = None,
        p_absorbed_original: FloatVector | None = None,

        x0: float | None = None,
        y0: float | None = None,

        x_log: FloatVector | None = None,
        y_log: FloatVector | None = None,
        dy_log: FloatVector | None = None,

        x_log_original: FloatVector | None = None,
        y_log_original: FloatVector | None = None,
        dy_log_original: FloatVector | None = None,

        info: Info = None,
        x_bounds: CoordBounds | None = None,
        get_mask: CachedGetMask | None = None,
        get_mask_original: CachedGetMask | None = None,
    ) -> _SpecKwargs:
        """
        Creates the keyword arguments for instantiating a _Spec object. 

        Notes
        -----
        This method ensures that all output arrays are C-contiguous and of the 
        correct data type, and it therefore creates copies of the input arrays.
        """
        n = x.size
        n_original = x_original.size

        kwargs = {
            "path": path,
            "title": title,
        }
        kwargs["_x"] = x = _asreadonlyarray(x)
        kwargs["_y"] = y = _asreadonlyarray(y)
        kwargs["_dy"] = dy = _asreadonlyarray(dy)
        kwargs["_dx"] = dx = _asreadonlyarray(dx)
        kwargs["_x_original"] = x_original = _asreadonlyarray(x_original)
        kwargs["_y_original"] = y_original = _asreadonlyarray(y_original)
        kwargs["_dy_original"] = dy_original = _asreadonlyarray(dy_original)
        kwargs["_dx_original"] = dx_original = _asreadonlyarray(dx_original)

        kwargs["_kernels_original"] = kernels_original
        kwargs["_kernels"] = kernels

        kwargs['_y_smooth'] = \
            full(n, nan, dtype=float64, order="C") \
            if y_smooth is None \
            else _asfcontig(y_smooth)
        kwargs["_y_pl"] = \
            zeros(n, dtype=float64, order="C") \
            if y_pl is None \
            else _asfcontig(y_pl)
        kwargs["_y_fe"] = \
            zeros(n, dtype=float64, order="C") \
            if y_fe is None \
            else _asfcontig(y_fe)
        kwargs["_y_ba"] = \
            zeros(n, dtype=float64, order="C") \
            if y_ba is None \
            else _asfcontig(y_ba)
        kwargs["_y_hg"] = \
            zeros(n, dtype=float64, order="C") \
            if y_hg is None \
            else _asfcontig(y_hg)
        kwargs["_y_em"] = \
            zeros(n, dtype=float64, order="C") \
            if y_em is None \
            else _asfcontig(y_em)

        kwargs["_y_smooth_original"] = \
            full(n_original, nan, dtype=float64, order="C") \
            if y_smooth_original is None \
            else _asfcontig(y_smooth_original)
        kwargs["_y_pl_original"] = \
            zeros(n_original, dtype=float64, order="C") \
            if y_pl_original is None \
            else _asfcontig(y_pl_original)
        kwargs["_y_fe_original"] = \
            zeros(n_original, dtype=float64, order="C") \
            if y_fe_original is None \
            else _asfcontig(y_fe_original)
        kwargs["_y_ba_original"] = \
            zeros(n_original, dtype=float64, order="C") \
            if y_ba_original is None \
            else _asfcontig(y_ba_original)
        kwargs["_y_hg_original"] = \
            zeros(n_original, dtype=float64, order="C") \
            if y_hg_original is None \
            else _asfcontig(y_hg_original)
        kwargs["_y_em_original"] = \
            zeros(n_original, dtype=float64, order="C") \
            if y_em_original is None \
            else _asfcontig(y_em_original)

        kwargs["_rejected_pixels"] = \
            zeros(n, dtype=bool_, order="C") \
            if rejected_pixels is None \
            else _asbcontig(rejected_pixels)
        kwargs["_absorbed_pixels"] = \
            zeros(n, dtype=bool_, order="C") \
            if absorbed_pixels is None \
            else _asbcontig(absorbed_pixels)
        kwargs["_valid_pixels"] = valid_pixels = _asbcontig(
            isfinite(x) & isfinite(y) & (dy > 0)
            if valid_pixels is None
            else valid_pixels
        )
        kwargs["_log_valid_pixels"] = _asbcontig(
            valid_pixels & (y > 0) 
            if log_valid_pixels is None
            else log_valid_pixels
        )
        kwargs["_p_absorbed"] = \
            ones(n, dtype=float64, order="C") \
            if p_absorbed is None \
            else _asfcontig(p_absorbed)

        kwargs["_rejected_pixels_original"] = \
            zeros(n_original, dtype=bool_, order="C") \
            if rejected_pixels_original is None \
            else _asbcontig(rejected_pixels_original)
        kwargs["_absorbed_pixels_original"] = \
            zeros(n_original, dtype=bool_, order="C") \
            if absorbed_pixels_original is None \
            else _asbcontig(absorbed_pixels_original)
        kwargs["_valid_pixels_original"] = valid_pixels_original = _asbcontig(
            isfinite(x_original) & isfinite(y_original) & (dy_original > 0)
            if valid_pixels_original is None
            else valid_pixels_original
        )
        kwargs["_log_valid_pixels_original"] = _asbcontig(
            valid_pixels_original & (y_original > 0)
            if log_valid_pixels_original is None
            else log_valid_pixels_original
        )
        kwargs["_p_absorbed_original"] = \
            ones(n_original, dtype=float64, order="C") \
            if p_absorbed_original is None \
            else _asfcontig(p_absorbed_original)

        if x_bounds is None:
            n_valid = valid_pixels.sum()
            if n_valid < 2:
                msg = "No. of valid pixels is less than 2!"
                logger.critical(msg)
                x_bounds = (0, inf)
            else:
                x_bounds = (
                    x[valid_pixels][0] * (1 - info.loading.sigma_res / 2),
                    x[valid_pixels][-1] * (1 + info.loading.sigma_res / 2),
                )

        kwargs["x_bounds"] = x_bounds

        kwargs["x0"] = x0 = x0 or info.continuum.x0
        kwargs["y0"] = y0 = y0 or info.continuum.y0

        kwargs["_x_log"] = _asfcontig(
            get_log(x, x0, kwargs["_log_valid_pixels"])
            if x_log is None
            else x_log
        )
        kwargs["_y_log"] = _asfcontig(
            get_log(y, y0, kwargs["_log_valid_pixels"])
            if y_log is None
            else y_log
        )
        kwargs["_dy_log"] = _asfcontig(
            get_log(dy, y, kwargs["_log_valid_pixels"])
            if dy_log is None
            else dy_log
        )

        kwargs["_x_log_original"] = _asfcontig(
            get_log(x_original, x0, kwargs["_log_valid_pixels_original"])
            if x_log_original is None
            else x_log_original
        )
        kwargs["_y_log_original"] = _asfcontig(
            get_log(y_original, y0, kwargs["_log_valid_pixels_original"])
            if y_log_original is None
            else y_log_original
        )
        kwargs["_dy_log_original"] = _asfcontig(
            get_log(dy_original, y_original, kwargs["_log_valid_pixels_original"])
            if dy_log_original is None
            else dy_log_original
        )

        kwargs["info"] = info
        kwargs["get_mask"] = get_mask or CachedGetMask.create(x)
        kwargs["get_mask_original"] = get_mask_original or CachedGetMask.create(x_original)

        return kwargs

    @classmethod
    @validate_call
    def get_kwargs_simple(
        cls,
        *,
        path: AbsoluteFilePath,
        title: str,

        x: FloatVector,
        y: FloatVector,
        dy: FloatVector,
        dx: FloatVector,
        kernels: FloatMatrix | csr_matrix_,

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
        get_mask: CachedGetMask | None = None,
    ) -> _SpecKwargs:
        """
        Creates the keyword arguments for instantiating a _Spec object for the 
        special case that the `x` and `x_original` arrays are identical, i.e. 
        that no rebinning has been done during data loading. This is typically 
        the case for SDSS spectra that are already logarithmically binned.  

        This function uses the same logic as `get_kwargs` all `*_original` 
        arrays are identical (same objects) to their non-original counterparts.

        Notes
        -----
        This method ensures that all output arrays are C-contiguous and of the 
        correct data type, and it therefore creates copies of the input arrays.
        """
        n = x.size

        kwargs = {
            "path": path,
            "title": title,
        }
        kwargs["_x"] = x = _asreadonlyarray(x)
        kwargs["_y"] = y = _asreadonlyarray(y)
        kwargs["_dy"] = dy = _asreadonlyarray(dy)
        kwargs["_dx"] = dx = _asreadonlyarray(dx)
        kwargs["_x_original"] = x
        kwargs["_y_original"] = y
        kwargs["_dy_original"] = dy
        kwargs["_dx_original"] = dx

        kwargs["_kernels"] = kernels
        kwargs["_kernels_original"] = kernels

        kwargs['_y_smooth'] = \
            full(n, nan, dtype=float64, order="C") \
            if y_smooth is None \
            else _asfcontig(y_smooth)
        kwargs["_y_pl"] = \
            zeros(n, dtype=float64, order="C") \
            if y_pl is None \
            else _asfcontig(y_pl)
        kwargs["_y_fe"] = \
            zeros(n, dtype=float64, order="C") \
            if y_fe is None \
            else _asfcontig(y_fe)
        kwargs["_y_ba"] = \
            zeros(n, dtype=float64, order="C") \
            if y_ba is None \
            else _asfcontig(y_ba)
        kwargs["_y_hg"] = \
            zeros(n, dtype=float64, order="C") \
            if y_hg is None \
            else _asfcontig(y_hg)
        kwargs["_y_em"] = \
            zeros(n, dtype=float64, order="C") \
            if y_em is None \
            else _asfcontig(y_em)

        kwargs["_y_smooth_original"] = kwargs["_y_smooth"]
        kwargs["_y_pl_original"] = kwargs["_y_pl"]
        kwargs["_y_fe_original"] = kwargs["_y_fe"]
        kwargs["_y_ba_original"] = kwargs["_y_ba"]
        kwargs["_y_hg_original"] = kwargs["_y_hg"]
        kwargs["_y_em_original"] = kwargs["_y_em"]

        kwargs["_rejected_pixels"] = \
            zeros(n, dtype=bool_, order="C") \
            if rejected_pixels is None \
            else _asbcontig(rejected_pixels)
        kwargs["_absorbed_pixels"] = \
            zeros(n, dtype=bool_, order="C") \
            if absorbed_pixels is None \
            else _asbcontig(absorbed_pixels)
        kwargs["_valid_pixels"] = valid_pixels = _asbcontig(
            isfinite(x) & isfinite(y) & (dy > 0)
            if valid_pixels is None
            else valid_pixels
        )
        kwargs["_log_valid_pixels"] = _asbcontig(
            valid_pixels & (y > 0) 
            if log_valid_pixels is None
            else log_valid_pixels
        )
        kwargs["_p_absorbed"] = \
            ones(n, dtype=float64, order="C") \
            if p_absorbed is None \
            else _asfcontig(p_absorbed)

        kwargs["_rejected_pixels_original"] = kwargs["_rejected_pixels"]
        kwargs["_absorbed_pixels_original"] = kwargs["_absorbed_pixels"]
        kwargs["_valid_pixels_original"] = kwargs["_valid_pixels"]
        kwargs["_log_valid_pixels_original"] = kwargs["_log_valid_pixels"]
        kwargs["_p_absorbed_original"] = kwargs["_p_absorbed"]

        if x_bounds is None:
            n_valid = valid_pixels.sum()
            if n_valid < 2:
                msg = "No. of valid pixels is less than 2!"
                logger.critical(msg)
                x_bounds = (0, inf)
            else:
                x_bounds = (
                    x[valid_pixels][0] * (1 - info.loading.sigma_res / 2),
                    x[valid_pixels][-1] * (1 + info.loading.sigma_res / 2),
                )

        kwargs["x_bounds"] = x_bounds

        kwargs["x0"] = x0 = x0 or info.continuum.x0
        kwargs["y0"] = y0 = y0 or info.continuum.y0

        kwargs["_x_log"] = _asfcontig(
            get_log(x, x0, kwargs["_log_valid_pixels"])
            if x_log is None
            else x_log
        )
        kwargs["_y_log"] = _asfcontig(
            get_log(y, y0, kwargs["_log_valid_pixels"])
            if y_log is None
            else y_log
        )
        kwargs["_dy_log"] = _asfcontig(
            get_log(dy, y, kwargs["_log_valid_pixels"])
            if dy_log is None
            else dy_log
        )

        kwargs["_x_log_original"] = kwargs["_x_log"]
        kwargs["_y_log_original"] = kwargs["_y_log"]
        kwargs["_dy_log_original"] = kwargs["_dy_log"]

        kwargs["info"] = info
        kwargs["get_mask"] = get_mask or CachedGetMask.create(x)
        kwargs["get_mask_original"] = kwargs["get_mask"]

        return kwargs

    def _validate_fields(self):
        """
        Verifies that all fields are initialized properly, i.e. if 
        `_no_original` is True, 'original' and non-'original' fields should be 
        identical.
        """
        def inner(k: str):
            l = k + "_original"
            f1 = getattr(self, k)
            f2 = getattr(self, l)
            if not (f1 is f2):
                a1 = hex(id(f1))
                a2 = hex(id(f2))
                raise ValueError(
                    f"Fields `{k}` and `{l}` are not identical: {a1} != {a2}"
                )
            
        if self._no_original:
            inner("_x")
            inner("_y")
            inner("_dy")
            inner("_dx")
            inner("_kernels")
            inner("_rejected_pixels")
            inner("_absorbed_pixels")
            inner("_valid_pixels")
            inner("_log_valid_pixels")
            inner("_p_absorbed")
            inner("_x_log")
            inner("_y_log")
            inner("_dy_log")
            inner("get_mask")
