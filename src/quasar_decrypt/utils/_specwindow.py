__all__ = ["_SpecWindow"]

from dataclasses import field
from functools import wraps
from logging import getLogger
from typing import Any, ClassVar, Self

import pydantic
from quasar_typing.bounds import CoordBounds
from quasar_typing.numpy import BoolVector, FloatMatrix, FloatVector
from quasar_typing.scipy import CSRMatrix
from quasar_typing.string_selections import BackgroundFlux
from quasar_utils.setup import Info

from ._basespec import _BaseSpec
from ._spec import _Spec
from .utils import CachedGetMask

logger = getLogger(__name__)


@pydantic.dataclasses.dataclass
class _SpecWindow(_BaseSpec):
    """
    Class for storing spectral windows. 

    Has no inherent attributes, but points to a '_Spec' object using properties. 
    Modifying properties will therefore also modify the attributes of the 
    '_Spec' object.
    """
    spectrum: _Spec = field(kw_only=True)
    x_bounds: CoordBounds = field(kw_only=True)

    get_mask: CachedGetMask = field(kw_only=True)
    get_mask_original: CachedGetMask = field(kw_only=True)

    default_bg: ClassVar[BackgroundFlux]

    def _finalize_unpickling(self, spectrum: _Spec) -> None:
        """
        Set the `spectrum` attribute after unpickling.
        """
        self.spectrum = spectrum
        if getattr(self, "get_mask", None) is None:
            self.get_mask = CachedGetMask.create(self.spectrum._x)
        if getattr(self, "get_mask_original", None) is None:
            self.get_mask_original = CachedGetMask.create(self.spectrum._x_original)

    ### create

    @classmethod
    @pydantic.validate_call
    def create(
        cls,
        *,
        spectrum: _Spec,
        x_bounds: CoordBounds,
        get_mask: CachedGetMask | None = None,
        get_mask_original: CachedGetMask | None = None,
    ) -> Self:
        """
        Creates a '_SpecWindow' object from a '_Spec' object and x_bounds.
        """
        return cls._create(
            spectrum=spectrum,
            x_bounds=x_bounds,
            get_mask=get_mask,
            get_mask_original=get_mask_original,
        )

    @classmethod
    @wraps(create)
    def _create(
        cls,
        *,
        spectrum: _Spec,
        x_bounds: CoordBounds,
        get_mask: CachedGetMask | None = None,
        get_mask_original: CachedGetMask | None = None,
    ) -> Self:
        if get_mask is None:
            get_mask = CachedGetMask.create(spectrum._x)

        if spectrum._no_original:
            get_mask_original = get_mask
        elif get_mask_original is None:
            get_mask_original = CachedGetMask.create(spectrum._x_original)

        return cls(
            spectrum=spectrum,
            x_bounds=x_bounds,
            get_mask=get_mask,
            get_mask_original=get_mask_original,
        )

    @property
    def _no_original(self) -> bool:
        return self.spectrum._no_original

    @property
    def mask(self) -> BoolVector:
        """
        Return the combined mask for the spectral window and the parent 
        spectrum's mask.

        When True, the corresponding data point is within the window's and 
        parent spectrum's `x_bounds`.
        """
        return self.spectrum.mask \
            & self.get_mask(*self.x_bounds)

    @property
    def mask_original(self) -> BoolVector:
        """
        Return the combined original mask for the spectral window and the parent 
        spectrum's mask.

        When True, the corresponding data point is within the window's and 
        parent spectrum's `x_bounds`.
        """
        return self.spectrum.mask_original \
            & self.get_mask_original(*self.x_bounds)

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
    def _kernels(self) -> FloatMatrix | CSRMatrix: 
        return self.spectrum._kernels

    @property
    def _x_original(self) -> FloatVector: 
        return self.spectrum._x_original

    @property
    def _y_original(self) -> FloatVector: 
        return self.spectrum._y_original

    @property
    def _dy_original(self) -> FloatVector: 
        return self.spectrum._dy_original

    @property
    def _dx_original(self) -> FloatVector: 
        return self.spectrum._dx_original

    @property
    def _kernels_original(self) -> FloatMatrix | CSRMatrix: 
        return self.spectrum._kernels_original

    ###

    @property
    def _y_smooth(self) -> FloatVector: 
        return self.spectrum._y_smooth

    @_y_smooth.setter
    def _y_smooth(self, value: FloatVector) -> None:
        self.spectrum._y_smooth[:] = value

    @property
    def _y_pl(self) -> FloatVector: 
        return self.spectrum._y_pl

    @_y_pl.setter
    def _y_pl(self, value: FloatVector) -> None:
        self.spectrum._y_pl[:] = value

    @property
    def _y_fe(self) -> FloatVector: 
        return self.spectrum._y_fe

    @_y_fe.setter
    def _y_fe(self, value: FloatVector) -> None:
        self.spectrum._y_fe[:] = value

    @property
    def _y_ba(self) -> FloatVector: 
        return self.spectrum._y_ba

    @_y_ba.setter
    def _y_ba(self, value: FloatVector) -> None:
        self.spectrum._y_ba[:] = value

    @property
    def _y_hg(self) -> FloatVector: 
        return self.spectrum._y_hg

    @_y_hg.setter
    def _y_hg(self, value: FloatVector) -> None:
        self.spectrum._y_hg[:] = value

    @property
    def _y_em(self) -> FloatVector: 
        return self.spectrum._y_em

    @_y_em.setter
    def _y_em(self, value: FloatVector) -> None:
        self.spectrum._y_em[:] = value

    ###

    @property
    def _y_smooth_original(self) -> FloatVector: 
        return self.spectrum._y_smooth_original

    @_y_smooth_original.setter
    def _y_smooth_original(self, value: FloatVector) -> None:
        self.spectrum._y_smooth_original[:] = value

    @property
    def _y_pl_original(self) -> FloatVector: 
        return self.spectrum._y_pl_original

    @_y_pl_original.setter
    def _y_pl_original(self, value: FloatVector) -> None:
        self.spectrum._y_pl_original[:] = value

    @property
    def _y_fe_original(self) -> FloatVector: 
        return self.spectrum._y_fe_original

    @_y_fe_original.setter
    def _y_fe_original(self, value: FloatVector) -> None:
        self.spectrum._y_fe_original[:] = value

    @property
    def _y_ba_original(self) -> FloatVector: 
        return self.spectrum._y_ba_original

    @_y_ba_original.setter
    def _y_ba_original(self, value: FloatVector) -> None:
        self.spectrum._y_ba_original[:] = value

    @property
    def _y_hg_original(self) -> FloatVector: 
        return self.spectrum._y_hg_original

    @_y_hg_original.setter
    def _y_hg_original(self, value: FloatVector) -> None:
        self.spectrum._y_hg_original[:] = value

    @property
    def _y_em_original(self) -> FloatVector: 
        return self.spectrum._y_em_original

    @_y_em_original.setter
    def _y_em_original(self, value: FloatVector) -> None:
        self.spectrum._y_em_original[:] = value

    ###

    @property
    def _rejected_pixels(self) -> BoolVector: 
        return self.spectrum._rejected_pixels

    @_rejected_pixels.setter
    def _rejected_pixels(self, value: BoolVector) -> None:
        self.spectrum._rejected_pixels[:] = value

    @property
    def _absorbed_pixels(self) -> BoolVector: 
        return self.spectrum._absorbed_pixels

    @_absorbed_pixels.setter
    def _absorbed_pixels(self, value: BoolVector) -> None:
        self.spectrum._absorbed_pixels[:] = value

    @property
    def _valid_pixels(self) -> BoolVector: 
        return self.spectrum._valid_pixels

    @_valid_pixels.setter
    def _valid_pixels(self, value: BoolVector) -> None:
        self.spectrum._valid_pixels[:] = value

    @property
    def _log_valid_pixels(self) -> BoolVector: 
        return self.spectrum._log_valid_pixels

    @_log_valid_pixels.setter
    def _log_valid_pixels(self, value: BoolVector) -> None:
        self.spectrum._log_valid_pixels[:] = value

    @property
    def _p_absorbed(self) -> FloatVector: 
        return self.spectrum._p_absorbed

    @_p_absorbed.setter
    def _p_absorbed(self, value: FloatVector) -> None:
        self.spectrum._p_absorbed[:] = value

    ###

    @property
    def _rejected_pixels_original(self) -> BoolVector: 
        return self.spectrum._rejected_pixels_original

    @_rejected_pixels_original.setter
    def _rejected_pixels_original(self, value: BoolVector) -> None:
        self.spectrum._rejected_pixels_original[:] = value

    @property
    def _absorbed_pixels_original(self) -> BoolVector: 
        return self.spectrum._absorbed_pixels_original

    @_absorbed_pixels_original.setter
    def _absorbed_pixels_original(self, value: BoolVector) -> None:
        self.spectrum._absorbed_pixels_original[:] = value

    @property
    def _valid_pixels_original(self) -> BoolVector: 
        return self.spectrum._valid_pixels_original

    @_valid_pixels_original.setter
    def _valid_pixels_original(self, value: BoolVector) -> None:
        self.spectrum._valid_pixels_original[:] = value

    @property
    def _log_valid_pixels_original(self) -> BoolVector: 
        return self.spectrum._log_valid_pixels_original

    @_log_valid_pixels_original.setter
    def _log_valid_pixels_original(self, value: BoolVector) -> None:
        self.spectrum._log_valid_pixels_original[:] = value

    @property
    def _p_absorbed_original(self) -> FloatVector: 
        return self.spectrum._p_absorbed_original

    @_p_absorbed_original.setter
    def _p_absorbed_original(self, value: FloatVector) -> None:
        self.spectrum._p_absorbed_original[:] = value

    ###

    @property
    def x0(self) -> float: 
        return self.spectrum.x0

    @x0.setter
    def x0(self, value: float) -> None:
        self.spectrum.x0 = value

    @property
    def y0(self) -> float: 
        return self.spectrum.y0

    @y0.setter
    def y0(self, value: float) -> None:
        self.spectrum.y0 = value

    ###

    @property
    def _x_log(self) -> FloatVector: 
        return self.spectrum._x_log

    @_x_log.setter
    def _x_log(self, value: FloatVector) -> None:
        self.spectrum._x_log[:] = value

    @property
    def _y_log(self) -> FloatVector: 
        return self.spectrum._y_log

    @_y_log.setter
    def _y_log(self, value: FloatVector) -> None:
        self.spectrum._y_log[:] = value

    @property
    def _dy_log(self) -> FloatVector: 
        return self.spectrum._dy_log

    @_dy_log.setter
    def _dy_log(self, value: FloatVector) -> None:
        self.spectrum._dy_log[:] = value

    ###

    @property
    def _x_log_original(self) -> FloatVector: 
        return self.spectrum._x_log_original

    @_x_log_original.setter
    def _x_log_original(self, value: FloatVector) -> None:
        self.spectrum._x_log_original[:] = value

    @property
    def _y_log_original(self) -> FloatVector: 
        return self.spectrum._y_log_original

    @_y_log_original.setter
    def _y_log_original(self, value: FloatVector) -> None:
        self.spectrum._y_log_original[:] = value

    @property
    def _dy_log_original(self) -> FloatVector: 
        return self.spectrum._dy_log_original

    @_dy_log_original.setter
    def _dy_log_original(self, value: FloatVector) -> None:
        self.spectrum._dy_log_original[:] = value

    ###

    @property
    def info(self) -> Info: 
        return self.spectrum.info

    @info.setter
    def info(self, value: Info) -> None:
        self.spectrum.info = value

    ###

    def _plot_helper[P](
        self,
        *,
        plotter_cls: type[P],
        lazy: bool,
        args: tuple[Any, ...],
        plotter_kwargs: dict[str, Any],
    ) -> P:
        plotter = plotter_cls(*args, **plotter_kwargs)

        if "title_prefix" not in plotter_kwargs:
            plotter_kwargs["title_prefix"] = f"{self.__class__.__name__} of: "
        if "title" not in plotter_kwargs:
            plotter_kwargs["title"] = self.spectrum.title
        if "xlim" not in plotter_kwargs:
            plotter_kwargs["xlim"] = self.x_bounds
        if "original" in plotter_kwargs:
            plotter_kwargs["original"] &= not self._no_original

        return plotter if lazy else plotter.plot()