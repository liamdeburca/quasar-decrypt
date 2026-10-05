from dataclasses import dataclass, field

from matplotlib.collections import FillBetweenPolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from numpy import float64, full_like, nan, where


@dataclass
class DataLayer:
    """
    Dataclass responsible for visualising the spectral data.
    """
    n_sigma: float = field(default=2.0, kw_only=True)
    _data_line: Line2D | None = field(default=None, init=False)
    _fit_line: Line2D | None = field(default=None, init=False)
    _ab_line: Line2D | None = field(default=None, init=False)
    _error_fill_0: FillBetweenPolyCollection | None = field(default=None, init=False)

    def plot_data(self) -> None:
        if self.color:
            self._data_line = self[0].step(
                self.x, 
                self.y, 
                color=self.color, 
                linestyle='solid',
                where='mid', 
                zorder=1,
                label="Data",
            )[0]

        if self.f_color:
            self._fit_line = self[0].step(
                self.x, 
                self.f, 
                color=self.f_color, 
                linestyle='solid',
                where='mid', 
                zorder=2,
                label="Model",
            )[0]

        if self.ab_color and self._n_abs > 0:
            self._ab_line = self[0].step(
                self.x,
                where(self.absorbed_pixels, self.y, nan),
                color=self.ab_color,
                linestyle='solid',
                where='mid',
                zorder=1,
                label="Abs.",
            )[0]

        if self.e_color:
            self._error_fill_0 = self[0].fill_between(
                self.x, 
                self.y - self.n_sigma * self.dy, 
                self.y + self.n_sigma * self.dy, 
                color=self.e_color, 
                step='mid', 
                zorder=0,
                label=rf"$\pm{self.n_sigma:.1f}\sigma$",
            )

@dataclass
class DifferenceLayer:
    n_sigma: float = field(default=2.0, kw_only=True)

    _difference_line: Line2D | None = field(default=None, init=False)
    _error_fill_1: FillBetweenPolyCollection | None = field(default=None, init=False)

    def plot_differences(self) -> None:
        if self.color:
            _d = where(self.valid_pixels, self.y - self.f, nan) 
            self._difference_line = self[1].step(
                self.x, _d,
                color=self.color,
                linestyle='solid',
                where='mid',
                zorder=1,
                label="Difference",
            )[0]
        if self.e_color:
            _dy = where(self.valid_pixels, self.dy, nan)
            self._error_fill_1 = self[1].fill_between(
                self.x, -self.n_sigma * _dy, self.n_sigma * _dy,
                color=self.e_color,
                step='mid',
                zorder=0,
                label=rf"$\pm{self.n_sigma:.1f}\sigma$",
            )

@dataclass
class ResidualLayer:
    n_sigma: float = field(default=2.0, kw_only=True)
    
    _residual_line: Line2D | None = field(default=None, init=False)
    _error_fill_2: Rectangle | None = field(default=None, init=False)

    def plot_residuals(self) -> None:
        if self.color:
            mask = self.valid_pixels & (self.dy > 0)
            z = full_like(self.valid_pixels, nan, dtype=float64)
            z[mask] = (self.y[mask] - self.f[mask]) / self.dy[mask]

            self._residual_line = self[1].step(
                self.x, z,
                color=self.color,
                linestyle='solid',
                where='mid',
                zorder=1,
                label="Residual",
            )[0]
        if self.e_color:
            self._error_fill_2 = self[1].axhspan(
                -self.n_sigma, self.n_sigma,
                color=self.e_color,
                zorder=0,
                label=rf"$\pm{self.n_sigma:.1f}\sigma$",
            )