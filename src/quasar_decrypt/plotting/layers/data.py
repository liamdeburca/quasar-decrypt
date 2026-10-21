from dataclasses import dataclass, field

from matplotlib.collections import FillBetweenPolyCollection
from matplotlib.lines import Line2D
from numpy import nan, where


@dataclass
class DataLayer:
    """
    Dataclass responsible for visualising the spectral data.
    """
    n_sigma: float = field(default=1.0, kw_only=True)
    downsample: int = field(default=1, kw_only=True)

    _data_line: Line2D | None = field(default=None, init=False)
    _fit_line: Line2D | None = field(default=None, init=False)
    _ab_line: Line2D | None = field(default=None, init=False)
    _error_fill_0: FillBetweenPolyCollection | None = field(default=None, init=False)

    def plot_data(self) -> None:
        sel = slice(None, None, self.downsample)

        if self.color:
            self._data_line = self[0].step(
                self.x[sel], 
                self.y[sel], 
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
                0.0,
                self.n_sigma * self.dy,
                color=self.e_color, 
                step='mid', 
                zorder=0,
                label=rf"$\pm{self.n_sigma:.1f}\sigma$",
            )