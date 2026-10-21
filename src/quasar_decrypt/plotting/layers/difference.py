from dataclasses import dataclass, field

from matplotlib.collections import FillBetweenPolyCollection
from matplotlib.lines import Line2D
from numpy import nan, where


@dataclass
class DifferenceLayer:
    n_sigma: float = field(default=1.0, kw_only=True)
    downsample: int = field(default=1, kw_only=True)

    _difference_line: Line2D | None = field(default=None, init=False)
    _error_fill_1: FillBetweenPolyCollection | None = field(default=None, init=False)

    def plot_differences(self) -> None:
        sel = slice(None, None, self.downsample)

        if self.color:
            self._difference_line = self[1].step(
                self.x[sel], 
                where(self.valid_pixels[sel], self.y[sel] - self.f[sel], nan),
                color=self.color,
                linestyle='solid',
                where='mid',
                zorder=1,
                label="Difference",
            )[0]

        if self.e_color:
            _dy = where(self.valid_pixels, self.dy, nan)
            self._error_fill_1 = self[1].fill_between(
                self.x, 
                -self.n_sigma * _dy, 
                self.n_sigma * _dy,
                color=self.e_color,
                step='mid',
                zorder=0,
                label=rf"$\pm{self.n_sigma:.1f}\sigma$",
            )