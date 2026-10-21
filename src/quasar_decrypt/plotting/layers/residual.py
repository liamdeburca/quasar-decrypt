from dataclasses import dataclass, field

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from numpy import float64, full_like, nan


@dataclass
class ResidualLayer:
    n_sigma: float = field(default=2.0, kw_only=True)
    downsample: int = field(default=1, kw_only=True)
    
    _residual_line: Line2D | None = field(default=None, init=False)
    _error_fill_2: Rectangle | None = field(default=None, init=False)

    def plot_residuals(self) -> None:
        sel = slice(None, None, self.downsample)

        if self.color:
            mask = self.valid_pixels & (self.dy > 0)
            z = full_like(self.valid_pixels, nan, dtype=float64)
            z[mask] = (self.y[mask] - self.f[mask]) / self.dy[mask]

            self._residual_line = self[1].step(
                self.x[sel], 
                z[sel],
                color=self.color,
                linestyle='solid',
                where='mid',
                zorder=1,
                label="Residual",
            )[0]
        if self.e_color:
            self._error_fill_2 = self[1].axhspan(
                -self.n_sigma, 
                self.n_sigma,
                color=self.e_color,
                zorder=0,
                label=rf"$\pm{self.n_sigma:.1f}\sigma$",
            )