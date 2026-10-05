from dataclasses import dataclass, field

from matplotlib.patches import Rectangle
from numpy import bool_, zeros_like
from quasar_typing.numpy import BoolVector

from ..defaults import DEFAULT_COLORS


@dataclass
class WindowsLayer:
    """
    Dataclass responsible for visualizing window intervals.
    """
    windows: list[tuple[float, float]]

    w_color: str | None = field(
        default_factory=lambda: DEFAULT_COLORS['w'],
        kw_only=True,
    )
    window_patches: list[Rectangle] = field(
        default_factory=list,
        init=False,
    )

    @property
    def mask(self) -> BoolVector:
        mask = zeros_like(self._x, dtype=bool_)
        for lb, ub in self.windows:
            mask |= (lb <= self._x) & (self._x <= ub)
        return mask

    def plot_windows(self) -> None:
        xmin = self._x[0]
        xmax = self._x[-1]        
        for ax in self:
            for lb, ub in self.windows:
                if (ub <= xmin) or (xmax <= lb):
                    continue

                rec = ax.axvspan(
                    max(lb, xmin), 
                    min(ub, xmax),
                    color=self.w_color, 
                    zorder=-1,
                )
                self.window_patches.append(rec)