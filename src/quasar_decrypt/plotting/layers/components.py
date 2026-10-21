from dataclasses import dataclass, field

from matplotlib.lines import Line2D
from numpy import isfinite


@dataclass
class ComponentsLayer:
    """
    Dataclass responsible for visualizing the components of a spectrum.
    """
    _sm_line: Line2D | None = field(default=None, init=False)
    _pl_line: Line2D | None = field(default=None, init=False)
    _fe_line: Line2D | None = field(default=None, init=False)
    _ba_line: Line2D | None = field(default=None, init=False)
    _hg_line: Line2D | None = field(default=None, init=False)
    _em_line: Line2D | None = field(default=None, init=False)

    def plot_components(self) -> None:
        if self.sm_color and isfinite(self.y_smooth).any():
            self._sm_line = self[0].step(
                self.x, 
                self.y_smooth, 
                color=self.sm_color, 
                linestyle=self.sm_linestyle,
                where='mid', 
                zorder=1,
                label="Sm.",
            )[0]

        if self.pl_color and (self.y_pl != 0).any():
            self._pl_line = self[0].step(
                self.x, 
                self.y_pl, 
                color=self.pl_color, 
                linestyle=self.pl_linestyle,
                where='mid', 
                zorder=6,
                label="P.L.",
            )[0]

        if self.fe_color and (self.y_fe != 0).any():
            self._fe_line = self[0].step(
                self.x, 
                self.y_pl + self.y_fe, 
                color=self.fe_color, 
                linestyle=self.fe_linestyle,
                where='mid', 
                zorder=5,
                label="Fe C.",
            )[0]

        if self.ba_color and (self.y_ba != 0).any():
            self._ba_line = self[0].step(
                self.x, 
                self.y_pl + self.y_ba, 
                color=self.ba_color, 
                linestyle=self.ba_linestyle,
                where='mid', 
                zorder=4,
                label="Ba.C.",
            )[0]

        if self.hg_color and (self.y_hg != 0).any():
            self._hg_line = self[0].step(
                self.x, 
                self.y_pl + self.y_hg, 
                color=self.hg_color, 
                linestyle=self.hg_linestyle,
                where='mid', 
                zorder=3,
                label="Gal.",
            )[0]

        if self.em_color and (self.y_em != 0).any():
            self._em_line = self[0].step(
                self.x, 
                self.y_pl + self.y_em, 
                color=self.em_color, 
                linestyle=self.em_linestyle,
                where='mid', 
                zorder=2,
                label="Lines",
            )[0]