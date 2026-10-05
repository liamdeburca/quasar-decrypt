from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.pyplot import close, subplots
from numpy import isfinite

from ..defaults import DEFAULT_COLORS, DEFAULT_LINESTYLES

AxesLimits = tuple[float | None, float | None]
HoriLoc = Literal["left", "center", "right"]
VertLoc = Literal["bottom", "center", "top"]
Scale = Literal["linear", "log"]
LegendLoc = Literal[
    "best",
    "upper right", "upper left",
    "lower left", "lower right",
    "right",
    "center left", "center right",
    "lower center", "upper center",
    "center"
]


@dataclass
class FigureManager:
    """
    Dataclass for managing the Figure and Axes objects of a plot.
    """
    fig: Figure | None = field(default=None, kw_only=True)
    axes: list[Axes] = field(default_factory=list, kw_only=True)

    nrows: int = field(default=1, kw_only=True)
    ncols: int = field(default=1, kw_only=True)
    figsize: tuple[float, float] = field(default=(8, 6), kw_only=True)
    dpi: int = field(default=300, kw_only=True)
    sharex: bool = field(default=False, kw_only=True)
    sharey: bool = field(default=False, kw_only=True)
    height_ratios: tuple[float, ...] | None = field(default=None, kw_only=True)
    width_ratios: tuple[float, ...] | None = field(default=None, kw_only=True)
    hspace: float | None = field(default=None, kw_only=True)
    wspace: float | None = field(default=None, kw_only=True)

    def __getitem__(self, index: int) -> Axes | None:
        return self.axes[index]

    def __iter__(self) -> Iterator[Axes]:
        return iter(self.axes)

    def __len__(self) -> int:
        return len(self.axes)

    def instantiate_figure(self) -> None:         
        if self.fig is None:
            fig, axes = subplots(
                self.nrows, self.ncols, 
                figsize=self.figsize, 
                dpi=self.dpi,
                sharex=self.sharex, 
                sharey=self.sharey,
                height_ratios=self.height_ratios,
                width_ratios=self.width_ratios,
            )
            fig.subplots_adjust(wspace=self.wspace, hspace=self.hspace)

            self.fig = fig
            self.axes = [axes] if isinstance(axes, Axes) else list(axes.flatten())

    def show(self) -> Self:
        self.fig.show()
        return self

    def savefig(self, fname: str | Path, **kwargs) -> Self:
        self.fig.savefig(fname, **kwargs)
        return self
    
    def close(self) -> None:
        self.fig.clear()
        close(self.fig)


@dataclass
class TitleManager:
    """
    Dataclass for managing the title of a plot.
    """
    title: str | Literal[False] | None = field(default=None, kw_only=True)

    title_prefix: str = field(default="", kw_only=True)
    title_suffix: str = field(default="", kw_only=True)
    title_loc: HoriLoc = field(default="left", kw_only=True)

    _title: str | None = field(default=None, init=False)

    def format_title(self) -> None:
        if self.title is not False:
            title = self.title
            if title is None:
                title = getattr(
                    self.spectrum, 
                    "title", 
                    f"<{self.spectrum.__class__.__name__} at {hex(id(self.spectrum))}>"
                )

            self._title = self.title_prefix + title + self.title_suffix
            self[0].set_title(self._title, loc=self.title_loc)

@dataclass
class TitleDisabler:
    title: None = field(default=None, kw_only=True)

    title_prefix: None = field(default=None, kw_only=True)
    title_suffix: None = field(default=None, kw_only=True)
    title_loc: None = field(default=None, kw_only=True)

    _title: None = field(default=None, init=False)

    def format_title(self) -> None:
        raise NotImplementedError("'format_title' is disabled.")

@dataclass
class SubtitleManager:
    """
    Dataclass for managing the subtitle of a plot.
    
    The following information is shown:
    - no. of pixels
    - no. of (log-)valid pixels
    - no. of rejected pixels
    - no. of absorbed pixels
    - chi-square value
    """
    subtitle_disable: bool = field(default=False, kw_only=True)

    subtitle_prefix: str = field(default="", kw_only=True)
    subtitle_suffix: str = field(default="", kw_only=True)
    subtitle_loc: HoriLoc = field(default="right", kw_only=True)

    _subtitle: str | None = field(default=None, init=False)

    def format_subtitle(self) -> None:
        if self.subtitle_disable:
            return

        subtitle = f"n={isfinite(self.x).sum()}, "

        if getattr(self, "yscale", "linear") == "linear":
            subtitle += f"val.={self.n_val}, "
            mask = self.valid_pixels
        else:
            subtitle += f"logval.={self.n_logval}, "
            mask = self.log_valid_pixels

        subtitle += f"rej.={self.n_rej}, abs.={self.n_abs}, "

        chi2 = (((self.y[mask] - self.f[mask]) / self.dy[mask]) ** 2).sum()
        subtitle += rf"$\chi^2$={chi2:.1f}"

        self._subtitle = self.subtitle_prefix + subtitle + self.subtitle_suffix
        self[0].set_title(self._subtitle, loc=self.subtitle_loc, fontsize="small")

@dataclass
class SubtitleDisabler:
    subtitle_disable: bool = field(default=True, kw_only=True)

    subtitle_prefix: None = field(default=None, kw_only=True)
    subtitle_suffix: None = field(default=None, kw_only=True)
    subtitle_loc: None = field(default=None, kw_only=True)

    _subtitle: None = field(default=None, init=False)

    def format_subtitle(self) -> None:
        raise NotImplementedError("'format_subtitle' is disabled.")

@dataclass
class ColorsManager:
    """
    Dataclass managing the colors of different spectral components.
    """
    color: str | None = field(default_factory=lambda: DEFAULT_COLORS['data'], kw_only=True)
    f_color: str | None = field(default_factory=lambda: DEFAULT_COLORS['fit'], kw_only=True)
    pl_color: str | None = field(default_factory=lambda: DEFAULT_COLORS['pl'], kw_only=True)
    fe_color: str | None = field(default_factory=lambda: DEFAULT_COLORS['fe'], kw_only=True)
    ba_color: str | None = field(default_factory=lambda: DEFAULT_COLORS['ba'], kw_only=True)
    hg_color: str | None = field(default_factory=lambda: DEFAULT_COLORS['hg'], kw_only=True)
    em_color: str | None = field(default_factory=lambda: DEFAULT_COLORS['em'], kw_only=True)

    sm_color: str | None = field(default_factory=lambda: DEFAULT_COLORS['sm'], kw_only=True)
    ab_color: str | None = field(default_factory=lambda: DEFAULT_COLORS['ab'], kw_only=True)

    e_color: str | None = field(default_factory=lambda: DEFAULT_COLORS['e'], kw_only=True)
    w_color: str | None = field(default_factory=lambda: DEFAULT_COLORS['w'], kw_only=True)


@dataclass
class LinestylesManager:
    """
    Dataclass managing the linestyles of different spectral components.
    """
    pl_linestyle: str = field(default_factory=lambda: DEFAULT_LINESTYLES.get('pl', 'solid'), kw_only=True)
    fe_linestyle: str = field(default_factory=lambda: DEFAULT_LINESTYLES.get('fe', 'dotted'), kw_only=True)
    ba_linestyle: str = field(default_factory=lambda: DEFAULT_LINESTYLES.get('ba', 'dashed'), kw_only=True)
    hg_linestyle: str = field(default_factory=lambda: DEFAULT_LINESTYLES.get('hg', 'dashed'), kw_only=True)
    em_linestyle: str = field(default_factory=lambda: DEFAULT_LINESTYLES.get('em', 'solid'), kw_only=True)

    sm_linestyle: str = field(default_factory=lambda: DEFAULT_LINESTYLES.get('sm', 'solid'), kw_only=True)

    
@dataclass
class LegendsManager:
    """
    Dataclass managing the legends of the plot.
    """
    legend_locs: tuple[LegendLoc | int | None, ...] = field(default=('best',), kw_only=True)
    legend_ncols: tuple[int | None, ...] = field(default=(1,), kw_only=True)
    legend_fontsize: str | float = field(default="small", kw_only=True)

    _legends: list[Legend] = field(default_factory=list, init=False)

    def format_legends(self) -> None:
        self._legends.clear()

        m = len(self.legend_ncols)
        for i, (ax, loc) in enumerate(zip(self, self.legend_locs)):
            if loc is None:
                continue
            ncols = self.legend_ncols[i] if i < m else 1
            self._legends.append(ax.legend(loc=loc, ncol=ncols, fontsize=self.legend_fontsize))

@dataclass
class LegendsDisabler:
    legend_locs: None = field(default=None, kw_only=True)
    legend_ncols: None = field(default=None, kw_only=True)
    legend_fontsize: None = field(default=None, kw_only=True)

    _legends: None = field(default=None, init=False)

    def format_legends(self) -> None:
        raise NotImplementedError("'format_legends' is disabled.")