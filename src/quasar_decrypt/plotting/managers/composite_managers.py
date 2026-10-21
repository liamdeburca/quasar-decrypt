from dataclasses import dataclass, field
from typing import Literal

from .labels import (
    DoubleLabelsManager,
    DoubleLabelsManagerDifference,
    DoubleLabelsManagerResidual,
    SingleLabelsManager,
    TripleLabelsManager,
    TripleLabelsManagerDifferenceAndTemplate,
    TripleLabelsManagerResidualAndTemplate,
)
from .limits import (
    DoubleLimitsManager,
    SingleLimitsManager,
    TripleLimitsManager,
)
from .misc import (
    ColorsManager,
    FigureManager,
    LegendsManager,
    LinestylesManager,
    SubtitleManager,
    TitleManager,
)
from .scales import (
    DoubleScalesManager,
    SingleScalesManager,
    TripleScalesManager,
)
from .spectrum_manager import SpectrumManager


@dataclass
class GeneralManager(
    SpectrumManager,
    FigureManager,
    TitleManager,
    SubtitleManager,
    ColorsManager,
    LinestylesManager,
    LegendsManager,
):
    """General plotting manager that combines all individual managers."""


### Multiple subplots


@dataclass
class SinglePlotManager(
    GeneralManager,
    SingleLabelsManager,
    SingleScalesManager,
    SingleLimitsManager,
):
    """General plotting manager for a single plot."""

    # Override 'FigureManager' attributes
    nrows: Literal[1] = field(default=1, init=False)
    ncols: Literal[1] = field(default=1, init=False)
    sharex: Literal[False] = field(default=False, init=False)
    sharey: Literal[False] = field(default=False, init=False)
    height_ratios: None = field(default=None, init=False)
    width_ratios: None = field(default=None, init=False)
    hspace: None = field(default=None, init=False)
    wspace: None = field(default=None, init=False)


@dataclass
class DoublePlotManager(
    GeneralManager,
    DoubleLabelsManager,
    DoubleScalesManager,
    DoubleLimitsManager,
):
    """General plotting manager for a double plot."""

    # Override 'FigureManager' attributes
    nrows: Literal[1] = field(default=1, init=False)
    ncols: Literal[2] = field(default=2, init=False)
    sharex: Literal[False] = field(default=False, init=False)
    sharey: Literal[False] = field(default=False, init=False)
    height_ratios: None = field(default=None, init=False)
    width_ratios: None = field(default=None, init=False)
    hspace: None = field(default=None, init=False)
    wspace: None = field(default=None, init=False)

    # Override 'LabelsManager' attributes
    y1label: None = field(default=None, init=False)
    y1loc: Literal["center"] = field(default="center", init=False)

    # Override 'LimitsManager' attributes
    y1lim: None = field(default=None, init=False)


@dataclass
class TriplePlotManager(
    GeneralManager,
    TripleLabelsManager,
    TripleScalesManager,
    TripleLimitsManager,
):
    """General plotting manager for a triple plot."""

    # Override 'FigureManager' attributes
    nrows: Literal[1] = field(default=1, init=False)
    ncols: Literal[3] = field(default=3, init=False)
    sharex: Literal[False] = field(default=False, init=False)
    sharey: Literal[False] = field(default=False, init=False)
    height_ratios: None = field(default=None, init=False)
    width_ratios: None = field(default=None, init=False)
    hspace: None = field(default=None, init=False)
    wspace: None = field(default=None, init=False)

    # Override 'LabelsManager' attributes
    y1label: None = field(default=None, init=False)
    y1loc: Literal["center"] = field(default="center", init=False)
    y2label: None = field(default=None, init=False)
    y2loc: Literal["center"] = field(default="center", init=False)

    # Override 'LimitsManager' attributes
    y1lim: None = field(default=None, init=False)
    y2lim: None = field(default=None, init=False)


### Plot types (2)


@dataclass
class StandardWithResidualsPlotManager(
    GeneralManager,
    DoubleLabelsManagerResidual,
    DoubleLimitsManager,
    DoubleScalesManager,
):
    """General plotting manager for a standard plot with a residuals window."""

    # Override 'FigureManager' attributes
    nrows: Literal[2] = field(default=2, init=False)
    ncols: Literal[1] = field(default=1, init=False)
    sharex: Literal[True] = field(default=True, init=False)
    sharey: Literal[False] = field(default=False, init=False)
    height_ratios: tuple[float, float] = field(default=(3, 1), kw_only=True)
    width_ratios: None = field(default=None, init=False)
    hspace: float = field(default=0.05, kw_only=True)
    wspace: None = field(default=None, init=False)


@dataclass
class StandardWithDifferencePlotManager(
    GeneralManager,
    DoubleLabelsManagerDifference,
    DoubleLimitsManager,
    DoubleScalesManager,
):
    """General plotting manager for a standard plot with a difference window."""

    # Override 'FigureManager' attributes
    nrows: Literal[2] = field(default=2, init=False)
    ncols: Literal[1] = field(default=1, init=False)
    sharex: Literal[True] = field(default=True, init=False)
    sharey: Literal[False] = field(default=False, init=False)
    height_ratios: tuple[float, float] = field(default=(3, 1), kw_only=True)
    width_ratios: None = field(default=None, init=False)
    hspace: float = field(default=0.05, kw_only=True)
    wspace: None = field(default=None, init=False)


### Plot types (3)


@dataclass
class StandardWithResidualsAndTemplateManager(
    GeneralManager,
    TripleLabelsManagerResidualAndTemplate,
    TripleLimitsManager,
    TripleScalesManager,
):
    """
    General plotting manager for a standard plot with residuals and a window
    showing the template(s).
    """

    # Override 'FigureManager' attributes
    nrows: Literal[3] = field(default=3, init=False)
    ncols: Literal[1] = field(default=1, init=False)
    sharex: Literal[True] = field(default=True, init=False)
    sharey: Literal[False] = field(default=False, init=False)
    height_ratios: tuple[float, float, float] = field(default=(3, 1, 1), kw_only=True)
    width_ratios: None = field(default=None, init=False)
    hspace: float = field(default=0.05, kw_only=True)
    wspace: None = field(default=None, init=False)


@dataclass
class StandardWithDifferenceAndTemplateManager(
    GeneralManager,
    TripleLabelsManagerDifferenceAndTemplate,
    TripleLimitsManager,
    TripleScalesManager,
):
    """
    General plotting manager for a standard plot with differences and a window
    showing the template(s).
    """

    # Override 'FigureManager' attributes
    nrows: Literal[3] = field(default=3, init=False)
    ncols: Literal[1] = field(default=1, init=False)
    sharex: Literal[True] = field(default=True, init=False)
    sharey: Literal[False] = field(default=False, init=False)
    height_ratios: tuple[float, float, float] = field(default=(3, 1, 1), kw_only=True)
    width_ratios: None = field(default=None, init=False)
    hspace: float = field(default=0.05, kw_only=True)
    wspace: None = field(default=None, init=False)
