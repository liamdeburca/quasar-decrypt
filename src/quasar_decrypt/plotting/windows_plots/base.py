from dataclasses import dataclass, field
from typing import Self

from ..layers import (
    DataWithComponentsLayer,
    DifferenceLayer,
    ResidualLayer,
    WindowsLayer,
)
from ..managers import (
    SinglePlotManager,
    StandardWithDifferencePlotManager,
    StandardWithResidualsPlotManager,
)
from ..managers.misc import LegendLoc


@dataclass
class _Base(DataWithComponentsLayer):
    # Overwrite 'LegendManager' fields
    legend_locs: tuple[LegendLoc | None, ...] = field(default=('upper left',), kw_only=True)
    legend_ncols: tuple[int | None, ...] = field(default=(2,), kw_only=True)


@dataclass
class _BaseWindowsPlot(
    _Base,
    SinglePlotManager, 
    WindowsLayer,
):
    """Base class for creating an annotated plot of a model fit to data."""
    def plot(self) -> Self:
        self.instantiate_figure()
        self.format_title()
        self.format_subtitle()

        self.plot_data()
        self.plot_components()
        self.plot_windows()

        self.format_labels()
        self.format_scales()
        self.format_limits()
        self.format_annotation()
        self.format_legends()

        return self

@dataclass
class _BaseWindowsPlotWithResiduals(
    _Base,
    StandardWithResidualsPlotManager,
    ResidualLayer,
    WindowsLayer,
):
    """
    Base class for creating an annotated plot of a model fit to data, with 
    residuals plotted in a separate window.
    """
    def plot(self) -> Self:
        self.instantiate_figure()
        self.format_title()
        self.format_subtitle()

        self.plot_data()
        self.plot_components()
        self.plot_residuals()
        self.plot_windows()

        self.format_labels()
        self.format_scales()
        self.format_limits()
        self.format_annotation()
        self.format_legends()

        return self

@dataclass
class _BaseWindowsPlotWithDifferences(
    _Base,
    StandardWithDifferencePlotManager,
    DifferenceLayer,
    WindowsLayer,
):
    """
    Base class for creating an annotated plot of a model fit to data, with 
    differences plotted in a separate window.
    """
    def plot(self) -> Self:
        self.instantiate_figure()
        self.format_title()
        self.format_subtitle()

        self.plot_data()
        self.plot_components()
        self.plot_differences()
        self.plot_windows()

        self.format_labels()
        self.format_scales()
        self.format_limits()
        self.format_annotation()
        self.format_legends()

        return self