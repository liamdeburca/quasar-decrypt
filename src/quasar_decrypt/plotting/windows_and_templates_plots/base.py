from dataclasses import dataclass, field
from typing import Self

from ..layers import (
    DataWithComponentsLayer,
    DifferenceLayer,
    ResidualLayer,
    WindowsLayer,
)
from ..managers import (
    StandardWithDifferenceAndTemplateManager,
    StandardWithResidualsAndTemplateManager,
)
from ..managers.misc import LegendLoc


@dataclass
class _Base(DataWithComponentsLayer):
    # Overwrite 'LegendManager' fields
    legend_locs: tuple[LegendLoc | None, ...] = field(default=('upper left',), kw_only=True)
    legend_ncols: tuple[int | None, ...] = field(default=(2,), kw_only=True)


@dataclass
class _BaseTemplatesWithResidualsPlot(
    _Base, 
    StandardWithResidualsAndTemplateManager, 
    ResidualLayer,
):
    """
    Class for creating an annotated plot of a model, with data-model residuals 
    in the second window, and templates in the third.
    """
    def plot(self) -> Self:
        self.instantiate_figure()
        self.format_title()
        self.format_subtitle()

        self.plot_data()
        self.plot_components()
        self.plot_templates()
        self.plot_residuals()

        self.format_labels()
        self.format_scales()
        self.format_limits()
        self.format_annotation()
        self.format_legends()

        return self


@dataclass
class _BaseTemplatesWithDifferencesPlot(
    _Base,
    StandardWithDifferenceAndTemplateManager, 
    DifferenceLayer,
):
    """
    Class for creating an annotated plot of a model, with data-model differences 
    in the second window, and templates in the third.
    """
    def plot(self) -> Self:
        self.instantiate_figure()
        self.format_title()
        self.format_subtitle()

        self.plot_data()
        self.plot_components()
        self.plot_templates()
        self.plot_differences()

        self.format_labels()
        self.format_scales()
        self.format_limits()
        self.format_annotation()
        self.format_legends()

        return self


@dataclass
class _BaseWindowsAndTemplatesWithResidualsPlot(
    _Base,
    StandardWithResidualsAndTemplateManager, 
    ResidualLayer,
    WindowsLayer,
):
    """
    Class for creating an annotated plot of a model, with data-model residuals 
    in the second window, and templates in the third, with windows overlaid.
    """
    def plot(self) -> Self:
        self.instantiate_figure()
        self.format_title()
        self.format_subtitle()

        self.plot_data()
        self.plot_components()
        self.plot_templates()
        self.plot_residuals()
        self.plot_windows()

        self.format_labels()
        self.format_scales()
        self.format_limits()
        self.format_annotation()
        self.format_legends()

        return self

@dataclass
class _BaseWindowsAndTemplatesWithDifferencesPlot(
    _Base,
    StandardWithDifferenceAndTemplateManager, 
    DifferenceLayer,
    WindowsLayer,
):
    """
    Class for creating an annotated plot of a model, with data-model differences 
    in the second window, and templates in the third, with windows overlaid.
    """
    def plot(self) -> Self:
        self.instantiate_figure()
        self.format_title()
        self.format_subtitle()

        self.plot_data()
        self.plot_components()
        self.plot_templates()
        self.plot_differences()
        self.plot_windows()

        self.format_labels()
        self.format_scales()
        self.format_limits()
        self.format_annotation()
        self.format_legends()

        return self