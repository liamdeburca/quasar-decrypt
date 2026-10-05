__all__ = [
    "DifferenceAndTemplatePlotter",
    "DifferencePlotter",
    "QuickPlotter",
    "ResidualAndTemplatePlotter",
    "ResidualPlotter",
    "WindowedDifferenceAndTemplatePlotter",
    "WindowedDifferencePlotter",
    "WindowedQuickPlotter",
    "WindowedResidualAndTemplatePlotter",
    "WindowedResidualPlotter",
]

from dataclasses import dataclass
from typing import Self

from .layers import (
    DataWithComponentsLayer,
    DifferenceLayer,
    ResidualLayer,
    WindowsLayer,
)
from .managers import (
    SinglePlotManager,
    StandardWithDifferenceAndTemplateManager,
    StandardWithDifferencePlotManager,
    StandardWithResidualsAndTemplateManager,
    StandardWithResidualsPlotManager,
)


@dataclass
class QuickPlotter(SinglePlotManager, DataWithComponentsLayer):
    """
    Dataclass for quickly plotting a spectrum and any modelled components. Inherits from:
    
    - SpectrumManager
    - FigureManager
    - ColorsManager
    - LabelsManager
    - LimitsManager
    - ScaleManager
    - DataLayer
    - ComponentsLayer
    """
    def plot(self) -> Self:
        self.instantiate_figure()
        self.format_title()
        self.format_subtitle()

        self.plot_data()
        self.plot_components()

        self.format_labels()
        self.format_scales()
        self.format_limits()
        self.format_legends()

        return self

### With differences / residuals

@dataclass
class DifferencePlotter(StandardWithDifferencePlotManager, DataWithComponentsLayer, DifferenceLayer):
    """
    Dataclass for plotting the difference between the data and the modelled components.
    """
    def plot(self) -> Self:
        self.instantiate_figure()
        self.format_title()
        self.format_subtitle()

        self.plot_data()
        self.plot_components()
        self.plot_differences()

        self.format_labels()
        self.format_scales()
        self.format_limits()
        self.format_legends()

        return self

@dataclass
class ResidualPlotter(StandardWithResidualsPlotManager, DataWithComponentsLayer, ResidualLayer):
    """
    Dataclass for plotting the residuals between the data and the modelled components.
    """
    def plot(self) -> Self:
        self.instantiate_figure()
        self.format_title()
        self.format_subtitle()

        self.plot_data()
        self.plot_components()
        self.plot_residuals()

        self.format_labels()
        self.format_scales()
        self.format_limits()
        self.format_legends()

        return self

### With differences / residuals + templates

@dataclass
class DifferenceAndTemplatePlotter(
    StandardWithDifferenceAndTemplateManager, 
    DataWithComponentsLayer, 
    DifferenceLayer,
):
    """
    Dataclass for plotting the difference between the data and the modelled 
    components, along with the templates.
    """
    def plot(self) -> Self:
        self.instantiate_figure()
        self.format_title()
        self.format_subtitle()

        self.plot_data()
        self.plot_components()
        self.plot_differences()
        self.plot_templates()

        self.format_labels()
        self.format_scales()
        self.format_limits()
        self.format_legends()

        return self

@dataclass
class ResidualAndTemplatePlotter(
    StandardWithResidualsAndTemplateManager, 
    DataWithComponentsLayer, 
    ResidualLayer,
):
    """
    Dataclass for plotting the residuals between the data and the modelled 
    components, along with the templates.
    """
    def plot(self) -> Self:
        self.instantiate_figure()
        self.format_title()
        self.format_subtitle()

        self.plot_data()
        self.plot_components()
        self.plot_residuals()
        self.plot_templates()

        self.format_labels()
        self.format_scales()
        self.format_limits()
        self.format_legends()

        return self

### With windows

@dataclass
class WindowedQuickPlotter(QuickPlotter, WindowsLayer):
    """
    Dataclass for quickly plotting a spectrum, the specified windows, and any 
    modelled components.
    """
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
        self.format_legends()

        return self

### With windows, differences / residuals

@dataclass
class WindowedDifferencePlotter(DifferencePlotter, WindowsLayer):
    """
    Dataclass for plotting the difference between the data and the modelled components,
    along with the specified windows.
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
        self.format_legends()

        return self

@dataclass
class WindowedResidualPlotter(ResidualPlotter, WindowsLayer):
    """
    Dataclass for plotting the residuals between the data and the modelled components,
    along with the specified windows.
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
        self.format_legends()

        return self

### With windows, differences / residuals + templates

@dataclass
class WindowedDifferenceAndTemplatePlotter(
    DifferenceAndTemplatePlotter, 
    WindowsLayer,
):
    """
    Dataclass for plotting the difference between the data and the modelled 
    components, along with the specified windows and the templates.
    """
    def plot(self) -> Self:
        self.instantiate_figure()
        self.format_title()
        self.format_subtitle()

        self.plot_data()
        self.plot_components()
        self.plot_differences()
        self.plot_templates()
        self.plot_windows()

        self.format_labels()
        self.format_scales()
        self.format_limits()
        self.format_legends()

        return self

@dataclass
class WindowedResidualAndTemplatePlotter(
    ResidualAndTemplatePlotter, 
    WindowsLayer,
):
    """
    Dataclass for plotting the residuals between the data and the modelled 
    components, along with the specified windows and the templates.
    """
    def plot(self) -> Self:
        self.instantiate_figure()
        self.format_title()
        self.format_subtitle()

        self.plot_data()
        self.plot_components()
        self.plot_residuals()
        self.plot_templates()
        self.plot_windows()

        self.format_labels()
        self.format_scales()
        self.format_limits()
        self.format_legends()

        return self