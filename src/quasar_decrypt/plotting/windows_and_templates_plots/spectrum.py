from dataclasses import dataclass
from typing import Self

from ..layers.templates import AllTemplatesLayer
from ..managers.model_manager import ModelManager
from .base import _BaseTemplatesWithResidualsPlot


@dataclass
class AllTemplatesWithResidualsPlot(
    _BaseTemplatesWithResidualsPlot,
    ModelManager,
    AllTemplatesLayer,
):
    """
    Class for creating an annotated plot of a model, with data-model 
    residuals in the second window, and templates in the third.
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
        self.format_legends()

        return self

@dataclass
class AllTemplatesWithDifferencesPlot(
    _BaseTemplatesWithResidualsPlot,
    ModelManager,
    AllTemplatesLayer,
):
    """
    Class for creating an annotated plot of a model, with data-model 
    differences in the second window, and templates in the third.
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
        self.format_legends()

        return self