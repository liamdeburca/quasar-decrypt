from dataclasses import dataclass

from ..annotation import HostGalaxyModelAnnotater
from ..layers.templates import HostTemplatesLayer
from .base import (
    _BaseTemplatesWithDifferencesPlot,
    _BaseTemplatesWithResidualsPlot,
    _BaseWindowsAndTemplatesWithDifferencesPlot,
    _BaseWindowsAndTemplatesWithResidualsPlot,
)


@dataclass
class HostTemplatesWithResidualsPlot(
    _BaseTemplatesWithResidualsPlot, 
    HostGalaxyModelAnnotater,
    HostTemplatesLayer,
):
    """
    Class for creating an annotated plot of a HostModel, with data-model 
    residuals in the second window, and templates in the third.
    """

@dataclass
class HostTemplatesWithDifferencesPlot(
    _BaseTemplatesWithDifferencesPlot, 
    HostGalaxyModelAnnotater,
    HostTemplatesLayer,
):
    """
    Class for creating an annotated plot of a HostModel, with data-model 
    differences in the second window, and templates in the third.
    """

@dataclass
class HostWindowsAndTemplatesWithResidualsPlot(
    _BaseWindowsAndTemplatesWithResidualsPlot, 
    HostGalaxyModelAnnotater,
    HostTemplatesLayer,
):
    """
    Class for creating an annotated plot of a HostModel, with data-model 
    residuals in the second window, and templates in the third, with windows 
    overlaid.
    """

@dataclass
class HostWindowsAndTemplatesWithDifferencesPlot(
    _BaseWindowsAndTemplatesWithDifferencesPlot, 
    HostGalaxyModelAnnotater,
    HostTemplatesLayer,
):
    """
    Class for creating an annotated plot of a HostModel, with data-model 
    differences in the second window, and templates in the third, with windows 
    overlaid.
    """
