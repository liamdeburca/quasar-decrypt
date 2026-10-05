from dataclasses import dataclass

from ..annotation import HostGalaxyModelAnnotater
from .base import (
    _BaseWindowsPlot,
    _BaseWindowsPlotWithDifferences,
    _BaseWindowsPlotWithResiduals,
)


@dataclass
class HostWindowsPlot(_BaseWindowsPlot, HostGalaxyModelAnnotater):
    """Class for creating an annotated plot of a model fit to data."""

@dataclass
class HostWindowsPlotWithResiduals(_BaseWindowsPlotWithResiduals, HostGalaxyModelAnnotater):
    """
    Class for creating an annotated plot of a model fit to data, with 
    residuals plotted in a separate window.
    """

@dataclass
class HostWindowsPlotWithDifferences(_BaseWindowsPlotWithDifferences, HostGalaxyModelAnnotater):
    """
    Class for creating an annotated plot of a model fit to data, with 
    differences plotted in a separate window.
    """