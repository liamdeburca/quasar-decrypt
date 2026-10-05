from dataclasses import dataclass

from ..annotation import IronModelAnnotater
from .base import (
    _BaseWindowsPlot,
    _BaseWindowsPlotWithDifferences,
    _BaseWindowsPlotWithResiduals,
)


@dataclass
class IronWindowsPlot(_BaseWindowsPlot, IronModelAnnotater):
    """Class for creating an annotated plot of a model fit to data."""

@dataclass
class IronWindowsPlotWithResiduals(_BaseWindowsPlotWithResiduals, IronModelAnnotater):
    """
    Class for creating an annotated plot of a model fit to data, with 
    residuals plotted in a separate window.
    """

@dataclass
class IronWindowsPlotWithDifferences(_BaseWindowsPlotWithDifferences, IronModelAnnotater):
    """
    Class for creating an annotated plot of a model fit to data, with 
    differences plotted in a separate window.
    """