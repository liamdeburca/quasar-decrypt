from dataclasses import dataclass

from ..annotation import PowerLawModelAnnotater
from .base import (
    _BaseWindowsPlot,
    _BaseWindowsPlotWithDifferences,
    _BaseWindowsPlotWithResiduals,
)


@dataclass
class ContinuumWindowsPlot(_BaseWindowsPlot, PowerLawModelAnnotater):
    """Class for creating an annotated plot of a model fit to data."""

@dataclass
class ContinuumWindowsPlotWithResiduals(_BaseWindowsPlotWithResiduals, PowerLawModelAnnotater):
    """
    Class for creating an annotated plot of a model fit to data, with 
    residuals plotted in a separate window.
    """

@dataclass
class ContinuumWindowsPlotWithDifferences(_BaseWindowsPlotWithDifferences, PowerLawModelAnnotater):
    """
    Class for creating an annotated plot of a model fit to data, with 
    differences plotted in a separate window.
    """