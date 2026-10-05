from dataclasses import dataclass

from ..annotation import BalmerModelAnnotater
from .base import (
    _BaseWindowsPlot,
    _BaseWindowsPlotWithDifferences,
    _BaseWindowsPlotWithResiduals,
)


@dataclass
class BalmerWindowsPlot(_BaseWindowsPlot, BalmerModelAnnotater):
    """Class for creating an annotated plot of a model fit to data."""
    
@dataclass
class BalmerWindowsPlotWithResiduals(_BaseWindowsPlotWithResiduals, BalmerModelAnnotater):
    """
    Class for creating an annotated plot of a model fit to data, with 
    residuals plotted in a separate window.
    """

@dataclass
class BalmerWindowsPlotWithDifferences(_BaseWindowsPlotWithDifferences, BalmerModelAnnotater):
    """
    Class for creating an annotated plot of a model fit to data, with 
    differences plotted in a separate window.
    """