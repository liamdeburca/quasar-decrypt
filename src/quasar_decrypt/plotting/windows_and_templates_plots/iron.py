from dataclasses import dataclass

from ..annotation import IronModelAnnotater
from ..layers.templates import IronTemplatesLayer
from .base import (
    _BaseTemplatesWithDifferencesPlot,
    _BaseTemplatesWithResidualsPlot,
    _BaseWindowsAndTemplatesWithDifferencesPlot,
    _BaseWindowsAndTemplatesWithResidualsPlot,
)


@dataclass
class IronTemplatesWithResidualsPlot(
    _BaseTemplatesWithResidualsPlot, 
    IronModelAnnotater,
    IronTemplatesLayer,
):
    """
    Class for creating an annotated plot of an IronModel, with data-model 
    residuals in the second window, and templates in the third.
    """

@dataclass
class IronTemplatesWithDifferencesPlot(
    _BaseTemplatesWithDifferencesPlot, 
    IronModelAnnotater,
    IronTemplatesLayer,
):
    """
    Class for creating an annotated plot of an IronModel, with data-model 
    differences in the second window, and templates in the third.
    """

@dataclass
class IronWindowsAndTemplatesWithResidualsPlot(
    _BaseWindowsAndTemplatesWithResidualsPlot, 
    IronModelAnnotater,
    IronTemplatesLayer,
):
    """
    Class for creating an annotated plot of an IronModel, with data-model 
    residuals in the second window, and templates in the third, with windows 
    overlaid.
    """

@dataclass
class IronWindowsAndTemplatesWithDifferencesPlot(
    _BaseWindowsAndTemplatesWithDifferencesPlot, 
    IronModelAnnotater,
    IronTemplatesLayer,
):
    """
    Class for creating an annotated plot of an IronModel, with data-model 
    differences in the second window, and templates in the third, with windows 
    overlaid.
    """
