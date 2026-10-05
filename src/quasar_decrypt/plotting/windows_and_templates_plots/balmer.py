from dataclasses import dataclass

from ..annotation import BalmerModelAnnotater
from ..layers.templates import BalmerTemplatesLayer
from .base import (
    _BaseTemplatesWithDifferencesPlot,
    _BaseTemplatesWithResidualsPlot,
    _BaseWindowsAndTemplatesWithDifferencesPlot,
    _BaseWindowsAndTemplatesWithResidualsPlot,
)


@dataclass
class BalmerTemplatesWithResidualsPlot(
    _BaseTemplatesWithResidualsPlot, 
    BalmerModelAnnotater,
    BalmerTemplatesLayer,
):
    """
    Class for creating an annotated plot of a BalmerModel, with data-model 
    residuals in the second window, and templates in the third.
    """

@dataclass
class BalmerTemplatesWithDifferencesPlot(
    _BaseTemplatesWithDifferencesPlot, 
    BalmerModelAnnotater,
    BalmerTemplatesLayer,
):
    """
    Class for creating an annotated plot of a BalmerModel, with data-model 
    differences in the second window, and templates in the third.
    """

@dataclass
class BalmerWindowsAndTemplatesWithResidualsPlot(
    _BaseWindowsAndTemplatesWithResidualsPlot, 
    BalmerModelAnnotater,
    BalmerTemplatesLayer,
):
    """
    Class for creating an annotated plot of a BalmerModel, with data-model 
    residuals in the second window, and templates in the third, with windows 
    overlaid.
    """

@dataclass
class BalmerWindowsAndTemplatesWithDifferencesPlot(
    _BaseWindowsAndTemplatesWithDifferencesPlot, 
    BalmerModelAnnotater,
    BalmerTemplatesLayer,
):
    """
    Class for creating an annotated plot of a BalmerModel, with data-model 
    differences in the second window, and templates in the third, with windows 
    overlaid.
    """
