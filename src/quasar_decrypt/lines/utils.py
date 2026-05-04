from quasar_utils.setup import Info

from numpy import float64
from numpy.typing import NDArray
from astropy.units import Quantity
from pandas import read_csv
from functools import partial

from pydantic import validate_call

from quasar_typing.pathlib import AbsoluteFilePath
from quasar_typing.pandas import LineList

def velocity_to_wavelength(
    wave: float, 
    vel: float
) -> float:
    """
    Calculates a wavelength offset using the Doppler equation:
        dx = x * (v / c)

    Parameters
    ----------
    wave : float
        Unitless wavelength. 
    vel : float
        Velocity in units of the speed of light.
    
    Returns
    -------
    dx : float
        Unitless wavelength offset.
    """
    return wave * vel

def wavelength_to_velocity(
    wave: float,
    off: float
) -> float:
    """
    Calculates the velocity offset using the Doppler equation:
        dx = x * (v / c) -> v / c = dx / x

    Parameters
    ----------
    wave : float
        Unitless wavelength. 
    off : float
        Unitless wavelength offset.

    Returns
    -------
    vel : float
        Velocity offset in units of the speed of light. 
    """
    return off / wave

def common_middle(
    wave1: float | NDArray[float64],
    wave2: float | NDArray[float64],
    vel1: float | NDArray[float64],
    vel2: float | NDArray[float64],
) -> float | NDArray[float64]:
    """
    Finds the middle point between two wavelengths, such that the relative 
    velocity offsets from both points are equal (but with opposite units). 

    Parameters
    ----------
    wave1 : float or numpy.array
        Unitless wavelength. 
    wave2 : float or numpy.array
        Unitless wavelength.
    vel1 : float or numpy.array
        Velocity offset in units of the speed of light.
    vel2 : float or numpy.array
        Velocity offset in units of the speed of light.
    """
    return (wave1 * wave2 * (vel1 + vel2)) / (wave1 * vel1 + wave2 * vel2)