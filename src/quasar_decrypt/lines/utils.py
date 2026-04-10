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

###

def n_max_converter(s: str) -> int:
    if len(s) == 0: return 1
    else:           return int(s)

def line_converter(info: Info, s: str) -> float:
    assert len(s) > 0

    if len(s.split(' ')) == 1: return float(s)
    else:                      return info.units.getWavelength(Quantity(s))

def needs_line_converter(s: str) -> float | None:
    return s or None

def strength_lower_converter(info: Info, s: str) -> float:
    if len(s) == 0: return info.lines['strength_bounds'][0]

    if len(s.split(' ')) == 1: return float(s)
    else:                      return info.units.getStrength(Quantity(s))

def strength_upper_converter(info: Info, s: str) -> float:
    if len(s) == 0: return info.lines['strength_bounds'][1]

    if len(s.split(' ')) == 1: return float(s)
    else:                      return info.units.getStrength(Quantity(s))

def sigma_v_lower_converter(info: Info, s: str) -> float:
    if len(s) == 0: return info.lines['sigma_v_bounds'][0]

    if len(s.split(' ')) == 1: return float(s)
    else:                      return info.units.getC(Quantity(s))

def sigma_v_upper_converter(info: Info, s: str) -> float:
    if len(s) == 0: return info.lines['sigma_v_bounds'][1]

    if len(s.split(' ')) == 1: return float(s)
    else:                      return info.units.getC(Quantity(s))

def v_off_lower_converter(info: Info, s: str) -> float:
    if len(s) == 0: return info.lines['v_off_bounds'][0]

    if len(s.split(' ')) == 1: return float(s)
    else:                      return info.units.getC(Quantity(s))

def v_off_upper_converter(info: Info, s: str) -> float:
    if len(s) == 0: return info.lines['v_off_bounds'][1]

    if len(s.split(' ')) == 1: return float(s)
    else:                      return info.units.getC(Quantity(s))

def is_copy_of_converter(s: str) -> str | None:
    return s or None

def scale_init_converter(info: Info, s: str) -> float:
    if len(s) == 0: return info.lines['scale_init']
    else:           return float(s)

def scale_lower_converter(info: Info, s: str) -> float:
    if len(s) == 0: return info.lines['scale_bounds'][0]
    else:           return float(s)

def scale_upper_converter(info: Info, s: str) -> float:
    if len(s) == 0: return info.lines['scale_bounds'][1]
    else:           return float(s)

@validate_call
def read_linelist(
    path: AbsoluteFilePath,
    info: Info,
) -> LineList:
    """
    ** PYDANTIC VALIDATED FUNCTION **
    """
    df = read_csv(
        path,
        skipinitialspace = True,
        usecols = LineList.REQUIRED_COLUMNS,
        converters = dict(
            n_max          = n_max_converter,
            is_copy_of     = is_copy_of_converter,
            needs_line     = needs_line_converter,
            line           = partial(line_converter,           info),
            strength_lower = partial(strength_lower_converter, info),
            strength_upper = partial(strength_upper_converter, info),
            sigma_v_lower  = partial(sigma_v_lower_converter,  info),
            sigma_v_upper  = partial(sigma_v_upper_converter,  info),
            v_off_lower    = partial(v_off_lower_converter,    info),
            v_off_upper    = partial(v_off_upper_converter,    info),
            scale_init     = partial(scale_init_converter,     info),
            scale_lower    = partial(scale_lower_converter,    info),
            scale_upper    = partial(scale_upper_converter,    info),
        )
    )
    df.sort_values('line', inplace=True)
    return df