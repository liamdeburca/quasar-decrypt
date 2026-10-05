from astropy.constants import c
from numpy import argwhere, float64, fromiter, sign, unique
from numpy.typing import NDArray

C_KMS: float = c.to("km/s").value

def _clip(val: float, lb: float, ub: float) -> float:
    return max(lb, min(ub, val))

def velocity_to_wavelength(wave: float, vel: float) -> float:
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


def wavelength_to_velocity(wave: float, off: float) -> float:
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
        Velocity offset (a.u.).
    vel2 : float or numpy.array
        Velocity offset (a.u.).
    """
    out = (wave1 * wave2 * (vel1 + vel2)) / (wave1 * vel1 + wave2 * vel2)
    assert wave1 < out < wave2
    return out


def common_middle_r(
    wave1: float,
    wave2: float,
    vel1: float,
    vel2: float,
) -> float:
    return min(1, (wave2 - wave1) / (wave1 * vel1 + wave2 * vel2) * C_KMS)

###


def update_v_off_bounds(
    *,
    v_off_bounds: dict[str, tuple[float, float]],
    waves: dict[str, float],
    x_bounds: tuple[float, float],
) -> None:
    """
    Update the `v_off_bounds` dictionary in place. 

    This function ensures that the wavelength interval defined by the velocity 
    offset bounds for each wavelength does not overlap with adjacent wavelength 
    intervals, and stays within the provided wavelength bounds (`x_bounds`).

    Parameters
    ----------
    v_off_bounds : dict[str, tuple[float, float]]
        Dictionary mapping wavelength names to their velocity offset bounds (in 
        km/s).
    waves : dict[str, float]
        Dictionary mapping wavelength names to their unitless wavelengths.
    x_bounds : tuple[float, float]
        The absolute wavelength bounds.

    Notes
    -----
    This function considers the special cases:

    1. If two adjacent wavelengths have competing velocity offset bounds, i.e. 
    the upper bound of the left wavelength is positive and the lower bound of 
    the right wavelength is negative, the function finds a common middle point 
    that ensures that the relative velocity offsets from the points are equal.
    
    2. If the left velocity offset bound is negative or 0, the function locks 
    the left velocity offset bound and adjusts the right velocity offset bound 
    accordingly to prevent overlap. 
    
    3. If the right velocity offset bound is positive or 0, the function locks 
    the right velocity offset bound and adjusts the left velocity offset bound 
    accordingly to prevent overlap. 
    """
    # Get furthest extents of each line
    _v_off_bounds: dict[float, tuple[float, float]] = {}

    for name, wave  in waves.items():
        b = v_off_bounds[name]
        if wave in _v_off_bounds:
            _v_off_bounds[wave] = (
                min(_v_off_bounds[wave][0], b[0]),
                max(_v_off_bounds[wave][1], b[1]),
            )
        else:
            _v_off_bounds[wave] = b

    uniques = unique(sorted(waves.values()))
    lbs = fromiter((_v_off_bounds[wave][0] for wave in uniques), dtype=float64)
    ubs = fromiter((_v_off_bounds[wave][1] for wave in uniques), dtype=float64)
    del _v_off_bounds

    v_lbs: list[float | None] = [None for _ in uniques]
    v_ubs: list[float | None] = [None for _ in uniques]

    for i in range(uniques.size):
        wave1 = uniques[i]
        lb1 = lbs[i]
        ub1 = ubs[i]

        match (sign(lb1), sign(ub1)):
            case (-1, -1):
                if v_lbs[i] is None:
                    v_lbs[i] = lb1
                if v_ubs[i] is None:
                    v_ubs[i] = ub1
            case (-1, 0):
                if v_lbs[i] is None:
                    v_lbs[i] = lb1
                v_ubs[i] = 0.0
                if i < uniques.size - 1:
                    wave2 = uniques[i+1]
                    vmin = (wave1 / wave2 - 1) * C_KMS
                    v_lbs[i+1] = max(vmin, lbs[i+1])
                    v_ubs[i+1] = max(vmin, ubs[i+1])
            case (-1, 1):
                # Standard case
                if v_lbs[i] is None:
                    v_lbs[i] = lb1

                if uniques.size == 1 or i == uniques.size - 1:
                    v_ubs[i] = ub1
                else:
                    wave2 = uniques[i+1]
                    lb2 = lbs[i+1]
                    ub2 = ubs[i+1]

                    if ub2 < 0.0:
                        # 'lb2' and 'ub2' both compete
                        r1 = common_middle_r(wave1, wave2, ub1, -lb2)
                        v_ubs[i]   = r1 * ub1
                        v_lbs[i+1] = r1 * lb2
                        r2 = common_middle_r(wave1, wave2, ub1, -ub2)
                        v_ubs[i+1] = r2 * ub2
                    elif lb2 < 0.0:
                        # 'lb2' competes
                        r = common_middle_r(wave1, wave2, ub1, -lb2)
                        v_ubs[i]   = r * ub1
                        v_lbs[i+1] = r * lb2
                    else:
                        # No competition
                        v_ubs[i]   = min((wave2 / wave1 - 1) * C_KMS, ub1)

            case (0, 0):
                v_lbs[i] = 0.0
                v_ubs[i] = 0.0
                if i < uniques.size - 1:
                    wave2 = uniques[i+1]
                    vmin = (wave1 / wave2 - 1) * C_KMS
                    v_lbs[i+1] = max(vmin, lbs[i+1])
                    v_ubs[i+1] = max(vmin, ubs[i+1])

            case (0, 1):
                v_lbs[i] = 0.0
                if i == uniques.size - 1:
                    v_ubs[i] = ub1
                else:
                    wave2 = uniques[i+1]
                    lb2 = lbs[i+1]
                    ub2 = ubs[i+1]

                    if ub2 < 0.0:
                        # 'lb2' and 'ub2' both compete
                        r1 = common_middle_r(wave1, wave2, ub1, -lb2)
                        v_ubs[i]   = r1 * ub1
                        v_lbs[i+1] = r1 * lb2
                        r2 = common_middle_r(wave1, wave2, ub1, -ub2)
                        v_ubs[i+1] = r2 * ub2
                    elif lb2 < 0.0:
                        # 'lb2' competes
                        r = common_middle_r(wave1, wave2, ub1, -lb2)
                        v_ubs[i]   = r * ub1
                        v_lbs[i+1] = r * lb2
                    else:
                        # No competition
                        v_ubs[i]   = min((wave2 / wave1 - 1) * C_KMS, ub1)

            case (1, 1):
                if i == uniques.size - 1:
                    v_lbs[i] = lb1
                    v_ubs[i] = ub1
                else:
                    wave2 = uniques[i+1]
                    lb2 = lbs[i+1]
                    ub2 = ubs[i+1]

                    if ub2 < 0.0:
                        # 'lb2' and 'ub2' both compete
                        r1 = common_middle_r(wave1, wave2, lb1, -lb2)
                        v_lbs[i]   = r1 * lb1
                        r2 = common_middle_r(wave1, wave2, ub1, -lb2)
                        v_ubs[i]   = r2 * ub1
                        v_lbs[i+1] = r2 * lb2
                        r3 = common_middle_r(wave1, wave2, ub1, -ub2)
                        v_ubs[i+1] = r3 * ub2
                    elif lb2 < 0.0:
                        # 'lb2' competes
                        r1 = common_middle_r(wave1, wave2, lb1, -lb2)
                        v_lbs[i]   = r1 * lb1
                        r2 = common_middle_r(wave1, wave2, ub1, -lb2)
                        v_ubs[i]   = r2 * ub1
                        v_lbs[i+1] = r2 * lb2
                    else:
                        # No competition
                        v_lbs[i]   = min((wave2 / wave1 - 1) * C_KMS, lb1)
                        v_ubs[i]   = min((wave2 / wave1 - 1) * C_KMS, ub1)

            case (1, -1) | (1, 0) | (0, -1):
                raise ValueError

    for i, wave in enumerate(uniques):
        vmin = (x_bounds[0] / wave - 1) * C_KMS
        vmax = (x_bounds[1] / wave - 1) * C_KMS

        v_lbs[i] = _clip(v_lbs[i], vmin, vmax)
        v_ubs[i] = _clip(v_ubs[i], vmin, vmax)

    for name, wave in waves.items():
        i = argwhere(uniques == wave).flatten()[0]
        v_off_bounds[name] = (v_lbs[i], v_ubs[i])
