from numpy import empty, float64, fromiter, unique
from numpy.typing import NDArray


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
        Velocity offset in units of the speed of light.
    vel2 : float or numpy.array
        Velocity offset in units of the speed of light.
    """
    return (wave1 * wave2 * (vel1 + vel2)) / (wave1 * vel1 + wave2 * vel2)


###


def update_v_off_bounds(
    *,
    v_off_bounds: dict[str, tuple[float, float]],
    lines: dict[str, float],
    x_bounds: tuple[float, float],
) -> dict[float, tuple[float, float]]:
    """
    Changes the 'v_off_bounds' dictionary inplace using the 'lines' dictionary.
    """
    # Get furthest extents of each line
    _v_off_bounds: dict[float, tuple[float, float]] = {}
    for name, line in lines.items():
        b = v_off_bounds[name]
        if line in v_off_bounds:
            _v_off_bounds[line] = (
                min(v_off_bounds[line][0], b[0]),
                max(v_off_bounds[line][1], b[1]),
            )
        else:
            _v_off_bounds[line] = b

    uniques = unique(list(lines.values()))
    lbs = fromiter((_v_off_bounds[line][0] for line in uniques), dtype=float64)
    ubs = fromiter((_v_off_bounds[line][1] for line in uniques), dtype=float64)
    del _v_off_bounds

    # Use furthest extents to find the middle points between each line,
    # and update the bounds
    _x = empty(uniques.size + 1)
    _x[0] = max(x_bounds[0], uniques[0] * (1 + lbs[0]))
    _x[1:-1] = common_middle(
        uniques[:-1],
        uniques[1:],
        -ubs[:-1],
        lbs[1:],  # ! Does this make sense?
    )
    _x[-1] = min(x_bounds[1], uniques[-1] * (1 + ubs[-1]))

    # Update the 'v_off_bounds' dictionary inplace with the new bounds
    x1_dict = dict(zip(uniques, _x[:-1]))
    x2_dict = dict(zip(uniques, _x[1:]))
    for name, line in lines.items():
        curr_bounds = v_off_bounds[name]
        v_off_bounds[name] = (
            max(curr_bounds[0], x1_dict[line] / line - 1),
            min(curr_bounds[1], x2_dict[line] / line - 1),
        )
