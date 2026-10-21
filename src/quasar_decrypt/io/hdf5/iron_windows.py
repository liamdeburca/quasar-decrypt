from h5py import File, Group
from numpy import dtype, float32

from quasar_decrypt.iron import IronWindows
from quasar_decrypt.spectrum import Spectrum

from .utils import specwindowlist_to_hdf5


def _validate(grp: Group) -> None:
    if grp.name.split("/")[-1] != "fe":
        raise ValueError(f"IronWindows group must be named 'fe', got {grp.name.split('/')[-1]!r}")

    if grp.attrs.get("class") != "IronWindows":
        raise ValueError(f"Group {grp.name!r} is not an IronWindows group")
    # empty group is allowed; no subwindows may be present
    missing = []
    for sub in grp.values():
        if sub.attrs.get("class") != "IWindow":
            missing.append(f"subgroup {sub.name} missing class=IWindow")
            continue
        if "x_bounds" not in sub:
            missing.append(f"{sub.name}/x_bounds")
    if missing:
        raise ValueError("IronWindows validation failed: " + ", ".join(missing))

def iron_windows_to_hdf5(
    f: File | Group,
    iron_windows: IronWindows,
    _dtype: str | dtype = float32,
    force: bool = False,
    **compression_kwargs,
) -> Group:
    """Create a `h5py.Group` with the following structure:

    fe/
      IWindow_0 : Group, see `specwindow_to_hdf5`
      IWindow_1 : ...

    Parameters
    ----------
    f : h5py.File | h5py.Group
        Destination HDF5 file or group.
    iron_windows : IronWindows
        Collection of iron windows to serialize.
    _dtype : str | numpy.dtype
        Numeric datatype used for float datasets.
    force : bool
        Overwrite existing group if True.

    Returns
    -------
    h5py.Group
        The created IronWindows group.

    Notes
    -----
    The function creates a group named 'fe' containing subgroups
    'IWindow_0', 'IWindow_1', ... each with a dataset 'x_bounds'.
    """
    grp = specwindowlist_to_hdf5(
        f,
        iron_windows,
        group_name="fe",
        _dtype=_dtype,
        force=force,
        **compression_kwargs,
    )
    grp.attrs["format_version"] = 1

    for sub in grp.values():
        sub.attrs["format_version"] = 1

    _validate(grp)
    return grp


def hdf5_to_iron_windows(grp: Group, spectrum: Spectrum) -> None:
    """Load iron windows from HDF5 into a Spectrum.

    Parameters
    ----------
    grp : h5py.Group
        The HDF5 group containing IronWindows subgroups.
    spectrum : Spectrum
        The Spectrum object to populate.

    Returns
    -------
    None
    """

    _validate(grp)

    windows = []
    for sub in filter(lambda s: s.attrs.get("class") == "IWindow", grp.values()):
        windows.append(tuple(float(b) for b in sub["x_bounds"][()]))

    spectrum.instantiateIron(windows=windows)
