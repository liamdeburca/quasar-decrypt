from h5py import File, Group
from numpy import dtype, float32

from quasar_decrypt.host import HostWindows
from quasar_decrypt.spectrum import Spectrum

from .utils import specwindowlist_to_hdf5


def _validate(grp: Group) -> None:
    if grp.name.split("/")[-1] != "hg":
        raise ValueError(f"HostWindows group must be named 'hg', got {grp.name.split('/')[-1]!r}")

    if grp.attrs.get("class") != "HostWindows":
        raise ValueError(f"Group {grp.name!r} is not a HostWindows group")
    # empty group is allowed; no subwindows may be present
    missing = []
    for sub in grp.values():
        if sub.attrs.get("class") != "HWindow":
            missing.append(f"subgroup {sub.name} missing class=HWindow")
            continue
        if "x_bounds" not in sub:
            missing.append(f"{sub.name}/x_bounds")
    if missing:
        raise ValueError("HostWindows validation failed: " + ", ".join(missing))

def host_windows_to_hdf5(
    f: File | Group,
    host_windows: HostWindows,
    _dtype: str | dtype = float32,
    force: bool = False,
    **compression_kwargs,
) -> Group:
    """Create a `h5py.Group` with the following structure:

    hg/
      HWindow_0 : Group, see `specwindow_to_hdf5`
      HWindow_1 : ...

    Parameters
    ----------
    f : h5py.File | h5py.Group
        Destination HDF5 file or group.
    host_windows : HostWindows
        Collection of host windows to serialize.
    _dtype : str | numpy.dtype
        Numeric datatype used for float datasets.
    force : bool
        Overwrite existing group if True.

    Returns
    -------
    h5py.Group
        The created HostWindows group.

    Notes
    -----
    The function creates a group named 'hg' containing subgroups
    'HWindow_0', 'HWindow_1', ... each with a dataset 'x_bounds'.
    """
    grp = specwindowlist_to_hdf5(
        f,
        host_windows,
        group_name="hg",
        _dtype=_dtype,
        force=force,
        **compression_kwargs,
    )
    grp.attrs["format_version"] = 1

    for sub in grp.values():
        sub.attrs["format_version"] = 1

    _validate(grp)
    return grp


def hdf5_to_host_windows(grp: Group, spectrum: Spectrum) -> None:
    """Load host windows from HDF5 into a Spectrum.

    Parameters
    ----------
    grp : h5py.Group
        The HDF5 group containing HostWindows subgroups.
    spectrum : Spectrum
        The Spectrum object to populate.

    Returns
    -------
    None
    """

    _validate(grp)

    windows = []
    for sub in filter(lambda s: s.attrs.get("class") == "HWindow", grp.values()):
        windows.append(tuple(float(b) for b in sub["x_bounds"][()]))

    spectrum.instantiateHost(windows=windows)
