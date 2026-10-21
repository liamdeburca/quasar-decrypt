"""Utilities for serializing parts of a ``_Spec`` object into HDF5 groups.

Parameters
----------
None

Returns
-------
None
"""
from functools import partial
from typing import Any

from h5py import File, Group
from numpy import bool_, dtype, float32, isfinite
from scipy.sparse import issparse

from quasar_decrypt.utils._spec import _Spec
from quasar_decrypt.utils._specwindow import _SpecWindow
from quasar_decrypt.utils._specwindowlist import _SpecWindowList


def _cd_f(
    grp: Group, 
    name: str, 
    data: Any, 
    _dtype: str | dtype, 
    **compression_kwargs,
) -> None:
    grp.create_dataset(name, data=data, dtype=_dtype, **compression_kwargs)

def _cd_b(
    grp: Group, 
    name: str, 
    data: Any, 
    **compression_kwargs,
) -> None:
    grp.create_dataset(name, data=data, dtype=bool_, **compression_kwargs)

def spec_to_hdf5(
    f: File | Group,
    spec: _Spec,
    _dtype: str | dtype = float32,
    force: bool = False,
    **compression_kwargs,
)-> Group:
    """Create a `h5py.Group` with the following structure:

    Spec/
        array datasets
        original/
                original array datasets

    Parameters
    ----------
    f : h5py.File | h5py.Group
        Destination HDF5 file or group.
    spec : _Spec
        The spec object to serialize.
    group_name : str | None
        Name for the created group. If None the spec class name is used.
    _dtype : str | numpy.dtype
        Numeric datatype for float datasets.
    force : bool
        Overwrite an existing group when True.

    Returns
    -------
    h5py.Group
        The created HDF5 group containing the serialized spec.
    
    Notes
    -----
    The created group contains datasets for bounds and rebinned arrays
    (x, y, dy, dx), optional original subgroup 'original' with the same
    structure, and optional support datasets such as 'kernels',
    'y_smooth', 'p_absorbed', and pixel masks.
    """
    if "spec" in f:
        if not force:
            raise ValueError("Group 'spec' already exists in the HDF5 file.")
        del f["spec"]

    grp = f.create_group("spec")

    cd_f = partial(_cd_f, _dtype=_dtype, **compression_kwargs)
    cd_b = partial(_cd_b, **compression_kwargs)

    cd_f(grp, "x_bounds", spec.x_bounds)
    cd_f(grp, "x0", spec.x0)
    cd_f(grp, "y0", spec.y0)

    cd_f(grp, "x", spec._x)
    cd_f(grp, "y", spec._y)
    cd_f(grp, "dy", spec._dy)
    cd_f(grp, "dx", spec._dx)

    cd_f(grp, "x_log", spec._x_log)
    cd_f(grp, "y_log", spec._y_log)
    cd_f(grp, "dy_log", spec._dy_log)

    if spec._kernels.shape[1] > 1:
        # Do not save kernels if they only have a single column.
        kernels = spec._kernels
        if issparse(kernels):
            kernels = kernels.toarray()
        cd_f(grp, "kernels", kernels)

    if isfinite(spec._y_smooth).any():
        # Only save y_smooth if it has been modified
        cd_f(grp, "y_smooth", spec._y_smooth)
    if (spec._p_absorbed != 1.0).any():
        # Only save p_absorbed if it has been modified
        cd_f(grp, "p_absorbed", spec._p_absorbed)
    
    if (spec._y_pl != 0.0).any():
        # Only save y_pl if it has been modified
        cd_f(grp, "y_pl", spec._y_pl)
    if (spec._y_fe != 0.0).any():
        # Only save y_fe if it has been modified
        cd_f(grp, "y_fe", spec._y_fe)
    if (spec._y_ba != 0.0).any():
        # Only save y_ba if it has been modified
        cd_f(grp, "y_ba", spec._y_ba)
    if (spec._y_hg != 0.0).any():
        # Only save y_hg if it has been modified
        cd_f(grp, "y_hg", spec._y_hg)
    if (spec._y_em != 0.0).any():
        # Only save y_em if it has been modified
        cd_f(grp, "y_em", spec._y_em)

    if spec._rejected_pixels.any():
        # Only save rejected_pixels if there are any
        cd_b(grp, "rejected_pixels", spec._rejected_pixels)
    if spec._absorbed_pixels.any():
        # Only save absorbed_pixels if there are any
        cd_b(grp, "absorbed_pixels", spec._absorbed_pixels)
    if not spec._valid_pixels.all():
        # Only save valid_pixels if there are non-valid pixels
        cd_b(grp, "valid_pixels", spec._valid_pixels)
    if not spec._log_valid_pixels.all():
        # Only save log_valid_pixels if there are non-log-valid pixels
        cd_b(grp, "log_valid_pixels", spec._log_valid_pixels)

    if not spec._no_original:
        orig = grp.create_group("original")

        cd_f(orig, "x", spec._x_original)
        cd_f(orig, "y", spec._y_original)
        cd_f(orig, "dy", spec._dy_original)
        cd_f(orig, "dx", spec._dx_original)

        cd_f(orig, "x_log", spec._x_log_original)
        cd_f(orig, "y_log", spec._y_log_original)
        cd_f(orig, "dy_log", spec._dy_log_original)

        if spec._kernels.shape[1] > 1:
            # Do not save kernels if they only have a single column.
            kernels = spec._kernels_original
            if issparse(kernels):
                kernels = kernels.toarray()
            cd_f(orig, "kernels", kernels)

        if isfinite(spec._y_smooth_original).any():
            # Only save y_smooth if it has been modified
            cd_f(orig, "y_smooth", spec._y_smooth_original)
        if (spec._p_absorbed_original != 1.0).any():
            # Only save p_absorbed if it has been modified
            cd_f(orig, "p_absorbed", spec._p_absorbed_original)
        
        if (spec._y_pl_original != 0.0).any():
            # Only save y_pl if it has been modified
            cd_f(orig, "y_pl", spec._y_pl_original)
        if (spec._y_fe_original != 0.0).any():
            # Only save y_fe if it has been modified
            cd_f(orig, "y_fe", spec._y_fe_original)
        if (spec._y_ba_original != 0.0).any():
            # Only save y_ba if it has been modified
            cd_f(orig, "y_ba", spec._y_ba_original)
        if (spec._y_hg_original != 0.0).any():
            # Only save y_hg if it has been modified
            cd_f(orig, "y_hg", spec._y_hg_original)
        if (spec._y_em_original != 0.0).any():
            # Only save y_em if it has been modified
            cd_f(orig, "y_em", spec._y_em_original)

        if spec._rejected_pixels_original.any():
            # Only save rejected_pixels if there are any
            cd_b(orig, "rejected_pixels", spec._rejected_pixels_original)
        if spec._absorbed_pixels_original.any():
            # Only save absorbed_pixels if there are any
            cd_b(orig, "absorbed_pixels", spec._absorbed_pixels_original)
        if not spec._valid_pixels_original.all():
            # Only save valid_pixels if there are non-valid pixels
            cd_b(orig, "valid_pixels", spec._valid_pixels_original)
        if not spec._log_valid_pixels_original.all():
            # Only save log_valid_pixels if there are non-log-valid pixels
            cd_b(orig, "log_valid_pixels", spec._log_valid_pixels_original)

    return grp


def specwindow_to_hdf5(
    f: File | Group,
    specwindow: _SpecWindow,
    group_name: str | None = None,
    _dtype: str | dtype = float32,
    force: bool = False,
    **compression_kwargs,
)-> Group:
    """
    Create a `h5py.Group` with the following structure:

    group_name/
              x_bounds : dataset (2,)

    Parameters
    ----------
    specwindow : _SpecWindow
        The spec window to serialize.
    f : h5py.File | h5py.Group
        Destination HDF5 file or group.
    group_name : str | None
        Name for the created subgroup. If None uses the specwindow class name.
    _dtype : str | numpy.dtype
        Numeric datatype for float datasets.
    force : bool
        Overwrite an existing group when True.

    Returns
    -------
    h5py.Group
        The created subgroup representing the spec window.
    
    Notes
    -----
    The created subgroup is named <SpecWindowClass>_N and contains a
    single dataset 'x_bounds'.
    """
    if group_name is None:
        group_name = specwindow.__class__.__name__

    group_name = group_name.lower()
    if group_name in f:
        if not force:
            raise ValueError(f"Group '{group_name}' already exists in the HDF5 file.")
        del f[group_name]

    grp = f.create_group(group_name)
    grp.attrs["class"] = specwindow.__class__.__name__

    grp.create_dataset(
        "x_bounds",
        data=specwindow.x_bounds,
        dtype=_dtype,
        **compression_kwargs,
    )
    return grp


def specwindowlist_to_hdf5(
    f: File | Group,
    specwindowlist: _SpecWindowList[_SpecWindow],
    group_name: str | None = None,
    _dtype: str | dtype = float32,
    force: bool = False,
    **compression_kwargs,
)-> Group:
    """Create a `h5py.Group` with the following structure:

    group_name/
              group_0 : Group, see `specwindow_to_hdf5`
              group_1 : ...

    Parameters
    ----------
    f : h5py.File | h5py.Group
        Destination HDF5 file or group.
    specwindowlist : _SpecWindowList[_SpecWindow]
        Iterable of spec windows to serialize.
    group_name : str | None
        Name for the created parent group. If None the class name is used.
    _dtype : str | numpy.dtype
        Numeric datatype for float datasets.
    force : bool
        Overwrite an existing group when True.

    Returns
    -------
    h5py.Group
        The created group that contains the serialized spec windows.
    
    Notes
    -----
    The parent group (named by group_name) contains subgroups named
    '<SpecWindowClass>_0', '<SpecWindowClass>_1', ... each with
    an 'x_bounds' dataset.
    """
    if group_name is None:
        group_name = specwindowlist.__class__.__name__

    group_name = group_name.lower()
    if group_name in f:
        if not force:
            raise ValueError(f"Group '{group_name}' already exists in the HDF5 file.")
        del f[group_name]

    grp = f.create_group(group_name)
    grp.attrs["class"] = specwindowlist.__class__.__name__

    for i, specwindow in enumerate(specwindowlist):
        specwindow_to_hdf5(
            grp,
            specwindow,
            group_name=f"{specwindow.__class__.__name__}_{i}".lower(),
            _dtype=_dtype,
            force=force,
            **compression_kwargs,
        )

    return grp
