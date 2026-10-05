"""
Utilities for extracting data arrays from a `_Spec` object and storing them in a 
HDF5 group.
"""
from functools import partial
from typing import Any

from h5py import File, Group
from numpy import bool_, dtype, float32, isfinite
from scipy.sparse import issparse

from ...utils._spec import _Spec
from ...utils._specwindow import _SpecWindow
from ...utils._specwindowlist import _SpecWindowList


def _cd_f(
    grp: Group, 
    name: str, 
    data: Any, 
    _dtype: str | dtype, 
    compression: str | None, 
    compression_opts: int | None,
) -> None:
    grp.create_dataset(
        name, 
        data=data, 
        dtype=_dtype, 
        compression=compression, 
        compression_opts=compression_opts,
    )

def _cd_b(
    grp: Group, 
    name: str, 
    data: Any, 
    compression: str | None, 
    compression_opts: int | None,
) -> None:
    grp.create_dataset(
        name, 
        data=data, 
        dtype=bool_, 
        compression=compression, 
        compression_opts=compression_opts,
    )

def spec_to_hdf5(
    spec: _Spec,
    f: File | Group,
    group_name: str | None = None,
    _dtype: str | dtype = float32,
    force: bool = False,
    compression: str | None = "gzip",
    compression_opts: int | None = None,
) -> Group:
    if group_name is None:
        group_name = spec.__class__.__name__

    if group_name in f:
        if not force:
            raise ValueError(f"Group '{group_name}' already exists in the HDF5 file.")
        del f[group_name]

    grp = f.create_group(group_name)

    cd_f = partial(
        _cd_f, 
        _dtype=_dtype, 
        compression=compression, 
        compression_opts=compression_opts,
    )
    cd_b = partial(
        _cd_b, 
        _dtype=bool_,
        compression=compression, 
        compression_opts=compression_opts,
    )

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
            cd_f(orig, "rejected_pixels", spec._rejected_pixels_original)
        if spec._absorbed_pixels_original.any():
            # Only save absorbed_pixels if there are any
            cd_f(orig, "absorbed_pixels", spec._absorbed_pixels_original)
        if not spec._valid_pixels_original.all():
            # Only save valid_pixels if there are non-valid pixels
            cd_f(orig, "valid_pixels", spec._valid_pixels_original)
        if not spec._log_valid_pixels_original.all():
            # Only save log_valid_pixels if there are non-log-valid pixels
            cd_f(orig, "log_valid_pixels", spec._log_valid_pixels_original)

    return grp


def specwindow_to_hdf5(
    specwindow: _SpecWindow,
    f: File | Group,
    group_name: str | None = None,
    _dtype: str | dtype = float32,
    force: bool = False,
    compression: str | None = None,
    compression_opts: int | None = None,
) -> Group:
    if group_name is None:
        group_name = specwindow.__class__.__name__

    if group_name in f:
        if not force:
            raise ValueError(f"Group '{group_name}' already exists in the HDF5 file.")
        del f[group_name]

    grp = f.create_group(group_name)
    grp.create_dataset(
        "x_bounds",
        data=specwindow.x_bounds,
        dtype=_dtype,
        compression=compression,
        compression_opts=compression_opts,
    )
    return grp


def specwindowlist_to_hdf5(
    specwindowlist: _SpecWindowList[_SpecWindow],
    f: File | Group,
    group_name: str | None = None,
    _dtype: str | dtype = float32,
    force: bool = False,
    compression: str | None = None,
    compression_opts: int | None = None,
) -> Group:
    if group_name is None:
        group_name = specwindowlist.__class__.__name__

    if group_name in f:
        if not force:
            raise ValueError(f"Group '{group_name}' already exists in the HDF5 file.")
        del f[group_name]

    grp = f.create_group(group_name)
    for i, specwindow in enumerate(specwindowlist):
        specwindow_to_hdf5(
            specwindow,
            grp,
            group_name=f"{specwindow.__class__.__name__}_{i}",
            _dtype=_dtype,
            force=force,
            compression=compression,
            compression_opts=compression_opts,
        )

    return grp
