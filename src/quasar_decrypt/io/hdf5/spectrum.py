from pathlib import Path
from typing import Any

from h5py import File, Group
from numpy import dtype, float32, float64, ones
from quasar_utils.setup import Info

from quasar_decrypt.spectrum import Spectrum

from .balmer_windows import balmer_windows_to_hdf5, hdf5_to_balmer_windows
from .continuum_windows import (
    continuum_windows_to_hdf5,
    hdf5_to_continuum_windows,
)
from .host_windows import hdf5_to_host_windows, host_windows_to_hdf5
from .iron_windows import hdf5_to_iron_windows, iron_windows_to_hdf5
from .line_windows import hdf5_to_line_windows, line_windows_to_hdf5
from .utils import spec_to_hdf5


def _validate(grp: File | Group, info: Info) -> Group:
    if not str(grp.name).endswith("spectrum"):
        if "spectrum" not in grp:
            raise ValueError(f"Missing group 'spectrum': {list(grp.keys())}")
        grp = grp["spectrum"]

    assert "version" in grp.attrs
    assert "class" in grp.attrs
    assert "path" in grp.attrs
    assert "title" in grp.attrs
    assert "units_hash_algo" in grp.attrs
    assert "units_hash" in grp.attrs

    hash_algo = grp.attrs["units_hash_algo"]
    curr_hash = grp.attrs["units_hash"]
    my_hash = info.units.create_hash(hash_algo)
    if curr_hash != my_hash:
        raise ValueError(f"UnitsInfo has mismatch: {hash_algo}")

    if "spec" not in grp:
        raise ValueError(f"Spectrum group {grp.name!r} missing 'spec' sub-group")

    return grp

def spectrum_to_hdf5(
    spectrum: Spectrum,
    f: File | Group,
    _dtype: str | dtype = float32,
    force: bool = False,
    **compression_kwargs,
) -> Group:
    """
    Creates a `h5py.Group` with the following structure:

    spectrum/
             spec/ (see `spec_to_hdf5`)
             pl/   (see `continuum_windows_to_hdf5`)
             fe/   (see `iron_windows_to_hdf5`)
             ba/   (see `balmer_windows_to_hdf5`)
             hg/   (see `host_windows_to_hdf5`)
             em/   (see `line_windows_to_hdf5`)
    """
    if "spectrum" in f:
        if not force:
            raise ValueError("Group 'spectrum' already exists in the HDF5 file.")
        del f["spectrum"]

    grp = f.create_group("spectrum")
    grp.attrs["version"] = 1
    grp.attrs["class"] = "Spectrum"

    # Metadata
    grp.attrs["path"] = str(spectrum.path)
    grp.attrs["title"] = spectrum.title

    grp.attrs["units_hash_algo"] = "sha256"
    grp.attrs["units_hash"] = spectrum.info.units.create_hash("sha256")
    
    spec_to_hdf5(
        grp,
        spectrum,
        _dtype=_dtype,
        force=force,
        **compression_kwargs,
    )
    if spectrum.pl is not None:
        continuum_windows_to_hdf5(
            grp,
            spectrum.pl,
            _dtype=_dtype,
            force=force,
            **compression_kwargs,
        )
    if spectrum.fe is not None:
        iron_windows_to_hdf5(
            grp,
            spectrum.fe,
            _dtype=_dtype,
            force=force,
            **compression_kwargs,
        )
    if spectrum.ba is not None:
        balmer_windows_to_hdf5(
            grp,
            spectrum.ba,
            _dtype=_dtype,
            force=force,
            **compression_kwargs,
        )
    if spectrum.hg is not None:
        host_windows_to_hdf5(
            grp,
            spectrum.hg,
            _dtype=_dtype,
            force=force,
            **compression_kwargs,
        )
    if spectrum.em is not None:
        line_windows_to_hdf5(
            grp,
            spectrum.em,
            _dtype=_dtype,
            force=force,
            **compression_kwargs,
        )

    _validate(grp, spectrum.info)
    return grp



def hdf5_to_spectrum(
    f: File | Group,
    info: Info,
) -> Spectrum:
    """Load a Spectrum from an HDF5 group.

    The structure of the HDF5 group is expected to match the layout created by 
    `spectrum_to_hdf5`.

    Parameters
    ----------
    f : h5py.File | h5py.Group
        The HDF5 file or group containing the spectrum group.
    info : Info
        Setup/info object used to create the Spectrum.
    group_name : str | None
        Name of the group to read; defaults to "spectrum".

    Returns
    -------
    Spectrum
        The instantiated Spectrum object populated from HDF5.
    """
    grp = _validate(f, info)

    kwargs: dict[str, Any] = {
        "info": info,
        "path": Path(grp.attrs["path"]),
        "title": str(grp.attrs["title"]),
    }

    spec: Group = grp["spec"]
    kwargs["x_bounds"] = tuple(map(float, spec["x_bounds"][()]))
    kwargs["x0"] = spec["x0"][()]
    kwargs["y0"] = spec["y0"][()]

    kwargs["x"] = spec["x"][()]
    kwargs["y"] = spec["y"][()]
    kwargs["dy"] = spec["dy"][()]
    kwargs["dx"] = spec["dx"][()]

    kwargs["x_log"] = spec["x_log"][()]
    kwargs["y_log"] = spec["y_log"][()]
    kwargs["dy_log"] = spec["dy_log"][()]

    if "kernels" in spec:
        kwargs["kernels"] = spec["kernels"][()]
    else:
        kwargs["kernels"] = ones((kwargs["x"].size, 1), float64)

    if "y_smooth" in spec:
        kwargs["y_smooth"] = spec["y_smooth"][()]
    if "p_absorbed" in spec:
        kwargs["p_absorbed"] = spec["p_absorbed"][()]
    
    if "y_pl" in spec:
        kwargs["y_pl"] = spec["y_pl"][()]
    if "y_fe" in spec:
        kwargs["y_fe"] = spec["y_fe"][()]
    if "y_ba" in spec:
        kwargs["y_ba"] = spec["y_ba"][()]
    if "y_hg" in spec:
        kwargs["y_hg"] = spec["y_hg"][()]
    if "y_em" in spec:
        kwargs["y_em"] = spec["y_em"][()]

    if "rejected_pixels" in spec:
        kwargs["rejected_pixels"] = spec["rejected_pixels"][()]
    if "absorbed_pixels" in spec:
        kwargs["absorbed_pixels"] = spec["absorbed_pixels"][()]
    if "valid_pixels" in spec:
        kwargs["valid_pixels"] = spec["valid_pixels"][()]
    if "log_valid_pixels" in spec:
        kwargs["log_valid_pixels"] = spec["log_valid_pixels"][()]

    if "original" not in spec:
        # No original data, i.e. rebinned and original data are identical
        # -> Set the required fields using the rebinned data
        kwargs["x_original"] = kwargs["x"]
        kwargs["y_original"] = kwargs["y"]
        kwargs["dy_original"] = kwargs["dy"]
        kwargs["dx_original"] = kwargs["dx"]
        kwargs["kernels_original"] = kwargs["kernels"]
    else:
        original: Group = spec["original"]

        kwargs["x_original"] = original["x"][()]
        kwargs["y_original"] = original["y"][()]
        kwargs["dy_original"] = original["dy"][()]
        kwargs["dx_original"] = original["dx"][()]

        kwargs["x_log_original"] = original["x_log"][()]
        kwargs["y_log_original"] = original["y_log"][()]
        kwargs["dy_log_original"] = original["dy_log"][()]

        if "kernels" in original:
            kwargs["kernels_original"] = original["kernels"][()]
        else:
            kwargs["kernels_original"] = kwargs["kernels"]
            
        if "y_smooth" in original:
            kwargs["y_smooth_original"] = original["y_smooth"][()]
        if "p_absorbed" in original:
            kwargs["p_absorbed_original"] = original["p_absorbed"][()]

        if "y_pl" in original:
            kwargs["y_pl_original"] = original["y_pl"][()]
        if "y_fe" in original:
            kwargs["y_fe_original"] = original["y_fe"][()]
        if "y_ba" in original:
            kwargs["y_ba_original"] = original["y_ba"][()]
        if "y_hg" in original:
            kwargs["y_hg_original"] = original["y_hg"][()]
        if "y_em" in original:
            kwargs["y_em_original"] = original["y_em"][()]

        if "rejected_pixels" in original:
            kwargs["rejected_pixels_original"] = original["rejected_pixels"][()]
        if "absorbed_pixels" in original:
            kwargs["absorbed_pixels_original"] = original["absorbed_pixels"][()]
        if "valid_pixels" in original:
            kwargs["valid_pixels_original"] = original["valid_pixels"][()]
        if "log_valid_pixels" in original:
            kwargs["log_valid_pixels_original"] = original["log_valid_pixels"][()]

    spectrum = Spectrum.create(**kwargs)

    if "pl" in grp:
        hdf5_to_continuum_windows(grp["pl"], spectrum)
    if "fe" in grp:
        hdf5_to_iron_windows(grp["fe"], spectrum)
    if "ba" in grp:
        hdf5_to_balmer_windows(grp["ba"], spectrum)
    if "hg" in grp:
        hdf5_to_host_windows(grp["hg"], spectrum)
    if "em" in grp:
        hdf5_to_line_windows(grp["em"], spectrum)

    return spectrum