from typing import Any

from h5py import File, Group, string_dtype
from numpy import dtype, float64, ones
from quasar_utils.setup import Info

from ...spectrum import Spectrum
from .utils import spec_to_hdf5, specwindowlist_to_hdf5


def spectrum_to_hdf5(
    spectrum: Spectrum,
    f: File | Group,
    group_name: str | None = None,
    _dtype: str | dtype = "f",
    force: bool = False,
    compression: str | None = None,
    compression_opts: int | None = None,
) -> Group:
    if group_name is None:
        group_name = spectrum.__class__.__name__

    if group_name in f:
        if not force:
            raise ValueError(f"Group '{group_name}' already exists in the HDF5 file.")
        del f[group_name]

    grp = f.create_group(group_name)
    
    if "meta" in f:
        if not force:
            raise ValueError("Group 'meta' already exists in the HDF5 file.")
        del f["meta"]
    
    meta = grp.create_group("meta")

    str_dt = string_dtype(encoding="utf-8")
    meta.create_dataset("path", data=str(spectrum.path), dtype=str_dt)
    meta.create_dataset("title", data=spectrum.title, dtype=str_dt)

    spec_to_hdf5(
        spectrum,
        grp,
        group_name="spec",
        _dtype=_dtype,
        force=force,
        compression=compression,
        compression_opts=compression_opts,
    )

    if spectrum.pl is not None:
        specwindowlist_to_hdf5(
            spectrum.pl,
            grp,
            group_name="pl",
            _dtype=_dtype,
            force=force,
            compression=compression,
            compression_opts=compression_opts,
        )
    if spectrum.fe is not None:
        specwindowlist_to_hdf5(
            spectrum.fe,
            grp,
            group_name="fe",
            _dtype=_dtype,
            force=force,
            compression=compression,
            compression_opts=compression_opts,
        )
    if spectrum.ba is not None:
        specwindowlist_to_hdf5(
            spectrum.ba,
            grp,
            group_name="ba",
            _dtype=_dtype,
            force=force,
            compression=compression,
            compression_opts=compression_opts,
        )
    if spectrum.hg is not None:
        specwindowlist_to_hdf5(
            spectrum.hg,
            grp,
            group_name="hg",
            _dtype=_dtype,
            force=force,
            compression=compression,
            compression_opts=compression_opts,
        )
    if spectrum.em is not None:
        specwindowlist_to_hdf5(
            spectrum.em,
            grp,
            group_name="em",
            _dtype=_dtype,
            force=force,
            compression=compression,
            compression_opts=compression_opts,
        )

    return grp


def hdf5_to_spectrum(
    f: File | Group,
    info: Info,
    group_name: str | None = None,
) -> Spectrum:
    if group_name is None:
        group_name = "Spectrum"

    if group_name not in f:
        raise ValueError(f"Group '{group_name}' does not exist in the HDF5 file.")

    grp = f[group_name]
    meta = grp["meta"]

    kwargs: dict[str, Any] = {
        "info": info,
        "path": meta["path"].asstr()[()],
        "title": meta["title"].asstr()[()],
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
    if "invalid_pixels" in spec:
        kwargs["invalid_pixels"] = spec["invalid_pixels"][()]

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

    spectrum = Spectrum.create.__wrapped__(Spectrum, **kwargs)

    if "pl" in grp:
        windows = [
            tuple(window["x_bounds"][()]) 
            for window in grp["pl"].values()
        ]
        spectrum.instantiateContinuum(windows=windows)
    if "fe" in grp:
        windows = [
            tuple(window["x_bounds"][()]) 
            for window in grp["fe"].values()
        ]
        spectrum.instantiateIron(windows=windows)
    if "ba" in grp:
        windows = [
            tuple(window["x_bounds"][()]) 
            for window in grp["ba"].values()
        ]
        spectrum.instantiateBalmer(windows=windows)
    if "hg" in grp:
        windows = [
            tuple(window["x_bounds"][()]) 
            for window in grp["hg"].values()
        ]
        spectrum.instantiateHost(windows=windows)

    return spectrum