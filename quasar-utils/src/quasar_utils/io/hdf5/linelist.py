from __future__ import annotations

from io import StringIO
from typing import Any

from astropy import units as u
from h5py import File, Group, string_dtype
from numpy import ndarray
from pandas import DataFrame

from ...setup.info import Info
from ....linelist import LineList


def _h5py_to_str(ds_or_val: Any) -> str:
    """Robustly convert an h5py dataset or value to Python str.

    Accepts h5py.Dataset-like objects or raw values (bytes, numpy.bytes_,
    str) and returns a decoded UTF-8 str.
    """
    try:
        # ds.asstr()[()] works for variable-length string dtypes
        return ds_or_val.asstr()[()]
    except Exception:
        val = (
            ds_or_val[()]
            if hasattr(ds_or_val, "__array__") or hasattr(ds_or_val, "value")
            else ds_or_val
        )
        if isinstance(val, (bytes, bytearray)):
            return val.decode("utf-8")
        return str(val)


def linelist_to_hdf5(
    linelist: LineList,
    f: File | Group,
    group_name: str = "quasar_utils/linelist",
    info: Info | None = None,
    force: bool = False,
    compression: str | None = "gzip",
    compression_opts: int | None = None,
) -> Group:
    """Write a LineList into an HDF5 file.

    The layout is:
      /<group_name>/columns/<colname>  -- dataset per column
      /<group_name>/units/<semantic>   -- unit strings for dimensions

    The function requires an Info instance to obtain canonical unit strings
    for writing dimensional columns. If info is None, unit attributes are
    omitted.
    """
    if group_name in f:
        if not force:
            raise ValueError(
                f"Group '{group_name}' already exists in the HDF5 file."
            )
        del f[group_name]

    grp = f.create_group(group_name)
    grp.attrs["format_version"] = "1"

    cols = grp.create_group("columns")
    units_grp = grp.create_group("units")

    # canonical unit strings from Info if provided
    unit_map: dict[str, str] = {}
    if info is not None:
        unit_map = {
            "wavelength": str(info.units.wavelength_unit),
            "velocity": str(info.units.velocity_unit),
            "strength": str(info.units.strength_unit),
        }

    str_dt = string_dtype(encoding="utf-8")

    # write columns
    for col in linelist.columns:
        data = linelist[col].to_numpy()
        # Choose dtype / write as dataset
        dset = cols.create_dataset(
            col,
            data=data,
            compression=compression,
            compression_opts=compression_opts,
        )

        # attach unit attribute for known semantic columns
        if col == "wave":
            if info is not None:
                dset.attrs["unit"] = unit_map["wavelength"]
                units_grp.create_dataset(
                    "wavelength", data=unit_map["wavelength"], dtype=str_dt
                )
        if col.startswith("v_") or col.startswith("fwhm"):
            if info is not None and "velocity" not in units_grp:
                dset.attrs["unit"] = unit_map["velocity"]
                # ensure units group has an entry
                units_grp.create_dataset(
                    "velocity", data=unit_map["velocity"], dtype=str_dt
                )
        if col.startswith("strength"):
            if info is not None and "strength" not in units_grp:
                dset.attrs["unit"] = unit_map["strength"]
                units_grp.create_dataset(
                    "strength", data=unit_map["strength"], dtype=str_dt
                )

    return grp


def hdf5_to_linelist(
    f: File | Group, info: Info, group_name: str = "quasar_utils/linelist"
) -> LineList:
    """Read a LineList from HDF5 and convert dimensional columns to the
    units given by the provided Info instance.

    The function requires an Info instance to convert stored quantities
    (which are saved with unit metadata) into the caller's preferred units
    using info.units.get*(...) helpers.
    """
    if group_name not in f:
        raise ValueError(
            f"Group '{group_name}' does not exist in the HDF5 file."
        )

    grp = f[group_name]
    cols = grp["columns"]

    # build units map from units group and per-dataset attrs
    units_map: dict[str, str | None] = {}
    if "units" in grp:
        for k, v in grp["units"].items():
            units_map[k] = _h5py_to_str(v)

    data: dict[str, Any] = {}
    for name, dset in cols.items():
        arr = dset[()]
        # decode string arrays
        if isinstance(arr, (bytes, bytearray)):
            arr = _h5py_to_str(arr)
        # numpy dtype string-like -> decode elements
        try:
            import numpy as _np

            if _np.issubdtype(arr.dtype, _np.bytes_):
                arr = _np.char.decode(arr, "utf-8")
        except Exception:
            pass

        # determine unit for this column
        unit = None
        if "unit" in dset.attrs:
            unit = _h5py_to_str(dset.attrs["unit"])
        else:
            # map known semantic columns
            if name == "wave":
                unit = units_map.get("wavelength")
            elif name.startswith("v_") or name.startswith("fwhm"):
                unit = units_map.get("velocity")
            elif name.startswith("strength"):
                unit = units_map.get("strength")

        # convert if unit is present and info is provided
        if unit is not None:
            # parse as astropy Unit and convert using info.units helpers
            try:
                quant = u.Quantity(arr, u.Unit(unit))
                # route by semantic
                if name == "wave":
                    converted = info.units.getUnitlessWavelength(quant)
                elif name.startswith("v_") or name.startswith("fwhm"):
                    # use getUnitlessKMS for velocities (expects Quantity in some velocity unit)
                    converted = info.units.getUnitlessVelocity(quant)
                elif name.startswith("strength"):
                    converted = info.units.getUnitlessStrength(quant)
                else:
                    # fallback: attempt to convert to info.units.wavelength_unit
                    converted = quant.to(info.units.wavelength_unit).value
                data[name] = converted
            except Exception:
                # if conversion fails, fall back to raw array
                data[name] = arr
        else:
            data[name] = arr

    df = DataFrame(data)
    # ensure column order preserved
    df = df[[c for c in cols.keys()]]
    return LineList(df)
