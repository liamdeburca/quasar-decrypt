from functools import partial
from math import isfinite

from h5py import File, Group, string_dtype
from numpy import asarray, bool_, dtype, float32, int8, nan, uint8

from quasar_decrypt.lines import LineWindows, LWindow
from quasar_decrypt.spectrum import Spectrum

from .utils import specwindow_to_hdf5


def _strength_bounds_get(lwindow, name) -> tuple[float, float]:
    val = lwindow.strength_bounds.get(name)
    if val is None:
        return (nan, nan)
    return val


def _fwhm_v_bounds_get(lwindow, name) -> tuple[float, float]:
    val = lwindow.fwhm_v_bounds.get(name)
    if val is None:
        return (nan, nan)
    return val


def _v_off_bounds_get(lwindow, name) -> tuple[float, float]:
    val = lwindow.v_off_bounds.get(name)
    if val is None:
        return (nan, nan)
    return val


def _scale_bounds_get(lwindow, name) -> tuple[float, float]:
    val = lwindow.scale_bounds.get(name)
    if val is None:
        return (nan, nan)
    return val

def _pair_arr(names, getter):
    out = []
    for name in names:
        val = getter(name)
        if val is None:
            out.append((nan, nan))
        else:
            out.append(val)
    return out


def _validate(grp: Group) -> None:
    if grp.name.split("/")[-1] != "em":
        raise ValueError(f"LineWindows group must be named 'em', got {grp.name.split('/')[-1]!r}")

    if grp.attrs.get("class") != "LineWindows":
        raise ValueError(f"Group {grp.name!r} is not a LineWindows group")
    
    # empty group is allowed; no subwindows may be present
    
    missing = []
    for sub in grp.values():
        if sub.attrs.get("class") != "LWindow":
            missing.append(f"subgroup {sub.name} missing class=LWindow")
            continue

        for ds in ("names", "complexes", "linetypes", "waves", "n_maxs"):
            if ds not in sub:
                missing.append(f"{sub.name}/{ds}")
    if missing:
        raise ValueError("LineWindows validation failed: " + ", ".join(missing))

### Saving

def lwindow_to_hdf5(
    f: File | Group, 
    lwindow: LWindow, 
    group_name: str | None = None,
    _dtype: str | dtype = float32,
    force: bool = False,
    **compression_kwargs,
) -> Group:
    """Create a `h5py.Group` with the following structure:

    sub/
       ...

    Parameters
    ----------
    f : h5py.File | h5py.Group
        Destination HDF5 file or group.
    lwindow : LWindow
        The LWindow instance to serialize.
    _dtype : str | numpy.dtype
        Numeric datatype used for float datasets.

    Returns
    -------
    h5py.Group
        The subgroup that was written to.
    """
    grp = specwindow_to_hdf5(
        f, lwindow, 
        group_name=group_name,
        _dtype=_dtype,
        force=force,
        **compression_kwargs,
    )

    names = sorted(lwindow.names, key=lambda n: lwindow.waves.get(n, 0.0))
    n = len(names)
    str_dt = string_dtype(encoding="utf-8")

    grp.create_dataset(
        "names",
        shape=(n,),
        data=names,
        dtype=str_dt,
        **compression_kwargs,
    )
    grp.create_dataset(
        "complexes",
        shape=(n,),
        data=[lwindow.complexes[nm] for nm in names],
        dtype=str_dt,
        **compression_kwargs,
    )
    grp.create_dataset(
        "linetypes",
        shape=(n,),
        data=[lwindow.linetypes[nm] for nm in names],
        dtype=str_dt,
        **compression_kwargs,
    )
    grp.create_dataset(
        "waves",
        shape=(n,),
        data=asarray([lwindow.waves[name] for name in names], _dtype),
        dtype=_dtype,
        **compression_kwargs,
    )
    grp.create_dataset(
        "n_maxs",
        shape=(n,),
        data=asarray([lwindow.n_maxs[name] for name in names], uint8),
        dtype=uint8,  # Save as 8-bit unsigned integer: min=0, max=255
        **compression_kwargs,
    )
    grp.create_dataset(
        "needs_line",
        shape=(n,),
        data=[lwindow.needs_line.get(name, "") for name in names],
        dtype=str_dt,
        **compression_kwargs,
    )
    grp.create_dataset(
        "scale_init",
        shape=(n,),
        data=asarray([lwindow.scale_init.get(name, nan) for name in names], _dtype),
        dtype=_dtype,
        **compression_kwargs,
    )
    grp.create_dataset(
        "scale_fixed",
        shape=(n,),
        data=asarray([lwindow.scale_fixed.get(name, False) for name in names], bool_),
        dtype=bool_,
        **compression_kwargs,
    )
    grp.create_dataset(
        "is_copy_of",
        shape=(n,),
        data=[lwindow.is_copy_of.get(name, "") or "" for name in names],
        dtype=str_dt,
        **compression_kwargs,
    )
    grp.create_dataset(
        "is_multiplet_of",
        shape=(n,),
        data=[lwindow.is_multiplet_of.get(name, "") or "" for name in names],
        dtype=str_dt,
        **compression_kwargs,
    )
    grp.create_dataset(
        "blacklist",
        shape=(n,),
        data=asarray([lwindow.blacklist.get(name, False) for name in names], bool_),
        dtype=bool_,
        **compression_kwargs,
    )
    grp.create_dataset(
        "_blacklist",
        shape=(n,),
        data=asarray([lwindow._blacklist.get(name, False) for name in names], bool_),
        dtype=bool_,
        **compression_kwargs,
    )
    grp.create_dataset(
        "cropped",
        shape=(n,),
        data=asarray([name in lwindow.cropped for name in names], bool_),
        dtype=bool_,
        **compression_kwargs,
    )

    ### Dictionaries

    pair_arr = partial(_pair_arr, names)

    grp.create_dataset(
        "i_bounds",
        shape=(n,2),
        data=asarray([lwindow.i_bounds[lwindow.complexes[name]] for name in names], _dtype),
        dtype=_dtype,
        **compression_kwargs,
    )

    grp.create_dataset(
        "strength_bounds",
        shape=(n, 2),
        data=asarray(pair_arr(partial(_strength_bounds_get, lwindow)), _dtype),
        dtype=_dtype,
        **compression_kwargs,
    )
    grp.create_dataset(
        "fwhm_v_bounds",
        shape=(n, 2),
        data=asarray(pair_arr(partial(_fwhm_v_bounds_get, lwindow)), _dtype),
        dtype=_dtype,
        **compression_kwargs,
    )
    grp.create_dataset(
        "v_off_bounds",
        shape=(n, 2),
        data=asarray(pair_arr(partial(_v_off_bounds_get, lwindow)), _dtype),
        dtype=_dtype,
        **compression_kwargs,
    )
    grp.create_dataset(
        "scale_bounds",
        shape=(n, 2),
        data=asarray(pair_arr(partial(_scale_bounds_get, lwindow)), _dtype),
        dtype=_dtype,
        **compression_kwargs,
    )

    ### Velocity profile copy relationships

    flattened_idx: list[int] = []
    flattened_name: list[str] = []
    counts: list[int] = []
    for name in names:
        lst = lwindow.copies_to.get(name, [])
        counts.append(len(lst))
        for idx, mimic in lst:
            flattened_idx.append(-1 if idx is None else int(idx))
            flattened_name.append(mimic)

    grp.create_dataset(
        "copies_to_counts",
        data=asarray(counts, uint8),
        **compression_kwargs,
    )
    grp.create_dataset(
        "copies_to_idx",
        data=asarray(flattened_idx, int8),
        **compression_kwargs,
    )
    grp.create_dataset(
        "copies_to_name",
        data=flattened_name,
        dtype=str_dt,
        **compression_kwargs,
    )

    ### Neighbours

    grp.create_dataset(
        "neighbours",
        shape=(2,),
        data=asarray([(-1 if x is None else int(x)) for x in lwindow.neighbours], dtype=int8),
        dtype=int8,
        **compression_kwargs,
    )

    return grp

def line_windows_to_hdf5(
    f: File | Group,
    line_windows: LineWindows,
    _dtype: str | dtype = float32,
    force: bool = False,
    **compression_kwargs,
) -> Group:
    """Create a `h5py.Group` with the following structure:

    em/
      LWindow_0 : Group, see `lwindow_to_hdf5`
      LWindow_1 : ...

    Parameters
    ----------
    f : h5py.File | h5py.Group
        Destination HDF5 file or group.
    line_windows : LineWindows
        Collection of LWindow instances to write.
    _dtype : str | numpy.dtype
        Numeric datatype used for float datasets.
    force : bool
        Overwrite existing group if True.

    Returns
    -------
    h5py.Group
        The created LineWindows group.
    
    Notes
    -----
    Creates a top-level group named 'em' (when called from spectrum write).
    Inside it creates subgroups named 'LWindow_0', 'LWindow_1', ... each
    with class='LWindow' and the datasets described in :func:`lwindow_to_hdf5`.
    """
    group_name = "em"
    if group_name in f:
        if not force:
            raise ValueError(f"Group '{group_name}' already exists in the HDF5 file.")
        del f[group_name]

    grp = f.create_group(group_name)
    grp.attrs["class"] = "LineWindows"
    grp.attrs["format_version"] = 1

    for i, lwindow in enumerate(line_windows):
        sub = lwindow_to_hdf5(
            grp, lwindow, 
            group_name=f"LWindow_{i}", 
            force=force, 
            _dtype=_dtype, 
            **compression_kwargs,
        )
        sub.attrs["format_version"] = 1

    _validate(grp)
    return grp

### Loading 

def hdf5_to_lwindow(sub: Group, lwindow: LWindow) -> None:
    """Populate an LWindow from an HDF5 subgroup.

    Parameters
    ----------
    sub : h5py.Group
        The subgroup containing datasets for a single LWindow.
    lwindow : LWindow
        The LWindow object to populate.

    Returns
    -------
    None
    """
    # Clear existing data
    lwindow.names.clear()
    lwindow.linetypes.clear()
    lwindow.complexes.clear()
    lwindow.waves.clear()
    lwindow.n_maxs.clear()
    lwindow.needs_line.clear()

    lwindow.strength_bounds.clear()
    lwindow.fwhm_v_bounds.clear()
    lwindow.v_off_bounds.clear()

    lwindow.is_copy_of.clear()
    lwindow.is_multiplet_of.clear()

    lwindow.scale_init.clear()
    lwindow.scale_bounds.clear()
    lwindow.scale_fixed.clear()

    lwindow.copies_to.clear()

    lwindow.i_bounds.clear()
    lwindow.blacklist.clear()
    lwindow._blacklist.clear()
    
    lwindow.neighbours = (None, None)
    lwindow.model = None
    lwindow.fit = None
    lwindow.fit_info = None
    lwindow.fits.clear()
    lwindow.fit_infos.clear()

    lwindow.cropped.clear()
    lwindow.bootstrapper = None
    lwindow.error_result = None
    lwindow._y_em_contrib[:] = 0.0

    ###

    names = sub["names"].asstr()
    complexes = sub["complexes"].asstr()
    linetypes = sub["linetypes"].asstr()
    waves = sub["waves"]
    n_maxs = sub["n_maxs"]

    needs_line = sub["needs_line"].asstr()
    
    scale_init = sub["scale_init"]
    scale_fixed = sub["scale_fixed"]
    is_copy_of = sub["is_copy_of"].asstr()
    is_multiplet_of = sub["is_multiplet_of"].asstr()
    blacklist = sub["blacklist"]
    _blacklist = sub["_blacklist"]
    cropped = sub["cropped"]

    strength_bounds = sub["strength_bounds"]
    fwhm_v_bounds = sub["fwhm_v_bounds"]
    v_off_bounds = sub["v_off_bounds"]
    scale_bounds = sub["scale_bounds"]
    i_bounds = sub["i_bounds"]

    
    for i, name in enumerate(map(str, names)):
        lwindow.names.add(name)
        
        lwindow.linetypes[name] = str(linetypes[i])
        lwindow.waves[name] = float(waves[i])
        lwindow.n_maxs[name] = int(n_maxs[i])

        _complex = str(complexes[i])
        lwindow.complexes[name] = _complex
        if _complex not in lwindow.i_bounds:
            lwindow.i_bounds[_complex] = tuple(map(float, i_bounds[i]))


        if needs_line[i]:
            lwindow.needs_line[name] = str(needs_line[i])

        if isfinite(scale_init[i]):
            lwindow.scale_init[name] = float(scale_init[i])
            lwindow.scale_fixed[name] = bool(scale_fixed[i])
            lwindow.scale_bounds[name] = tuple(map(float, scale_bounds[i]))

        if is_copy_of[i]:
            lwindow.is_copy_of[name] = str(is_copy_of[i])
        if is_multiplet_of[i]:
            lwindow.is_multiplet_of[name] = str(is_multiplet_of[i])

        lwindow.blacklist[name] = bool(blacklist[i])
        lwindow._blacklist[name] = bool(_blacklist[i])

        if cropped[i]:
            lwindow.cropped.add(name)

        lwindow.strength_bounds[name] = tuple(map(float, strength_bounds[i]))
        lwindow.fwhm_v_bounds[name] = tuple(map(float, fwhm_v_bounds[i]))
        lwindow.v_off_bounds[name] = tuple(map(float, v_off_bounds[i]))

    ### `copies_to` dictionary

    copies_to_counts = sub["copies_to_counts"]
    copies_to_idx = sub["copies_to_idx"]
    copies_to_name = sub["copies_to_name"]

    m = len(copies_to_counts)
    s = 0
    for i, name in enumerate(map(str, names)):
        if i >= m:
            continue

        count = int(copies_to_counts[i])
        lst: list[tuple[int | None, str]] = []
        for _ in range(count):
            idx = int(copies_to_idx[s])
            if idx == -1:
                idx = None

            mimic = copies_to_name[s]
            lst.append((idx, mimic))
            s += 1
        
        if lst:
            lwindow.copies_to[name] = lst

    ### Neighbours

    lwindow.neighbours = tuple(
        None if int(idx) == -1 else int(idx)
        for idx in sub["neighbours"]
    )

def hdf5_to_line_windows(grp: Group, spectrum: Spectrum) -> None:
    """Load LineWindows from an HDF5 group into a Spectrum.

    Parameters
    ----------
    grp : h5py.Group
        The HDF5 group containing LineWindows subgroups.
    spectrum : Spectrum
        The Spectrum object to populate.

    Returns
    -------
    None
    """
    _validate(grp)

    spectrum.instantiateLines()
    for sub in filter(
        lambda s: s.attrs.get("class") == "LWindow", 
        grp.values(),
    ):
        lwindow = LWindow.create(
            spectrum=spectrum, 
            x_bounds=spectrum.x_bounds, 
        )
        hdf5_to_lwindow(sub, lwindow)
        spectrum.em.append(lwindow)
