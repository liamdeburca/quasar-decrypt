from pathlib import Path

from h5py import File
from numpy import dtype, float32
from quasar_models.io.json import file_to_model, model_to_file
from quasar_utils.setup import Info

from quasar_decrypt import Spectrum

from .add_models_utils import add_model_to_spectrum
from .hdf5.spectrum import hdf5_to_spectrum, spectrum_to_hdf5


def dump_spectrum(
    dir_path: str | Path,
    spectrum: Spectrum,
    ext: str = "toon",
    force: bool = False,
    _dtype: str | dtype = float32,
    _name: str = "spectrum",
) -> Path:
    """
    Dumps a directory containing the spectrum and its associated model (if any)
    to the specified directory.

    The resulting directory structure will be:

    .../spectrum/
                spectrum.h5 (see `spectrum_to_hdf5`)
                model.* (if any, see `model_to_file`)
    """
    dir_path = Path(dir_path)
    if dir_path.name != _name:
        dir_path = dir_path / _name

    dir_path.mkdir(parents=True, exist_ok=force)

    model = spectrum.getModel()
    if model is not None:
        ext = ext.removeprefix(".").lower()
        model_path = dir_path / f"model.{ext}"
        model_to_file(model_path, model, spectrum.info)

    hdf5_path = dir_path / "spectrum.h5"
    with File(hdf5_path, "w") as f:
        spectrum_to_hdf5(spectrum, f, _dtype=_dtype, force=force)

    return dir_path


def load_spectrum(
    dir_path: str | Path,
    info: Info,
    _name: str = "spectrum",
) -> Spectrum:
    """
    Loads a spectrum and its associated model (if any) from the specified 
    directory. The directory is expected to have the structure created by 
    `dump_spectrum`.
    """
    dir_path = Path(dir_path)
    if dir_path.name != _name:
        dir_path = dir_path / _name

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path!s}")

    try:
        model_path = next(dir_path.glob("model.*"))
        model = file_to_model(model_path, info)
    except StopIteration:
        model = None

    hdf5_path = dir_path / "spectrum.h5"
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Could not find HDF5 file: {hdf5_path!s}")

    with File(hdf5_path, "r") as f:
        spectrum = hdf5_to_spectrum(f, info)

    add_model_to_spectrum(spectrum, model)
    return spectrum