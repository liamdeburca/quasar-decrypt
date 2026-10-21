"""
Utilities for adding models instances to Spectrum objects loaded from HDF5 files.
"""
from collections import defaultdict
from typing import Any, Optional, Union

from quasar_models import (
    BalmerModel,
    EmModel,
    HostGalaxyModel,
    IronModel,
    PowerLawModel,
)
from quasar_models.modeling import BaseModel
from quasar_typing.astropy import CompoundModelType

from quasar_decrypt import Spectrum


def add_balmer_to_spectrum(
    spectrum: Spectrum,
    model: Optional[BalmerModel] = None,
) -> None:
    """
    Add a BalmerModel instance to a Spectrum object.
    """
    if spectrum.ba is None:
        raise ValueError("BalmerWindows has not been initialized!")

    spectrum.ba.model = model
    spectrum.ba.applyFit(model, fit_info=None, update_emission=True)

def add_powerlaw_to_spectrum(
    spectrum: Spectrum,
    model: Optional[PowerLawModel] = None,
) -> None:
    """
    Add a PowerLawModel instance to a Spectrum object.
    """
    if spectrum.pl is None:
        raise ValueError("ContinuumWindows has not been initialized!")
    
    spectrum.pl.fit_raw = model
    spectrum.pl.fit_sc = model
    spectrum.pl.applyFit(model, fit_info=None, update_emission=True)

def add_host_to_spectrum(
    spectrum: Spectrum,
    model: Optional[HostGalaxyModel] = None,
) -> None:
    """
    Add a HostGalaxyModel instance to a Spectrum object.
    """
    if spectrum.hg is None:
        raise ValueError("HostWindows has not been initialized!")

    spectrum.hg.templates.clear()
    spectrum.hg.models.clear()
    spectrum.hg.model = None
    spectrum.hg.fit = None
    spectrum.hg.fit_info = None

    if model is not None:
        key = model.template.name
        spectrum.hg.templates[key] = model.template
        spectrum.hg.models[key] = model

    spectrum.hg.model = model
    spectrum.hg.applyFit(model, fit_info=None, update_emission=True)

def add_iron_to_spectrum(
    spectrum: Spectrum,
    model: Union[IronModel, CompoundModelType[IronModel], None] = None,
) -> None:
    """
    Add an IronModel instance to a Spectrum object.
    """
    if spectrum.fe is None:
        raise ValueError("IronWindows has not been initialized!")

    spectrum.fe.fit_info = None
    spectrum.fe.templates.clear()
    spectrum.fe.template_models.clear()
    if model is not None:
        ms = (model,) if model.n_submodels == 1 else tuple(model)
        for m in ms:
            key = m.template.name
            spectrum.fe.templates[key] = m.template
            spectrum.fe.template_models[key] = m

    spectrum.fe.applyFit(model, fit_info=None, update_emission=True)

def add_lines_to_spectrum(
    spectrum: Spectrum,
    model: Union[EmModel, CompoundModelType[EmModel], None] = None,
) -> None:
    """
    Add emission line model(s) to a Spectrum object.
    """
    if spectrum.em is None:
        raise ValueError("LineWindows has not been initialized!")

    for lwindow in spectrum.em:
        lwindow.model = None
        lwindow.fit = None
        lwindow.fit_info = None
        lwindow.fits.clear()
        lwindow.fit_infos.clear()
        lwindow._y_em_contrib[:] = 0.0

    if model is None:
        return

    name_to_idx_map: dict[str, int] = {
        name: idx
        for idx, lwindow in enumerate(spectrum.em)
        for name in lwindow.names
    }
    models_dict = defaultdict(list)
    ms = (model,) if model.n_submodels == 1 else tuple(model)
    for m in sorted(ms, key=lambda x: x.sorting_key):
        idx = name_to_idx_map[m.pure_name]
        models_dict[idx].append(m)

    compound_models_dict: dict[int, Any] = {
        idx: sum(val[1:], start=val[0]) if val else None
        for idx, val in models_dict.items()
    }
    for idx, compound_model in compound_models_dict.items():
        lwindow = spectrum.em[idx]
        lwindow.model = compound_model
        lwindow.fit = compound_model
        lwindow.updateLinesEmission(compound_model)

    # Force the line emission array to be the sum of individual contributions
    # ( Buggy otherwise -> fix should be identified later )
    spectrum._y_em[:] = sum(lwindow._y_em_contrib for lwindow in spectrum.em)

def add_model_to_spectrum(
    spectrum: Spectrum,
    model: Union[BaseModel, CompoundModelType[BaseModel], None] = None,
) -> None:
    """
    Adds a general (compound) model instance to a Spectrum object.
    """
    submodels_dict = {
        "pl": [],
        "ba": [],
        "fe": [],
        "hg": [],
        "em": [],
    }
    if model is not None:
        ms = (model,) if model.n_submodels == 1 else tuple(model)
        for m in sorted(ms, key=lambda x: x.sorting_key):
            submodels_dict[m.model_type].append(m)

    for key, value in submodels_dict.items():
        submodels_dict[key] = sum(value[1:], start=value[0]) if value else None

    if spectrum.pl is None:
        assert not submodels_dict["pl"]
    else:
        add_powerlaw_to_spectrum(spectrum, submodels_dict["pl"])

    if spectrum.ba is None:
        assert not submodels_dict["ba"]
    else:
        add_balmer_to_spectrum(spectrum, submodels_dict["ba"])

    if spectrum.fe is None:
        assert not submodels_dict["fe"]
    else:
        add_iron_to_spectrum(spectrum, submodels_dict["fe"])

    if spectrum.hg is None:
        assert not submodels_dict["hg"]
    else:
        add_host_to_spectrum(spectrum, submodels_dict["hg"])

    if spectrum.em is None:
        assert not submodels_dict["em"]
    else:
        add_lines_to_spectrum(spectrum, submodels_dict["em"])
