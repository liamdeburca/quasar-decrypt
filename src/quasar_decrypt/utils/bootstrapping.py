from logging import getLogger
from typing import Any, Optional, Union

from quasar_errors.bootstrapping import (
    BaseBootstrapper,
    FlexibleBootstrapper,
    RigidBootstrapper,
    SpectrumBootstrapper,
    StandardBootstrapper,
)
from quasar_models import (
    BaModel,
    EmModel,
    FeModel,
    HgModel,
)
from quasar_typing.astropy import CompoundModelType
from quasar_typing.literals import (
    BootstrapType,
    Variant,
)
from quasar_typing.multiprocessing import PoolType
from quasar_typing.numpy import RandomStateType
from quasar_typing.string_selections import DataTypes, ModelTypes, VaryLines
from quasar_utils.continuum_fit_result import ContinuumFitResult
from quasar_utils.setup import FitterKwargs, Info

from .masked_coords import MaskedCoords

logger = getLogger(__name__)


def validate_model_and_data(
    *,
    model_types: ModelTypes | None,
    data_types: DataTypes | None,
    variant: Variant,
) -> None:
    """
    Raises a ValueError if the required model types or data types are missing.
    """
    if variant == "spectrum" or variant == "standard":
        # No requirements
        required_model_types = set()
    elif variant == "flexible":
        # Power law and emission lines are refit
        required_model_types = {"pl", "em"}
    elif variant == "rigid":
        # Emission lines are refit
        required_model_types = {"em"}
    else:
        raise ValueError(f"Unknown bootstrap variant: {variant}")

    for m_type in required_model_types:
        if m_type in model_types:
            continue
        raise ValueError(f"Missing required model type: {m_type}")

    for m_type in model_types:
        if m_type in data_types:
            continue
        raise ValueError(f"Missing required data type: {m_type}")


def args_and_kwargs_helper(
    *,
    masked_coords: MaskedCoords,
    info: Info,
    variant: Variant,

    cfit: ContinuumFitResult | None,
    pl: Optional[BaModel] = None,
    fe: Union[FeModel, CompoundModelType[FeModel], None] = None,
    ba: Optional[BaModel] = None,
    hg: Optional[HgModel] = None,
    em: Union[EmModel, CompoundModelType[EmModel], None] = None,

    fitter_kwargs: FitterKwargs,
    iterations: int = 100,
    random_state: RandomStateType = None,
    bootstrap_type: BootstrapType = "model",
    n_sigmas: float = 2.0,
    vary_lines: VaryLines | None = None,
    pool: PoolType | None = None,
    replace_missing: bool = True,
    tqdm_disable: bool = False,
    tqdm_leave: bool = False,
) -> tuple[BaseBootstrapper, tuple[Any, ...], dict[str, Any]]:
    args = (
        masked_coords._x.copy(),
        masked_coords._y.copy(),
        masked_coords._dy.copy(),
        masked_coords._dx.copy(),
    )
    for arr in args:
        arr.setflags(write=True)

    kwargs = {
        "kernels": masked_coords._kernels,
        "eval_mask": masked_coords.primary_mask,
        "comp_mask": masked_coords.secondary_mask,
        "is_logbinned": not masked_coords._original,
        "info": info,

        "cfit": cfit,
        "pl": pl,
        "fe": fe,
        "ba": ba,
        "hg": hg,
        "em": em,

        "pool": pool,

        "fitter_kwargs": info.nonlinear.fitter_kwargs | fitter_kwargs,
        "iterations": iterations,
        "random_state": random_state,
        "bootstrap_type": bootstrap_type,
        "n_sigmas": n_sigmas,
        "replace_missing": replace_missing,
        "vary_lines": vary_lines or {"all"},
        "tqdm_disable": tqdm_disable,
        "tqdm_leave": tqdm_leave,
    }
    bootstrapper_cls: type[BaseBootstrapper] = {
        "spectrum": SpectrumBootstrapper,
        "standard": StandardBootstrapper,
        "flexible": FlexibleBootstrapper,
        "rigid": RigidBootstrapper,
    }.get(variant, None)
    if bootstrapper_cls is None:
        raise ValueError(f"Unknown bootstrapping variant: {variant}")

    return bootstrapper_cls, args, kwargs