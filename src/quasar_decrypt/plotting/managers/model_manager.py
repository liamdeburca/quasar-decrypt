from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Union

from quasar_models import (
    BalmerModel,
    EmModel,
    HostGalaxyModel,
    IronModel,
    PowerLawModel,
)
from quasar_models.modeling import SequentialModel
from quasar_models.modeling.base_model import BaseModel
from quasar_typing.astropy import CompoundModelType


@dataclass
class ModelManager:
    """Dataclass for managing an Astropy model."""
    model: Union[BaseModel, CompoundModelType[BaseModel], SequentialModel]

    @cached_property
    def _submodels(self) -> list[BaseModel]:
        if isinstance(self.model, SequentialModel):
            return list(self.model.submodels)
        return [self.model] if self.model.n_submodels == 1 else list(self.model)

    @cached_property
    def pl_model(self) -> Optional[PowerLawModel]:
        return next(
            filter(lambda m: m.model_type == 'pl', self._submodels),
            None,
        )

    @cached_property
    def fe_model(self) -> Optional[list[IronModel]]:
        return [m for m in self._submodels if m.model_type == 'fe'] or None

    @cached_property
    def ba_model(self) -> Optional[BalmerModel]:
        return next(
            filter(lambda m: m.model_type == 'ba', self._submodels),
            None,
        )

    @cached_property
    def hg_model(self) -> Optional[HostGalaxyModel]:
        return next(
            filter(lambda m: m.model_type == 'hg', self._submodels),
            None,
        )

    @cached_property
    def em_models(self) -> Optional[list[EmModel]]:
        return [m for m in self._submodels if m.model_type == 'em'] or None


@dataclass
class PowerLawModelManager(ModelManager):
    """Dataclass for managing a PowerLawModel."""
    model: Union[PowerLawModel, SequentialModel]


@dataclass
class IronModelManager(ModelManager):
    """Dataclass for managing an IronModel."""
    model: Union[IronModel, CompoundModelType[IronModel], SequentialModel]


@dataclass
class BalmerModelManager(ModelManager):
    """Dataclass for managing a BalmerModel."""
    model: Union[BalmerModel, CompoundModelType[BalmerModel], SequentialModel]


@dataclass
class HostGalaxyModelManager(ModelManager):
    """Dataclass for managing a HostGalaxyModel."""
    model: Union[HostGalaxyModel, CompoundModelType[HostGalaxyModel], SequentialModel]


@dataclass
class EmissionLineModelManager(ModelManager):
    """Dataclass for managing an EmissionLineModel."""
    model: Union[EmModel, CompoundModelType[EmModel], SequentialModel]