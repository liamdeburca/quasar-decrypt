from dataclasses import dataclass

from .components import ComponentsLayer
from .data import DataLayer


@dataclass
class DataWithComponentsLayer(DataLayer, ComponentsLayer):
    """Dataclass responsible for plotting the data and the modelled components."""
