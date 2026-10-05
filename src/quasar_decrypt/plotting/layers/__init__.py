__all__ = [
    "BalmerTemplatesLayer",
    "ComponentsLayer",
    "DataLayer",
    "DataWithComponentsLayer",
    "DifferenceLayer",
    "HostTemplatesLayer",
    "IronTemplatesLayer",
    "ResidualLayer",
    "WindowsLayer",
]
from dataclasses import dataclass

from .components import ComponentsLayer
from .misc import (
    DataLayer,
    DifferenceLayer,
    ResidualLayer,
)
from .templates import (
    BalmerTemplatesLayer,
    HostTemplatesLayer,
    IronTemplatesLayer,
)
from .windows import WindowsLayer


@dataclass
class DataWithComponentsLayer(DataLayer, ComponentsLayer):
    """Dataclass responsible for plotting the data and the modelled components."""