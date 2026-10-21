import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "components": ["ComponentsLayer"],
    "composite": ["DataWithComponentsLayer"],
    "data": ["DataLayer"],
    "difference": ["DifferenceLayer"],
    "residual": ["ResidualLayer"],
    "templates": [
        "BalmerTemplatesLayer",
        "HostTemplatesLayer",
        "IronTemplatesLayer",
    ],
    "windows": ["WindowsLayer"],
}

if TYPE_CHECKING:
    from .components import ComponentsLayer  # noqa: F401
    from .composite import DataWithComponentsLayer  # noqa: F401
    from .data import DataLayer  # noqa: F401
    from .difference import DifferenceLayer  # noqa: F401
    from .residual import ResidualLayer  # noqa: F401
    from .templates import (  # noqa: F401
        BalmerTemplatesLayer,
        HostTemplatesLayer,
        IronTemplatesLayer,
    )
    from .windows import WindowsLayer  # noqa: F401

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )