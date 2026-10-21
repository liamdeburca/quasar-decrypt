import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "emission_line_components": [
        "EmissionLineComponentsWithDifferences",
        "EmissionLineComponentsWithResiduals",
        "SimpleEmissionLineComponents",
    ],
}

if TYPE_CHECKING:
    from .emission_line_components import (  # noqa: F401
        EmissionLineComponentsWithDifferences,
        EmissionLineComponentsWithResiduals,
        SimpleEmissionLineComponents,
    )

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )
