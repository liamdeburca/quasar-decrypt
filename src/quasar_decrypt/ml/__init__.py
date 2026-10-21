import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "over": ["Over"],
    "under": ["Under"],
}

if TYPE_CHECKING:
    from .over import Over  # noqa: F401
    from .under import Under  # noqa: F401

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )
