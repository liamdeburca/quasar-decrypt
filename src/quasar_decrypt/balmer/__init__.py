import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "balmerwindows": ["BalmerWindows"],
    "bwindow": ["BWindow"],
}

if TYPE_CHECKING:
    from .balmerwindows import BalmerWindows  # noqa: F401
    from .bwindow import BWindow  # noqa: F401

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )
