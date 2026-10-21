import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "contwindows": ["ContinuumWindows"],
    "cwindow": ["CWindow"],
}

if TYPE_CHECKING:
    from .contwindows import ContinuumWindows  # noqa: F401
    from .cwindow import CWindow  # noqa: F401

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )
