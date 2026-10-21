import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "ironwindows": ["IronWindows"],
    "iwindow": ["IWindow"],
}

if TYPE_CHECKING:
    from .ironwindows import IronWindows  # noqa: F401
    from .iwindow import IWindow  # noqa: F401

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )
