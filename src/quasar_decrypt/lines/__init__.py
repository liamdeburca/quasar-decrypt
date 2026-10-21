import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "linewindows": ["LineWindows"],
    "lwindow": ["LWindow"],
}

if TYPE_CHECKING:
    from .linewindows import LineWindows  # noqa: F401
    from .lwindow import LWindow  # noqa: F401

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )
