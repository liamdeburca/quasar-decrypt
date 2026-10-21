import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "hostwindows": ["HostWindows"],
    "hwindow": ["HWindow"],
}

if TYPE_CHECKING:
    from .hostwindows import HostWindows  # noqa: F401
    from .hwindow import HWindow  # noqa: F401

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )
