import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "_basespec": ["_BaseSpec"],
    "_spec": ["_Spec"],
    "_specwindow": ["_SpecWindow"],
    "_specwindowlist": ["_SpecWindowList"],
    "masked_coords": [
        "ContiguousMaskedCoords",
        "MaskedCoords",
        "OriginalContiguousMaskedCoords",
        "OriginalMaskedCoords",
        "OriginalReadOnlyMaskedCoords",
        "ReadOnlyMaskedCoords",
    ],
    "utils": [
        "CachedGetMask", 
        "get_log", 
        "get_mask", 
        "stopwatch",
    ],
}

if TYPE_CHECKING:
    from ._basespec import _BaseSpec  # noqa: F401
    from ._spec import _Spec  # noqa: F401
    from ._specwindow import _SpecWindow  # noqa: F401
    from ._specwindowlist import _SpecWindowList  # noqa: F401
    from .masked_coords import (  # noqa: F401
        ContiguousMaskedCoords,
        MaskedCoords,
        OriginalContiguousMaskedCoords,
        OriginalMaskedCoords,
        OriginalReadOnlyMaskedCoords,
        ReadOnlyMaskedCoords,
    )
    from .utils import (  # noqa: F401
        CachedGetMask,
        get_log,
        get_mask,
        stopwatch,
    )

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )

