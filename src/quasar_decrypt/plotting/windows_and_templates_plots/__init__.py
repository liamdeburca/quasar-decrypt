import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "balmer": [
        "BalmerTemplatesWithDifferencesPlot",
        "BalmerTemplatesWithResidualsPlot",
        "BalmerWindowsAndTemplatesWithDifferencesPlot",
        "BalmerWindowsAndTemplatesWithResidualsPlot",
    ],
    "host": [
        "HostTemplatesWithDifferencesPlot",
        "HostTemplatesWithResidualsPlot",
        "HostWindowsAndTemplatesWithDifferencesPlot",
        "HostWindowsAndTemplatesWithResidualsPlot",
    ],
    "iron": [
        "IronTemplatesWithDifferencesPlot",
        "IronTemplatesWithResidualsPlot",
        "IronWindowsAndTemplatesWithDifferencesPlot",
        "IronWindowsAndTemplatesWithResidualsPlot",
    ],
    "spectrum": [
        "AllTemplatesWithDifferencesPlot",
        "AllTemplatesWithResidualsPlot",
    ],
}

if TYPE_CHECKING:
    from .balmer import (  # noqa: F401
        BalmerTemplatesWithDifferencesPlot,
        BalmerTemplatesWithResidualsPlot,
        BalmerWindowsAndTemplatesWithDifferencesPlot,
        BalmerWindowsAndTemplatesWithResidualsPlot,
    )
    from .host import (  # noqa: F401
        HostTemplatesWithDifferencesPlot,
        HostTemplatesWithResidualsPlot,
        HostWindowsAndTemplatesWithDifferencesPlot,
        HostWindowsAndTemplatesWithResidualsPlot,
    )
    from .iron import (  # noqa: F401
        IronTemplatesWithDifferencesPlot,
        IronTemplatesWithResidualsPlot,
        IronWindowsAndTemplatesWithDifferencesPlot,
        IronWindowsAndTemplatesWithResidualsPlot,
    )
    from .spectrum import (  # noqa: F401
        AllTemplatesWithDifferencesPlot,
        AllTemplatesWithResidualsPlot,
    )

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )
