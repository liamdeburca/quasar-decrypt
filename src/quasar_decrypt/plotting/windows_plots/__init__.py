import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "balmer": [
        "BalmerWindowsPlotWithDifferences",
        "BalmerWindowsPlotWithResiduals",
    ],
    "continuum": [
        "ContinuumWindowsPlotWithDifferences",
        "ContinuumWindowsPlotWithResiduals",
    ],
    "host": [
        "HostWindowsPlotWithDifferences",
        "HostWindowsPlotWithResiduals",
    ],
    "iron": [
        "IronWindowsPlotWithDifferences",
        "IronWindowsPlotWithResiduals",
    ],
}

if TYPE_CHECKING:
    from .balmer import (  # noqa: F401
        BalmerWindowsPlotWithDifferences,
        BalmerWindowsPlotWithResiduals,
    )
    from .continuum import (  # noqa: F401
        ContinuumWindowsPlotWithDifferences,
        ContinuumWindowsPlotWithResiduals,
    )
    from .host import (  # noqa: F401
        HostWindowsPlotWithDifferences,
        HostWindowsPlotWithResiduals,
    )
    from .iron import (  # noqa: F401
        IronWindowsPlotWithDifferences,
        IronWindowsPlotWithResiduals,
    )

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )
