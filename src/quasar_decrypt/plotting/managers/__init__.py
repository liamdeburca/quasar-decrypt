import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "composite_managers": [
        "DoublePlotManager",
        "GeneralManager",
        "SinglePlotManager",
        "StandardWithDifferencePlotManager",
        "StandardWithDifferenceAndTemplateManager",
        "StandardWithResidualsAndTemplateManager",
        "StandardWithResidualsPlotManager",
        "TriplePlotManager",
    ],
    "labels": [
        "DoubleLabelsManager",
        "DoubleLabelsManagerDifference",
        "DoubleLabelsManagerResidual",
        "SingleLabelsManager",
        "TripleLabelsManager",
        "TripleLabelsManagerDifferenceAndTemplate",
        "TripleLabelsManagerResidualAndTemplate",
    ],
    "limits": [
        "DoubleLimitsManager",
        "SingleLimitsManager",
        "TripleLimitsManager",
    ],
    "misc": [
        "ColorsManager",
        "FigureManager",
        "LegendsManager",
        "LinestylesManager",
        "SubtitleManager",
        "TitleManager",
    ],
    "scales": [
        "DoubleScalesManager",
        "SingleScalesManager",
        "TripleScalesManager",
    ],
    "spectrum_manager": ["SpectrumManager"],
}

if TYPE_CHECKING:
    from .composite_managers import (  # noqa: F401
        DoublePlotManager,
        GeneralManager,
        SinglePlotManager,
        StandardWithDifferencePlotManager,
        StandardWithDifferenceAndTemplateManager,
        StandardWithResidualsAndTemplateManager,
        StandardWithResidualsPlotManager,
        TriplePlotManager,
    )
    from .labels import (  # noqa: F401
        DoubleLabelsManager,
        DoubleLabelsManagerDifference,
        DoubleLabelsManagerResidual,
        SingleLabelsManager,
        TripleLabelsManager,
        TripleLabelsManagerDifferenceAndTemplate,
        TripleLabelsManagerResidualAndTemplate,
    )
    from .limits import (  # noqa: F401
        DoubleLimitsManager,
        SingleLimitsManager,
        TripleLimitsManager,
    )
    from .misc import (  # noqa: F401
        ColorsManager,
        FigureManager,
        LegendsManager,
        LinestylesManager,
        SubtitleManager,
        TitleManager,
    )
    from .scales import (  # noqa: F401
        DoubleScalesManager,
        SingleScalesManager,
        TripleScalesManager,
    )
    from .spectrum_manager import SpectrumManager  # noqa: F401

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )
