"""
The Host window class: 'HWindow'.
"""
from logging import getLogger
from typing import Self, ClassVar
from dataclasses import dataclass, field

from quasar_models.host import HostGalaxyModel
from quasar_typing.astropy import FitInfo
from quasar_typing.misc import BackgroundFlux
from quasar_utils.decorators import validate_call

from ..utils.specdata import SpecData

logger = getLogger(__name__)

@dataclass(init=False)
class HWindow(SpecData):
    model: HostGalaxyModel | None = field(default=None, init=False)
    fit: HostGalaxyModel | None = field(default=None, init=False)
    fit_info: FitInfo | None = field(default=None, init=False)

    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({'all', 'hg'})
    
    @validate_call
    def applyFit(
        self,
        fit: HostGalaxyModel,
        fit_info: FitInfo,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self.fit = fit
        self.fit_info = fit_info
        return self
    
    @validate_call
    def adoptFit(
        self,
        fit: HostGalaxyModel,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self.fit = fit
        self.fit_info = None
        return self