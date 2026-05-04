"""
The Balmer window class: 'BWindow'.
"""
from logging import getLogger
from typing import Self, ClassVar
from dataclasses import dataclass, field
from pydantic import validate_call

from quasar_models.balmer import BalmerModel
from quasar_typing.astropy import FitInfo
from quasar_typing.misc import BackgroundFlux

from ..utils.specdata import SpecData

logger = getLogger(__name__)

@dataclass(init=False)
class BWindow(SpecData):
    model: BalmerModel | None = field(default=None, init=False)
    fit: BalmerModel | None = field(default=None, init=False)
    fit_info: FitInfo | None = field(default=None, init=False)

    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({'all', 'ba'})
    
    @validate_call
    def applyFit(
        self,
        fit: BalmerModel,
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
        fit: BalmerModel,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self.fit = fit
        self.fit_info = None
        return self