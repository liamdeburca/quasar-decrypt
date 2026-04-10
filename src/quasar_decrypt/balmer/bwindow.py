"""
The Balmer window class: 'BWindow'.
"""
from logging import getLogger
from typing import Self, ClassVar, Optional
from dataclasses import dataclass, field
from pydantic import validate_call

from quasar_models.balmer import BalmerModel
from quasar_typing.astropy import FitInfo
from quasar_typing.misc.literals import BGFlux

from ..utils.specdata import SpecData

logger = getLogger(__name__)
logger.disabled = not getLogger().hasHandlers()

@dataclass(init=False)
class BWindow(SpecData):
    model: Optional[BalmerModel] = field(default=None, init=False)
    fit: Optional[BalmerModel] = field(default=None, init=False)
    fit_info: FitInfo | None = field(default=None, init=False)

    default_bg: ClassVar[BGFlux] = {'pl', 'fe', 'hg', 'em'}
    
    @validate_call(validate_return=False)
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