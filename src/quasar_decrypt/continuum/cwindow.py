from logging import getLogger
from typing import Self, ClassVar, Optional
from dataclasses import dataclass, field
from numpy import dot

from ..utils.specdata import SpecData

from pydantic import validate_call
from quasar_typing.numpy import FloatVector
from quasar_typing.astropy import FitInfo
from quasar_typing.misc.literals import BGFlux, Suffix

from quasar_models.continuum import PowerLawModel

logger = getLogger(__name__)
logger.disabled = not getLogger().hasHandlers()

@dataclass(init=False)
class CWindow(SpecData):
    fit_info: FitInfo | None = field(default=None, init=False)
    fit: Optional[PowerLawModel] = field(default=None, init=False)
    fit_raw: Optional[PowerLawModel] = field(default=None, init=False)
    fit_sc: Optional[PowerLawModel] = field(default=None, init=False)

    default_bg: ClassVar[BGFlux] = {'fe', 'ba', 'hg', 'em'}

    @validate_call(validate_return=False)
    def getResiduals(
        self,
        fit: PowerLawModel,
        log: bool = False,
        *,
        covered: bool = True,
        without_rejections: bool = False, 
        without_absorption: bool = False,
        valid: bool = True,
        log_valid: bool | None = None,
        bg_flux: BGFlux | None = None,
    ) -> FloatVector:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        if log_valid is None:
            log_valid = log

        x, y, dy = self.getMaskedCoords.__wrapped__(
            self,
            covered = covered,
            log = log,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
            valid = valid,
            log_valid = log_valid,
            bg_flux = bg_flux,
        )
        return (y - fit(x)) / dy

    @validate_call(validate_return=False)
    def getSNR(
        self,
        fit: PowerLawModel,
        *,
        covered: bool = True,
        without_rejections: bool = False, 
        without_absorption: bool = False,
        bg_flux: BGFlux | None = None,
    ) -> float:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if bg_flux is None:
            bg_flux = self.default_bg

        x, y, _ = self.getMaskedCoords.__wrapped__(
            self,
            covered = covered,
            log = False,
            without_rejections = without_rejections,
            without_absorption = without_absorption,
            valid = True,
            log_valid = False,
            bg_flux = bg_flux,
        )
        dx = x * self.info.loading['sigma_res']
        X = dx.sum()

        f = _f[0] if isinstance(_f := fit(x), tuple) else _f

        snr = dot(f, dx) / X
        snr /= (dot((y - f)**2, dx)**2 / X)**0.5

        return snr
    
    @validate_call(validate_return=False)
    def applyFit(
        self,
        fit: PowerLawModel,
        fit_info: FitInfo,
        suffix: Suffix | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self.fit_info = fit_info
        if suffix == 'raw':
            self.fit_raw = fit
        elif suffix == 'sc':
            self.fit_sc = fit
        else:
            self.fit = fit
        return self