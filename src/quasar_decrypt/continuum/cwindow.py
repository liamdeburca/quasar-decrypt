from dataclasses import dataclass, field
from logging import getLogger
from typing import ClassVar, Optional, Self

from numpy import dot
from quasar_models.continuum import PowerLawModel
from quasar_typing.astropy import FitInfo
from quasar_typing.misc import BackgroundFlux, Suffix
from quasar_typing.numpy import FloatVector
from quasar_utils.decorators import validate_call

from ..utils import _SpecWindow

logger = getLogger(__name__)


@dataclass(init=False)
class CWindow(_SpecWindow):
    fit_info: FitInfo | None = field(default=None, init=False)
    fit: Optional[PowerLawModel] = field(default=None, init=False)
    fit_raw: Optional[PowerLawModel] = field(default=None, init=False)
    fit_sc: Optional[PowerLawModel] = field(default=None, init=False)

    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({"all", "pl"})

    @validate_call
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
        bg_flux: BackgroundFlux | None = None,
    ) -> FloatVector:
        if bg_flux is None:
            bg_flux = self.default_bg

        if log_valid is None:
            log_valid = log

        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            mode="r",  # r: read only
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=valid,
            log_valid=log_valid,
            bg_flux=bg_flux,
        )
        return (masked_coords.y - fit(masked_coords.x)) / masked_coords.dy

    @validate_call
    def getSNR(
        self,
        fit: PowerLawModel,
        *,
        covered: bool = True,
        without_rejections: bool = False,
        without_absorption: bool = False,
        bg_flux: BackgroundFlux | None = None,
    ) -> float:
        if bg_flux is None:
            bg_flux = self.default_bg

        masked_coords = self.getMaskedCoords.__wrapped__(
            self,
            mode="r",  # r: read only
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=True,
            log_valid=False,
            bg_flux=bg_flux,
        )
        X = masked_coords.dx.sum()
        f = fit(masked_coords.x)

        snr = dot(f, masked_coords.dx) / X
        snr /= (
            dot((masked_coords.y - f) ** 2, masked_coords.dx) ** 2 / X
        ) ** 0.5

        return snr

    @validate_call
    def applyFit(
        self,
        fit: PowerLawModel,
        *,
        fit_info: FitInfo | None = None,
        suffix: Suffix | None = None,
        update_emission: bool = True,
    ) -> Self:
        self.fit_info = fit_info

        if suffix == "raw":
            self.fit_raw = fit
        elif suffix == "sc":
            self.fit_sc = fit
        else:
            self.fit = fit

        if update_emission:
            self.updateContinuumEmission.__wrapped__(self, fit)

        return self
