from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

from astropy.constants import c
from numpy import array, interp
from quasar_errors.measures.exact import (
    get_left_edge_at_height,
    get_peak,
    get_right_edge_at_height,
)
from quasar_models import GaussianModel
from quasar_typing.literals import FWHMStrategy
from quasar_typing.numpy import BoolVector, FloatMatrix, FloatVector
from quasar_utils.setup import Info

C_KMS: float = c.to("km/s").value


@dataclass(kw_only=True)
class GaussianField:
    model: GaussianModel
    x: FloatVector
    valid_pixels: FloatVector
    info: Info
    y_pl: FloatVector

    _color: str | Callable[[str], str]
    _linestyle: str | Callable[[str], str]

    bullet: str

    @property
    def mu(self) -> float:
        return self.model.mu

    @property
    def peak(self) -> float:
        return self.model.peak

    @property
    def y(self) -> FloatVector:
        return self.model(self.x)

    @property
    def y_base(self) -> float:
        return interp(self.mu, self.x[self.valid_pixels], self.y_pl[self.valid_pixels])

    @property
    def strength(self) -> float:
        return self.model.strength.value

    @property
    def strength_str(self) -> str:
        return self.info.units.formatStrength(self.strength, with_unit=False)

    @property
    def fwhm(self) -> float:
        return self.model.fwhm

    @property
    def fwhm_str(self) -> str:
        return self.info.units.formatWavelength(self.fwhm, with_unit=False)

    @property
    def fwhm_v(self) -> float:
        return self.fwhm / self.mu * C_KMS

    @property
    def fwhm_v_str(self) -> str:
        return self.info.units.formatKMS(self.fwhm_v, with_unit=False)

    @property
    def offset(self) -> float:
        return self.model.v_off.value

    @property
    def offset_str(self) -> str:
        return self.info.units.formatKMS(self.offset, with_unit=False)

    @property
    def color(self) -> str:
        return self._color \
            if isinstance(self._color, str) \
            else self._color(self.model.name)

    @property
    def linestyle(self) -> str:
        return self._linestyle \
            if isinstance(self._linestyle, str) \
            else self._linestyle(self.model.name)

    @property
    def label(self) -> str:
        return self.model.name.replace('#', r"$\#$")

    @property
    def pretty_label(self) -> str:
        return '\n'.join([
            self.label,
            f"{self.bullet} Int.Flux: {self.strength_str}",
            f"{self.bullet} FWHM: {self.fwhm_str}",
            f"{self.bullet} Vel.FWHM: {self.fwhm_v_str}",
            f"{self.bullet} Vel.Off.: {self.offset_str}"
        ])

    ### Plotting

    @property
    def point(self) -> tuple[float, float]:
        return (self.mu, self.y_base + self.peak / 2)

    @property
    def xerr(self) -> float:
        return self.fwhm / 2

    @property
    def yerr(self) -> float:
        return self.peak / 2


@dataclass(kw_only=True)
class CombinedGaussianField:
    models: list[GaussianModel]
    x: FloatVector
    valid_pixels: BoolVector
    info: Info
    y_pl: FloatVector
    wave: float
    fwhm_strategy: FWHMStrategy

    color: str
    linestyle: str

    bullet: str

    def __call__(self, x: FloatVector) -> FloatVector:
        return sum(m(x) for m in self.models)

    @cached_property
    def mu_and_peak(self) -> tuple[float, float]:
        return get_peak(self, self.x[self.valid_pixels])

    @property
    def mu(self) -> float:
        return self.mu_and_peak[0]

    @property
    def peak(self) -> float:
        return self.mu_and_peak[1]

    @property
    def y(self) -> FloatVector:
        return self(self.x)

    @property
    def y_base(self) -> float:
        return interp(self.mu, self.x[self.valid_pixels], self.y_pl[self.valid_pixels])

    @property
    def strength(self) -> float:
        return sum(m.strength.value for m in self.models)

    @property
    def strength_str(self) -> str:
        return self.info.units.formatStrength(self.strength, with_unit=False)

    @cached_property
    def fwhm_left(self) -> float:
        return get_left_edge_at_height(
            self, 
            self.x[self.valid_pixels], 
            self.mu, 
            self.peak / 2,
            strategy=self.fwhm_strategy,
        )

    @cached_property
    def fwhm_right(self) -> float:
        return get_right_edge_at_height(
            self, 
            self.x[self.valid_pixels], 
            self.mu, 
            self.peak / 2,
            strategy=self.fwhm_strategy,
        )
    
    @property
    def fwhm(self) -> float:
        return self.fwhm_right - self.fwhm_left

    @property
    def fwhm_v(self) -> float:
        return self.fwhm / self.mu * C_KMS

    @property
    def fwhm_v_str(self) -> str:
        return self.info.units.formatKMS(self.fwhm_v, with_unit=False)
    
    @property
    def fwhm_str(self) -> str:
        return self.info.units.formatWavelength(self.fwhm, with_unit=False)

    @property
    def offset(self) -> float:
        return (self.mu / self.wave - 1) * C_KMS

    @property
    def offset_str(self) -> str:
        return self.info.units.formatKMS(self.offset, with_unit=False)

    @property
    def label(self) -> str:
        return "Total"

    @property
    def pretty_label(self) -> str:
        return '\n'.join([
            self.label,
            f"{self.bullet} Int.Flux: {self.strength_str}",
            f"{self.bullet} FWHM: {self.fwhm_str}",
            f"{self.bullet} Vel.FWHM: {self.fwhm_v_str}",
            f"{self.bullet} Vel.Off.: {self.offset_str}"
        ])

    ### Plotting

    @property
    def point(self) -> tuple[float, float]:
        return (self.mu, self.y_base + self.peak / 2)

    @property
    def xerr(self) -> FloatMatrix: # (2, 1)
        return array([
            self.mu - self.fwhm_left,
            self.fwhm_right - self.mu,
        ])[:,None]

    @property
    def yerr(self) -> float:
        return self.peak / 2