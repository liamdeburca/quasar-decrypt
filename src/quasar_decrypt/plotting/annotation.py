__all__ = [
    "BalmerModelAnnotater",
    "HostGalaxyModelAnnotater",
    "IronModelAnnotater",
    "PowerLawModelAnnotater",
]

from collections.abc import Iterable
from dataclasses import dataclass, field

from astropy.units import Quantity, Unit
from matplotlib.text import Annotation
from numpy import interp, nanmax, nanmin

from .managers.model_manager import (
    BalmerModelManager,
    HostGalaxyModelManager,
    IronModelManager,
    PowerLawModelManager,
)


@dataclass
class _SimpleAnnotater:
    """
    Dataclass for managing a simple, textbox annotation on a plot.
    """
    annotation_title: str = field(default="", kw_only=True)
    annotation_bullet: str = field(default=r"$\circ$", kw_only=True)

    annotation_xy: tuple[float, float] = field(default=(0.95, 0.95), kw_only=True)
    annotation_xycoords: str = field(default="axes fraction", kw_only=True)
    annotation_ha: str = field(default="right", kw_only=True)
    annotation_va: str = field(default="top", kw_only=True)
    annotation_fontsize: float | str = field(default="small", kw_only=True)

    verbose: int = field(default=0, kw_only=True)

    _annotation_lines: list[str] = field(default_factory=list, init=False)
    _annotation: Annotation | None = field(default=None, init=False)

    @property
    def _new_bullet(self) -> str:
        return self.annotation_bullet + " "

    def _format_flux(self, flux: Quantity) -> str:
        return self.info.units.formatFlux(flux, with_unit=False)

    def _format_wavelength(self, wavelength: Quantity) -> str:
        return self.info.units.formatWavelength(wavelength, with_unit=False)

    def _format_fwhm(self, fwhm: Quantity) -> str:
        return self.info.units.formatC(fwhm, with_unit=False)

    def _format_kms(self, value: Quantity) -> str:
        return self.info.units.formatKMS(value, with_unit=False)

    def _format_unitless(self, value: float) -> str:
        return self.info.units.formatUnitless(value, with_unit=False)

    def _add_annotation(self) -> None:
        label = "\n".join(self._annotation_lines)
        if self.annotation_title:
            label = self.annotation_title + "\n" + label

        self._annotation = self[0].annotate(
            label,
            xy=self.annotation_xy,
            xycoords=self.annotation_xycoords,
            ha=self.annotation_ha,
            va=self.annotation_va,
            fontsize=self.annotation_fontsize,
        )


@dataclass
class _FluxChecksAnnotater(_SimpleAnnotater):
    flux_checks: Iterable | None = field(default=None, kw_only=True)

    _flux_checks: list[float] = field(default_factory=list, init=False)

    def _add_flux_checks_to_annotation(self, suffix: str) -> None:
        self._flux_checks.clear()
        if self.flux_checks is None:
            return

        lb, ub = getattr(self, 'xlim', (None, None))
        if lb is None:
            lb = nanmin(self._x)
        if ub is None:
            ub = nanmax(self._x)

        self._flux_checks.extend(
            wave
            for wave in self.info.units.getUnitlessWavelength(self.flux_checks)
            if (lb <= wave) & (wave <= ub)
        )
        xp = self._x[self._valid_pixels]
        yp = getattr(self, f"_y_{suffix}")[self._valid_pixels]
        if self._flux_checks:
            self._annotation_lines.append('\nFlux density @:')
            fluxs = interp(self._flux_checks, xp, yp)
            for wave, flux in zip(self._flux_checks, fluxs):
                _wave = self.info.units.formatWavelength(wave, with_unit=False)
                _flux = self.info.units.formatFlux(flux, with_unit=False)
                self._annotation_lines.append(
                    self._new_bullet + f"{_wave}: {_flux}"
                )


@dataclass
class PowerLawModelAnnotater(_FluxChecksAnnotater, PowerLawModelManager):
    """
    Dataclass for creating a textbox annotation for a PowerLawModel on a plot.
    """
    # Override the default annotation title
    annotation_title: str = field(default="Power Law Model", init=False)
    flux_checks: Iterable | None = field(
        default=[1450.0, 1700.0, 1900.0, 2800.0, 5050.0, 6000.0, 7000.0] * Unit('angstrom'), 
        kw_only=True,
    )
    verbose: int = field(default=1, kw_only=True)

    def format_annotation(self) -> None:
        pl_model = self.pl_model
        if pl_model is None or self.verbose == 0:
            return 

        flux = pl_model.flux
        alpha = pl_model.alpha

        flux_line = self._new_bullet + "flux: " + self._format_flux(flux.value)
        alpha_line = self._new_bullet + "alpha: " + self._format_unitless(alpha.value)

        if self.verbose >= 2:
            flux_line += " " + self._format_flux(flux.bounds)
            alpha_line += " " + self._format_unitless(alpha.bounds)

        self._annotation_lines.append(flux_line)
        self._annotation_lines.append(alpha_line)

        self._add_flux_checks_to_annotation('pl')
        self._add_annotation()


@dataclass
class IronModelAnnotater(_FluxChecksAnnotater, IronModelManager):
    """
    Dataclass for creating a textbox annotation for one or more IronModels on a plot.
    """
    # Override the default annotation title one
    annotation_title: str = field(default="", init=False)
    flux_checks: Iterable | None = field(
        default=[1, 2, 3, 4, 5, 6, 7] * Unit('1000 angstrom'), 
        kw_only=True,
    )
    verbose: int = field(default=1, kw_only=True)

    def format_annotation(self) -> None:
        fe_models = self.fe_model
        if fe_models is None or self.verbose == 0:
            return

        for fe_model in fe_models:
            flux = fe_model.flux
            fwhm = fe_model.fwhm

            name_line = fe_model.name + ":"
            flux_line = self._new_bullet + "flux: " + self._format_flux(flux.value)
            fwhm_line = self._new_bullet + "fwhm: " + self._format_kms(fwhm.value)

            if self.verbose >= 2:
                flux_line += " " + self._format_flux(flux.bounds)
                fwhm_line += " " + self._format_kms(fwhm.bounds)

            self._annotation_lines.append(name_line)
            self._annotation_lines.append(flux_line)
            self._annotation_lines.append(fwhm_line)

            if self.verbose >= 3:
                temp = fe_model.template
                x_norm_line = self._new_bullet + "x-norm: " + self._format_wavelength(temp.x_norm)
                fwhm_norm_line = self._new_bullet + "fwhm-norm: " + self._format_kms(temp.fwhm_norm)

                self._annotation_lines.append(x_norm_line)
                self._annotation_lines.append(fwhm_norm_line)

        self._add_flux_checks_to_annotation('fe')
        self._add_annotation()


@dataclass
class BalmerModelAnnotater(_FluxChecksAnnotater, BalmerModelManager):
    """
    Dataclass for creating a textbox annotation for a BalmerModel on a plot.
    """
    # Override the default annotation title one
    annotation_title: str = field(default="", init=False)
    flux_checks: Iterable | None = field(
        default=[3000.0, 3500.0, 4000.0] * Unit('angstrom'), 
        kw_only=True,
    )
    verbose: int = field(default=1, kw_only=True)

    def format_annotation(self) -> None:
        ba_model = self.ba_model
        if ba_model is None or self.verbose == 0:
            return

        flux = ba_model.flux
        fwhm = ba_model.fwhm
        ratio = ba_model.ratio

        name_line = f"{ba_model.continuum_template.name}/{ba_model.series_template.name}:"
        flux_line = self._new_bullet + "flux: " + self._format_flux(flux.value)
        fwhm_line = self._new_bullet + "fwhm: " + self._format_kms(fwhm.value)
        ratio_line = self._new_bullet + "ratio: " + self._format_unitless(ratio.value)

        if self.verbose >= 2:
            flux_line += " " + self._format_flux(flux.bounds)
            fwhm_line += " " + self._format_kms(fwhm.bounds)
            ratio_line += " " + self._format_unitless(ratio.bounds)

        self._annotation_lines.append(name_line)
        self._annotation_lines.append(flux_line)
        self._annotation_lines.append(fwhm_line)
        self._annotation_lines.append(ratio_line)

        if self.verbose >= 3:
            seri = ba_model.series_template
            cont = ba_model.continuum_template

            x_norm_cont_line = self._new_bullet + "x-norm (cont.): " + self._format_wavelength(cont.x_norm)
            fwhm_norm_cont_line = self._new_bullet + "fwhm-norm (cont.): " + self._format_kms(cont.fwhm_norm)
            x_norm_seri_line = self._new_bullet + "x-norm (seri.): " + self._format_wavelength(seri.x_norm)
            fwhm_norm_seri_line = self._new_bullet + "fwhm-norm (seri.): " + self._format_kms(seri.fwhm_norm)

            self._annotation_lines.append(x_norm_cont_line)
            self._annotation_lines.append(fwhm_norm_cont_line)
            self._annotation_lines.append(x_norm_seri_line)
            self._annotation_lines.append(fwhm_norm_seri_line)

        self._add_flux_checks_to_annotation('ba')
        self._add_annotation()


@dataclass
class HostGalaxyModelAnnotater(_FluxChecksAnnotater, HostGalaxyModelManager):
    """
    Dataclass for creating a textbox annotation for a HostGalaxyModel on a plot.
    """
    # Override the default annotation title one
    annotation_title: str = field(default="", init=False)
    flux_checks: Iterable | None = field(
        default=[4000.0, 5000.0, 6000.0, 7000.0] * Unit('angstrom'), 
        kw_only=True,
    )
    verbose: int = field(default=1, kw_only=True)

    def format_annotation(self) -> None:
        hg_model = self.hg_model
        if hg_model is None or self.verbose == 0:
            return

        flux = hg_model.flux
        fwhm = hg_model.fwhm

        name_line = hg_model.name + ":"
        flux_line = self._new_bullet + "flux: " + self._format_flux(flux.value)
        fwhm_line = self._new_bullet + "fwhm: " + self._format_kms(fwhm.value)

        if self.verbose >= 2:
            flux_line += " " + self._format_flux(flux.bounds)
            fwhm_line += " " + self._format_kms(fwhm.bounds)

        self._annotation_lines.append(name_line)
        self._annotation_lines.append(flux_line)
        self._annotation_lines.append(fwhm_line)

        if self.verbose >= 3:
            temp = hg_model.template
            x_norm_line = self._new_bullet + "x-norm: " + self._format_wavelength(temp.x_norm)
            fwhm_norm_line = self._new_bullet + "fwhm-norm: " + self._format_kms(temp.fwhm_norm)
            self._annotation_lines.append(x_norm_line)
            self._annotation_lines.append(fwhm_norm_line)

        self._add_flux_checks_to_annotation('hg')
        self._add_annotation()