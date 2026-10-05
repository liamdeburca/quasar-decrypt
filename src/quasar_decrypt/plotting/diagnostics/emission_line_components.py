from collections.abc import Callable
from dataclasses import dataclass, field
from re import Pattern, fullmatch
from typing import Self

from astropy.constants import c
from astropy.units.quantity import Quantity
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from quasar_models.line import GaussianModel, _VProfileCopy
from quasar_typing.misc import FWHMStrategy

from ..annotation import _SimpleAnnotater
from ..layers import (
    DataWithComponentsLayer,
    DifferenceLayer,
    ResidualLayer,
)
from ..managers import (
    SinglePlotManager,
    StandardWithDifferencePlotManager,
    StandardWithResidualsPlotManager,
)
from ..managers.misc import (
    SubtitleDisabler,
    TitleDisabler,
)
from ..managers.model_manager import ModelManager
from ..rcparams_utils import get_linewidth, get_xtick_minor_width, get_ymargin
from .utils import (
    CombinedGaussianField,
    GaussianField,
)


@dataclass
class _Base(
    ModelManager,
    _SimpleAnnotater,
    DataWithComponentsLayer,
    TitleDisabler,
    SubtitleDisabler,
):
    pattern: str | Pattern[str]
    dv: float | Quantity
    
    color_selected: str | Callable[[str], str] = field(default="k", kw_only=True)
    color_other: str = field(default="grey", kw_only=True)
    color_combined: str = field(default='k', kw_only=True)

    linestyle_selected: str | Callable[[str], str] = field(default='solid', kw_only=True)
    linestyle_other: str = field(default='dotted', kw_only=True)
    linestyle_combined: str = field(default='solid', kw_only=True)

    linewidth_selected: float = field(default_factory=get_linewidth, kw_only=True)
    linewidth_combined: float = field(default_factory=get_linewidth, kw_only=True)
    linewidth_otherwise: float = field(default_factory=get_xtick_minor_width, kw_only=True)

    fwhm_strategy: FWHMStrategy = field(default="average", kw_only=True)
    verbose: int = field(default=0, kw_only=True)

    # Override 'LimitsManager' fields
    xlim: tuple[float | None, float | None] = field(default=(None, None), init=False)
    # Override '_SimpleAnnotater' fields
    annotation_xy: tuple[float, float] = field(default=(0.05, 0.95), kw_only=True)
    annotation_ha: str = field(default="left", kw_only=True)

    # Shared private attributes
    _dv: float = field(init=False)
    _selected_gaussians: list[GaussianModel] = field(default_factory=list, init=False)
    _other_gaussians: list[GaussianModel] = field(default_factory=list, init=False)
    _irrelevant_gaussians: list[GaussianModel] = field(default_factory=list, init=False)

    _selected_waves: set[float] | None = field(default=None, init=False)
    _selected_mean_wave: float | None = field(default=None, init=False)

    _selected_data: dict[str, GaussianField] = field(default_factory=dict, init=False)
    _other_data: dict[str, GaussianField] = field(default_factory=dict, init=False)

    # Matplotlib object
    _other_lines: dict[str, Line2D] = field(default_factory=dict, init=False)

    _selected_lines: dict[str, Line2D] = field(default_factory=dict, init=False)
    _selected_crosses: dict[str, ErrorbarContainer] = field(default_factory=dict, init=False)

    _combined_line: Line2D | None = field(default=None, init=False)
    _combined_cross: ErrorbarContainer | None = field(default=None, init=False)

    _centre_lines: dict[str, Line2D] = field(default_factory=dict, init=False)
    _edges_lines: dict[str, tuple[Line2D, Line2D]] = field(default_factory=dict, init=False)
    _peak_lines: dict[str, Line2D] = field(default_factory=dict, init=False)

    def _filter_gaussian(self, model: GaussianModel) -> bool:
        if fullmatch(self.pattern, model.name):
            ls = self._selected_gaussians
        else:
            ls = self._other_gaussians
        ls.append(model)

    def _x_to_v(self, x: float) -> float:
        _c = c.to(self.info.units.velocity_unit).value
        return _c * (x / self._selected_mean_wave - 1)

    def _v_to_x(self, v: float) -> float:
        _c = c.to(self.info.units.velocity_unit).value
        return self._selected_mean_wave * (1 + v / _c)

    def _format_secondary_xaxis(self) -> None:
        self[0].xaxis.set_tick_params(top=False, which='both')
        secax = self[0].secondary_xaxis(
            'top', 
            functions=(self._x_to_v, self._v_to_x),
        )
        v_unit_str = self.info.units.velocity_unit.to_string('latex_inline')
        wave_str = self.info.units.formatWavelength(self._selected_mean_wave, with_unit=False)
        secax.set_xlabel(rf"Velocity offset ({v_unit_str}) from $\lambda${wave_str}")

    def _select_submodels(self) -> None:
        em_models = self.em_models
        if em_models is None:
            return

        self._selected_gaussians.clear()
        self._other_gaussians.clear()
        self._irrelevant_gaussians.clear()

        for em_model in em_models:
            if isinstance(em_model, _VProfileCopy):
                strength_scale: float = em_model.strength_scale
                n = em_model.n_profiles

                if n == 1:
                    strength = em_model.strength_1.value
                    fwhm_v = em_model.fwhm_v_1.value
                    v_off = em_model.v_off_1.value
                    name = em_model.name

                    m = GaussianModel.create(
                        em_model.wave, 
                        em_model.sigma_res, 
                        em_model.linetype,
                        strength=strength_scale * strength,
                        fwhm_v=fwhm_v,
                        v_off=v_off,
                        n_sigmas=em_model.n_sigmas,
                        name=name,
                    )
                    self._filter_gaussian(m)
                else:
                    for i in range(n):
                        strength = getattr(em_model, f"strength_{i+1}").value
                        fwhm_v = getattr(em_model, f"fwhm_v_{i+1}").value
                        v_off = getattr(em_model, f"v_off_{i+1}").value
                        name = f"{em_model.pure_name}#{i+1}"

                        m = GaussianModel.create(
                            em_model.wave, 
                            em_model.sigma_res, 
                            em_model.linetype,
                            strength=strength_scale * strength,
                            fwhm_v=fwhm_v, 
                            v_off=v_off,
                            n_sigmas=em_model.n_sigmas,
                            name=name,
                        )
                        self._filter_gaussian(m)
            else:
                self._filter_gaussian(em_model)

        if not self._selected_gaussians:
            raise ValueError(f"No Gaussian models matched the pattern '{self.pattern}'")

        self._selected_waves = {m.wave for m in self._selected_gaussians}
        self._selected_mean_wave = sum(self._selected_waves) / len(self._selected_waves)

        self._dv = self.info.units.getUnitlessVelocity(self.dv)
        lb = self._v_to_x(-self._dv)
        ub = self._v_to_x(self._dv)
        self.xlim = (lb, ub)
        self.mask = (lb <= self._x) & (self._x <= ub)

        self._irrelevant_gaussians.extend(filter(
            lambda m: m.wave < lb or ub < m.wave,
            self._other_gaussians,
        ))
        for m in self._irrelevant_gaussians:
            self._other_gaussians.remove(m)

    def _prepare_data(self) -> None:
        self._select_submodels()

        if isinstance(self.color_selected, str):
            color_selected = lambda _: self.color_selected
        else:
            color_selected = self.color_selected

        if isinstance(self.linestyle_selected, str):
            linestyle_selected = lambda _: self.linestyle_selected
        else:
            linestyle_selected = self.linestyle_selected

        self._selected_data.clear()
        for m in self._selected_gaussians:
            field = GaussianField(
                model=m,
                x=self.x,
                valid_pixels=self.valid_pixels,
                info=self.info,
                y_pl=self.y_pl,
                _color=color_selected,
                _linestyle=linestyle_selected,
                bullet=self.annotation_bullet,
            )
            self._selected_data[m.name] = field

        self._other_data.clear()
        for m in self._other_gaussians:
            m = GaussianField(
                model=m,
                x=self.x,
                valid_pixels=self.valid_pixels,
                info=self.info,
                y_pl=self.y_pl,
                _color=self.color_other,
                _linestyle=self.linestyle_other,
                bullet=self.annotation_bullet,
            )
            self._other_data[m.model.name] = m

    def _prepare_combined_data(self) -> None:

        field = None
        if len(self._selected_gaussians) == 1:
            field = next(iter(self._selected_data.values()))
        elif len(self._selected_gaussians) > 1:
            self._selected_data['total'] = field = CombinedGaussianField(
                models=self._selected_gaussians,
                x=self.x,
                valid_pixels=self.valid_pixels,
                info=self.info,
                y_pl=self.y_pl,
                wave=self._selected_mean_wave,
                fwhm_strategy=self.fwhm_strategy,
                color=self.color_selected,
                linestyle=self.linestyle_selected,
                bullet=self.annotation_bullet,
            )

        attr_name = "ylim" if len(self) == 1 else "y0lim"
        if (field is not None) and getattr(self, attr_name) == (None, None):
            ymin = field.y_base
            ymax = ymin + field.peak
            dy = field.peak * get_ymargin()
            setattr(self, attr_name, (ymin - dy, ymax + dy))
               

    def _plot_other(self) -> None:
        self._other_lines.clear()

        for name, data in filter(
            lambda item: item[1].color,
            self._other_data.items(),
        ):
            self._other_lines[name] = self[0].plot(
                data.x, self.y_pl + data.y,
                color=data.color,
                linestyle=data.linestyle,
                linewidth=self.linewidth_otherwise
            )[0]

    def _plot_selected(self) -> None:
        self._selected_lines.clear()

        for name, data in filter(
            lambda item: item[1].color,
            self._selected_data.items(),
        ):
            self._selected_lines[name] = self[0].plot(
                data.x, self.y_pl + data.y,
                color=data.color,
                linestyle=data.linestyle,
                linewidth=self.linewidth_selected
            )[0]

    def _plot_selected_crosses(self) -> None:
        self._selected_crosses.clear()
        if self.verbose <= 0:
            return

        for name, data in filter(
            lambda item: item[1].color,
            self._selected_data.items(),
        ):
            self._selected_crosses[name] = self[0].errorbar(
                *data.point, 
                xerr=data.xerr, 
                yerr=data.yerr,
                color=data.color,
                fmt='',
                elinewidth=self.linewidth_selected,
            )

    def _plot_combined(self) -> None:
        self._combined_line = None

        if 'total' not in self._selected_data:
            return
        
        data = self._selected_data['total']
        if not data.color:
            return

        self._combined_line = self[0].plot(
            self.x, self.y_pl + data.y,
            color=data.color,
            linestyle=data.linestyle,
            linewidth=self.linewidth_combined,
        )[0]

    def _plot_combined_cross(self) -> None:
        self._combined_cross = None
        if self.verbose <= 1 or 'total' not in self._selected_data:
            return

        data = self._selected_data['total']
        if not data.color:
            return

        self._combined_cross = self[0].errorbar(
            *data.point, 
            xerr=data.xerr, 
            yerr=data.yerr,
            color=data.color,
            fmt='',
            elinewidth=self.linewidth_otherwise,
        )

    def _plot_centres(self) -> None:
        self._centre_lines.clear()

        if self.verbose <= 0:
            return
        
        for name, data in filter(
            lambda item: item[1].color,
            self._selected_data.items(),
        ):
            self._centre_lines[name] = self[0].axvline(
                data.mu, 
                color=data.color,
                linestyle='solid',
                linewidth=self.linewidth_otherwise,
            )

    def _plot_edges(self) -> None:
        self._edges_lines.clear()

        if self.verbose <= 1:
            return
        
        for name, data in filter(
            lambda item: item[1].color,
            self._selected_data.items(),
        ):
            linestyle = 'solid' \
                if (name == 'total') or ('total' not in self._selected_data) \
                else 'dotted'

            self._edges_lines[name] = (
                self[0].axvline(
                    data.mu - data.fwhm / 2, 
                    color=data.color,
                    linestyle=linestyle,
                    linewidth=self.linewidth_otherwise,
                ),
                self[0].axvline(
                    data.mu + data.fwhm / 2, 
                    color=data.color,
                    linestyle=linestyle,
                    linewidth=self.linewidth_otherwise,
                )
            )

    def _plot_peak_lines(self) -> None:
        self._peak_lines.clear()

        if self.verbose <= 0:
            return
        
        y_lb, y_ub = self.ylim if len(self) == 1 else self.y0lim

        _y_lb, _y_ub = self[0].get_ylim()
        if y_lb is None:
            y_lb = _y_lb
        if y_ub is None:
            y_ub = _y_ub

        if len(self) == 1:
            self.ylim = (y_lb, y_ub)
        else:
            self.y0lim = (y_lb, y_ub)

        for name, data in filter(
            lambda item: item[1].color,
            self._other_data.items(),
        ):
            self._peak_lines[name] = self[0].plot(
                2 * [data.mu], 
                [y_lb, data.y_base + data.peak],
                color=data.color,
                linestyle='dashed',
                linewidth=self.linewidth_otherwise,
            )[0]

    def plot_emission_line_components(self) -> None:
        # Preparation
        self._prepare_data()
        self._prepare_combined_data()
        self._format_secondary_xaxis()
        
        self._plot_other()
        self._plot_selected()
        self._plot_selected_crosses()
        self._plot_combined()
        self._plot_combined_cross()

    def plot_emission_line_verticals(self) -> None:
        self._plot_centres()
        self._plot_edges()
        self._plot_peak_lines()

    def format_annotation(self) -> None:
        self._annotation_lines.clear()
        self._annotation = None

        if self.verbose <= 0:
            return

        if self.annotation_title is not None:
            self._annotation_lines.append(self.annotation_title)

        self._annotation_lines.extend(
            item[1].pretty_label
            for item in sorted(
                self._selected_data.items(),
                key=lambda item: -int(item[0] == 'total')
            )
        )
        if self.verbose >= 3:
            self._annotation_lines.extend(
                data.label for data in self._other_data.values()
            )

        self._annotation = self[0].annotate(
            '\n'.join(self._annotation_lines),
            xy=self.annotation_xy,
            xycoords=self.annotation_xycoords,
            ha=self.annotation_ha,
            va=self.annotation_va,
            fontsize=self.annotation_fontsize,
        )

@dataclass
class SimpleEmissionLineComponents(
    _Base,
    SinglePlotManager,
):
    def plot(self) -> Self:
        self.instantiate_figure()

        self.plot_data()
        self.plot_emission_line_components()
        self.plot_components()
        self.plot_emission_line_verticals()

        self.format_labels()
        self.format_scales()
        self.format_limits()
        self.format_annotation()
        self.format_legends()

        return self

@dataclass
class EmissionLineComponentsWithResiduals(
    _Base,
    StandardWithResidualsPlotManager,
    ResidualLayer,
):
    def plot(self) -> Self:
        self.instantiate_figure()

        self.plot_data()
        self.plot_emission_line_components()
        self.plot_components()
        self.plot_emission_line_verticals()
        self.plot_residuals()

        self.format_labels()
        self.format_scales()
        self.format_limits()
        self.format_annotation()
        self.format_legends()

        return self

@dataclass
class EmissionLineComponentsWithDifferences(
    _Base,
    StandardWithDifferencePlotManager,
    DifferenceLayer,
):
    def plot(self) -> Self:
        self.instantiate_figure()

        self.plot_data()
        self.plot_emission_line_components()
        self.plot_components()
        self.plot_emission_line_verticals()
        self.plot_differences()

        self.format_labels()
        self.format_scales()
        self.format_limits()
        self.format_annotation()
        self.format_legends()

        return self
