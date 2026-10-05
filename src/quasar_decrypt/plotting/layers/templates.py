from dataclasses import dataclass, field

from matplotlib.lines import Line2D
from quasar_models import (
    BalmerModel,
    HostGalaxyModel,
    IronModel,
)
from quasar_models._core.convolution import convolve as template_convolve
from quasar_models.modeling.template import BaseTemplate


def _evaluate_template(fwhm: float, template: BaseTemplate):
    return template_convolve(
        template.data / template.normalisation,
        template.fwhm,
        fwhm,
        template.sigma_res, template.n_scales,
    )

@dataclass
class _TemplatesLayer:
    """Dataclass for plotting the template in the 3rd subplot."""

    template_layer_base_linestyle: str = field(
        default="dotted",
        kw_only=True,
    )

    _template_layer_templates: dict[str, BaseTemplate] = field(
        default_factory=dict, 
        init=False,
    )
    _template_layer_model_fwhms: dict[str, float] = field(
        default_factory=dict, 
        init=False,
    )
    _template_layer_colors: dict[str, str] = field(
        default_factory=dict, 
        init=False,
    )
    _template_layer_linestyles: dict[str, str] = field(
        default_factory=dict, 
        init=False,
    )

    _template_layer_lines: dict[str, list[Line2D]] = field(
        default_factory=dict, 
        init=False,
    )

    def plot_templates(self) -> None:
        """
        Plots all templates in the 3rd subplot.
        """
        self[2].axhline(1.0, color='k')

        for name, template in self._template_layer_templates.items():
            y = _evaluate_template(template.fwhm_norm, template)

            color = self._template_layer_colors[name]
            linestyle = self._template_layer_linestyles[name]

            self._template_layer_lines[name] = self[2].plot(
                template.x, y,
                color=color,
                linestyle=self.template_layer_base_linestyle,
                label=name,
            )
            self[2].axvline(template.x_norm, color=color, linestyle='solid')

            fwhm = self._template_layer_model_fwhms[name]
            if fwhm == template.fwhm_norm:
                continue

            f = _evaluate_template(fwhm, template)

            self._template_layer_lines[name].append(self[2].plot(
                template.x, f,
                color=color,
                linestyle=linestyle,
                label=name,
            )[0])



@dataclass
class IronTemplatesLayer(_TemplatesLayer):
    def _prepare_iron_templates(self) -> None:
        fe_models: list[IronModel] = self.fe_model        
        for fe_model in fe_models:
            name = fe_model.name
            self._template_layer_templates[name] = fe_model.template
            self._template_layer_model_fwhms[name] = fe_model.fwhm.value
            self._template_layer_colors[name] = self.fe_color
            self._template_layer_linestyles[name] = self.fe_linestyle

    def plot_templates(self) -> None:
        if self.fe_model is not None:
            self._prepare_iron_templates()
        super().plot_templates()

@dataclass
class BalmerTemplatesLayer(_TemplatesLayer):
    def _prepare_balmer_templates(self) -> None:
        ba_model: BalmerModel = self.ba_model

        continuum_template = ba_model.continuum_template
        name = continuum_template.name
        self._template_layer_templates[name] = continuum_template
        self._template_layer_model_fwhms[name] = ba_model.fwhm.value
        self._template_layer_colors[name] = self.ba_color
        self._template_layer_linestyles[name] = self.ba_linestyle

        series_template = ba_model.series_template
        name = series_template.name
        self._template_layer_templates[name] = series_template
        self._template_layer_model_fwhms[name] = ba_model.fwhm.value
        self._template_layer_colors[name] = self.ba_color
        self._template_layer_linestyles[name] = self.ba_linestyle

    def plot_templates(self) -> None:
        if self.ba_model is not None:
            self._prepare_balmer_templates()
        super().plot_templates()

@dataclass
class HostTemplatesLayer(_TemplatesLayer):
    def _prepare_host_templates(self) -> None:
        hg_model: HostGalaxyModel = self.hg_model

        template = hg_model.template
        name = template.name
        self._template_layer_templates[name] = template
        self._template_layer_model_fwhms[name] = hg_model.fwhm.value
        self._template_layer_colors[name] = self.hg_color
        self._template_layer_linestyles[name] = self.hg_linestyle

    def plot_templates(self) -> None:
        if self.hg_model is not None:
            self._prepare_host_templates()
        super().plot_templates()

@dataclass
class AllTemplatesLayer(IronTemplatesLayer, BalmerTemplatesLayer, HostTemplatesLayer):
    def plot_templates(self) -> None:
        if self.fe_model is not None:
            self._prepare_iron_templates()
        if self.ba_model is not None:
            self._prepare_balmer_templates()
        if self.hg_model is not None:
            self._prepare_host_templates()

        super().plot_templates()