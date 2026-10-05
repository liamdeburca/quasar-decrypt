from dataclasses import dataclass, field

from .misc import HoriLoc, VertLoc


@dataclass
class SingleLabelsManager:
    """Basic labels manager for a single subplot."""
    xlabel: str | None = field(default="Rest wavelength", kw_only=True)
    ylabel: str | None = field(default="Flux density", kw_only=True)

    with_units: bool = field(default=True, kw_only=True)
    xloc: HoriLoc = field(default="right", kw_only=True)
    yloc: VertLoc = field(default="top", kw_only=True)

    def _format_xlabel(self) -> None:
        xlabel = self.xlabel
        if self.with_units:
            xlabel += f" ({self.info.units.wavelength_unit.to_string('latex_inline')})"

        if self.sharex:
            self[0].set_xlabel(xlabel, loc=self.xloc)
        else:
            for ax in self:
                ax.set_xlabel(xlabel, loc=self.xloc)

    def _format_ylabel(self) -> None:
        ylabel = self.ylabel
        if self.with_units:
            ylabel += f" ({self.info.units.flux_unit.to_string('latex_inline')})"

        self[0].set_ylabel(ylabel, loc=self.yloc)

    def format_labels(self) -> None:
        if self.xlabel: 
            self._format_xlabel()
        if self.ylabel: 
            self._format_ylabel()

# Two subplots

@dataclass
class DoubleLabelsManager:
    """Labels manager for a double subplot."""
    xlabel: str | None = field(default="Rest wavelength", kw_only=True)
    y0label: str | None = field(default="Flux density", kw_only=True)
    y1label: str | None = field(default=None, kw_only=True)

    with_units: bool = field(default=True, kw_only=True)
    xloc: HoriLoc = field(default="right", kw_only=True)
    y0loc: VertLoc = field(default="top", kw_only=True)
    y1loc: VertLoc = field(default="center", kw_only=True)

    def _format_xlabel(self) -> None:
        xlabel = self.xlabel
        if self.with_units:
            xlabel += f" ({self.info.units.wavelength_unit.to_string('latex_inline')})"

        if self.sharex:
            self[-1].set_xlabel(xlabel, loc=self.xloc)
        else:
            for ax in self:
                ax.set_xlabel(xlabel, loc=self.xloc)

    def _format_y0label(self) -> None:
        ylabel = self.y0label
        if self.with_units:
            ylabel += f" ({self.info.units.flux_unit.to_string('latex_inline')})"

        self[0].set_ylabel(ylabel, loc=self.y0loc)

    def _format_y1label(self) -> None:
        self[1].set_ylabel(self.y1label, loc=self.y1loc)

    def format_labels(self) -> None:
        if self.xlabel: 
            self._format_xlabel()
        if self.y0label: 
            self._format_y0label()
        if self.y1label: 
            self._format_y1label()


@dataclass
class DoubleLabelsManagerDifference(DoubleLabelsManager):
    """Labels manager for a double subplot with a data-model difference window.
    """
    y1label: str | None = field(default=r"$\Delta$", kw_only=True)
    y1loc: VertLoc = field(default="center", kw_only=True)


@dataclass
class DoubleLabelsManagerResidual(DoubleLabelsManager):
    """Labels manager for a double subplot with a data-model residuals window.
    """
    y1label: str | None = field(default=r"$\chi$", kw_only=True)
    y1loc: VertLoc = field(default="center", kw_only=True)


# Three subplots

@dataclass
class TripleLabelsManager(DoubleLabelsManager):
    """Labels manager for a triple subplot."""
    y2label: str | None = field(default=None, kw_only=True)
    y2loc: VertLoc = field(default="center", kw_only=True)

    def _format_y2label(self) -> None:
        self[2].set_ylabel(self.y2label, loc=self.y2loc)

    def format_labels(self) -> None:
        super().format_labels()
        if self.y2label: 
            self._format_y2label()

@dataclass
class TripleLabelsManagerDifferenceAndTemplate(TripleLabelsManager):
    """Labels manager for a triple subplot with a data-model difference window 
    and a template window."""
    y1label: str | None = field(default=r"$\Delta$", kw_only=True)
    y1loc: VertLoc = field(default="center", kw_only=True)

    y2label: str | None = field(default="Template", kw_only=True)
    y2loc: VertLoc = field(default="center", kw_only=True)

@dataclass
class TripleLabelsManagerResidualAndTemplate(TripleLabelsManager):
    """Labels manager for a triple subplot with a data-model residuals window 
    and a template window."""
    y1label: str | None = field(default=r"$\chi$", kw_only=True)
    y1loc: VertLoc = field(default="center", kw_only=True)

    y2label: str | None = field(default="Template", kw_only=True)
    y2loc: VertLoc = field(default="center", kw_only=True)