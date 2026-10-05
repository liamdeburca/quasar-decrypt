from dataclasses import dataclass, field

from .misc import Scale


@dataclass 
class SingleScalesManager:
    """Dataclass for managing axes scales for a single subplot."""
    xscale: Scale = field(default="linear", kw_only=True)
    yscale: Scale = field(default="linear", kw_only=True)

    def format_scales(self) -> None:
        self[0].set_xscale(self.xscale)
        self[0].set_yscale(self.yscale)


@dataclass
class DoubleScalesManager:
    """Dataclass for managing axes scales for a double subplot."""
    xscale: Scale = field(default="linear", kw_only=True)
    y0scale: Scale = field(default="linear", kw_only=True)
    y1scale: Scale = field(default="linear", kw_only=True)

    def format_scales(self) -> None:
        self[0].set_xscale(self.xscale)
        self[0].set_yscale(self.y0scale)
        self[1].set_yscale(self.y1scale)

@dataclass
class TripleScalesManager(DoubleScalesManager):
    """Dataclass for managing axes scales for a triple subplot."""
    y2scale: Scale = field(default="linear", kw_only=True)

    def format_scales(self) -> None:
        super().format_scales()
        self[2].set_yscale(self.y2scale)