from dataclasses import dataclass, field

from numpy import nanmax, nanmin

from .misc import AxesLimits


@dataclass
class SingleLimitsManager:
    """
    Dataclass managing the axes limits.
    """
    xlim: AxesLimits = field(default=(None, None), kw_only=True)
    ylim: AxesLimits = field(default=(None, None), kw_only=True)

    def _format_xlim(self) -> None:
        if self.xlim == (None, None):
            self.xlim = self[0].get_xlim()
        else:
            lb, ub = self.xlim
            if lb is None and (self.mask is None or self.mask.any()):
                lb = nanmin(self._x)
            if ub is None and (self.mask is None or self.mask.any()):
                ub = nanmax(self._x)

            self.xlim = (lb, ub)
            self[0].set_xlim(lb, ub)

    def _format_ylim(self) -> None:
        if self.ylim == (None, None):
            self.ylim = self[0].get_ylim()
        else:
            self[0].set_ylim(*self.ylim)

    def format_limits(self) -> None:
        self._format_xlim()
        self._format_ylim()

@dataclass
class DoubleLimitsManager:
    xlim: AxesLimits = field(default=(None, None), kw_only=True)
    y0lim: AxesLimits = field(default=(None, None), kw_only=True)
    y1lim: AxesLimits = field(default=(None, None), kw_only=True)

    def _format_xlim(self) -> None:
        SingleLimitsManager._format_xlim(self)

    def _format_y0lim(self) -> None:
        if self.y0lim == (None, None):
            self.y0lim = self[0].get_ylim()
        else:
            self[0].set_ylim(*self.y0lim)

    def _format_y1lim(self) -> None:
        if self.y1lim == (None, None):
            self.y1lim = self[1].get_ylim()
        else:
            self[1].set_ylim(*self.y1lim)

    def format_limits(self) -> None:
        self._format_xlim()
        self._format_y0lim()
        self._format_y1lim()

@dataclass 
class TripleLimitsManager(DoubleLimitsManager):
    y2lim: AxesLimits = field(default=(None, None), kw_only=True)

    def _format_y2lim(self) -> None:
        if self.y2lim == (None, None):
            self.y2lim = self[2].get_ylim()
        else:
            self[2].set_ylim(*self.y2lim)

    def format_limits(self) -> None:
        super().format_limits()
        self._format_y2lim()