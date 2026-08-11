from dataclasses import dataclass
from logging import getLogger
from typing import ClassVar

from quasar_typing.misc import BackgroundFlux

from ..utils import _SpecWindow

logger = getLogger(__name__)


@dataclass(init=False)
class IWindow(_SpecWindow):
    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({"all", "fe"})
