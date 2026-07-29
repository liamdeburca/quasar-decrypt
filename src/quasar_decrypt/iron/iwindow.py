from dataclasses import dataclass
from logging import getLogger
from typing import ClassVar

from quasar_typing.misc import BackgroundFlux

from quasar_decrypt.utils.specdata import SpecData

logger = getLogger(__name__)


@dataclass(init=False)
class IWindow(SpecData):
    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({"all", "fe"})
