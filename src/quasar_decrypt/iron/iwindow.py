from logging import getLogger
from typing import ClassVar
from dataclasses import dataclass

from ..utils.specdata import SpecData
from quasar_typing.misc.literals import BGFlux

logger = getLogger(__name__)
logger.disabled = not getLogger().hasHandlers()

@dataclass(init=False)
class IWindow(SpecData):
    default_bg: ClassVar[BGFlux] = {'pl', 'ba', 'hg', 'em'}