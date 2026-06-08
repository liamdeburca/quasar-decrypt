"""
The Host window class: 'HWindow'.
"""
from logging import getLogger
from typing import ClassVar
from dataclasses import dataclass

from quasar_decrypt.utils.specdata import SpecData
from quasar_typing.misc import BackgroundFlux

logger = getLogger(__name__)

@dataclass(init=False)
class HWindow(SpecData):
    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({'all', 'hg'})