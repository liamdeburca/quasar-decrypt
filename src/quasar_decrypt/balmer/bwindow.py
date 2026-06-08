"""
The Balmer window class: 'BWindow'.
"""
from logging import getLogger
from typing import ClassVar
from dataclasses import dataclass

from quasar_typing.misc import BackgroundFlux

from quasar_decrypt.utils.specdata import SpecData

logger = getLogger(__name__)

@dataclass(init=False)
class BWindow(SpecData):
    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({'all', 'ba'})