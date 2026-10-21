from logging import getLogger
from typing import ClassVar

import pydantic
from quasar_typing.string_selections import BackgroundFlux

from quasar_decrypt.utils import _SpecWindow

logger = getLogger(__name__)


@pydantic.dataclasses.dataclass
class CWindow(_SpecWindow):
    default_bg: ClassVar[BackgroundFlux] = BackgroundFlux({"all", "pl"})
