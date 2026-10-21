__all__ = [
    "CachedGetMask",
    "get_log",
    "get_mask",
    "stopwatch",
]

from time import perf_counter
from typing import Any

import numpy as np
import pydantic
from cachetools import LRUCache
from pydantic.dataclasses import dataclass
from quasar_typing.cachetools import LRUCacheType
from quasar_typing.numpy import BoolVector, FloatVector


def get_mask(
    arr: FloatVector,
    lb: float,
    ub: float,
) -> BoolVector:
    """
    NaN-insensitive function for finding the mask covering a '_BaseSpec' object.
    """
    mask = np.zeros_like(arr, dtype=bool)
    not_nan = np.isfinite(arr)

    # Early exit if bounds don't overlap with data range
    if (arr[not_nan][-1] < lb) or (ub < arr[not_nan][0]):
        return mask

    mask[:] = (lb <= arr) & (arr < ub)
    return mask


@dataclass
class CachedGetMask:
    arr: FloatVector
    cache: LRUCacheType

    @classmethod
    @pydantic.validate_call
    def create(cls, arr: FloatVector, maxsize: int = 1):
        arr = arr.copy(order="C")
        arr.setflags(write=False)
        return cls(arr, LRUCache(maxsize))
    
    def __call__(self, lb: float, ub: float) -> BoolVector:
        key = (lb, ub)
        if key not in self.cache:
            self.cache[key] = get_mask(self.arr, lb, ub)
        return self.cache[key]

    def __getstate__(self) -> dict[str, Any]:
        return {
            "arr": self.arr,
            "cache": self.cache,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.arr = state["arr"]
        self.cache = state["cache"]


def get_log(
    arr: FloatVector,
    norm: float | FloatVector,
    mask: BoolVector,
) -> FloatVector:
    if not isinstance(norm, np.ndarray):
        norm = np.full_like(arr, fill_value=norm)

    arr_log = np.full_like(arr, fill_value=np.nan)
    arr_log[mask] = np.log(arr[mask] / norm[mask])

    return arr_log


class stopwatch:
    def __init__(self, unit: float = 1):
        self.unit = unit

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = perf_counter() - self.start