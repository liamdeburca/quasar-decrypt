__all__ = [
    "CachedGetMask",
    "get_log",
    "get_mask",
]

from typing import Any

from cachetools import LRUCache
from numpy import full_like, isfinite, log, nan, ndarray, zeros_like
from pydantic.dataclasses import dataclass
from quasar_typing.cachetools import LRUCache_
from quasar_typing.numpy import BoolVector, FloatVector
from quasar_utils.decorators import validate_call


def get_mask(
    arr: FloatVector,
    lb: float,
    ub: float,
) -> BoolVector:
    """
    NaN-insensitive function for finding the mask covering a '_BaseSpec' object.
    """
    mask = zeros_like(arr, dtype=bool)
    not_nan = isfinite(arr)

    # Early exit if bounds don't overlap with data range
    if (arr[not_nan][-1] < lb) or (ub < arr[not_nan][0]):
        return mask

    mask[:] = (lb <= arr) & (arr < ub)
    return mask


@dataclass
class CachedGetMask:
    arr: FloatVector
    cache: LRUCache_

    @classmethod
    @validate_call
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
    if not isinstance(norm, ndarray):
        norm = full_like(arr, fill_value=norm)

    arr_log = full_like(arr, fill_value=nan)
    arr_log[mask] = log(arr[mask] / norm[mask])

    return arr_log
