__all__ = [
    'get_mask',
    'create_cached_get_mask',
    'get_log',
]

from numpy import zeros_like, isfinite, ndarray, full_like, nan, log
from functools import lru_cache
from quasar_typing.numpy import FloatVector, BoolVector

def get_mask(
    arr: FloatVector, 

    lb: float,
    ub: float,
) -> BoolVector:
    """
    NaN-insensitive function for finding the mask covering a '_SpecData' object.
    """
    mask = zeros_like(arr, dtype=bool)
    not_nan = isfinite(arr)
    
    # Early exit if bounds don't overlap with data range
    if (arr[not_nan][-1] < lb) or (ub < arr[not_nan][0]):
        return mask
    
    mask[:] = (lb <= arr) & (arr < ub)
    return mask

def create_cached_get_mask(arr: FloatVector, maxsize: int | None = 1):
    @lru_cache(maxsize=maxsize)
    def cached_get_mask(lb: float, ub: float) -> BoolVector:
        nonlocal arr
        return get_mask(arr, lb, ub)
    return cached_get_mask

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