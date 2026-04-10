from typing import Optional
from numpy import ndarray, argwhere
from time import perf_counter

class stopwatch:    
    def __init__(self, unit: float = 1):
        self.unit = unit
        
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = perf_counter() - self.start

def get_or_default(
        attr:str, 
        kwargs:dict, 
        default:dict):
    return kwargs.get(attr, default.get(attr))

def _fit_bounds(val, bounds):
    from numpy import inf

    lb = (
        bounds[0]
        if bounds[0] is not None \
        else -inf
    )
    ub = (
        bounds[1] \
        if bounds is not None \
        else inf
    )

    return max([min([val, ub]), lb])

def val_or_default(
        default: dict,
        kwargs: dict,
        key: str
):
    val = kwargs.get(key)    
    out = default[key] \
        if (val is None) \
        else val
    
    return out

def get_bounds_indices(arr: ndarray, bounds: tuple) -> tuple[int, int]:
    indices = argwhere((bounds[0] <= arr) & (arr < bounds[1])).flatten()
    
    if len(indices) == 0: return (0, 0)
    else:                 return (indices[0], indices[-1])
    
def apply_bounds(arr: ndarray, bounds: tuple) -> ndarray:
    idx_left, idx_right = get_bounds_indices(arr, bounds)
    return arr[idx_left:idx_right+1]

def nan_diffs(
    a: ndarray,
    b: ndarray,
    fill: float = 0,
    mask: Optional[ndarray[bool]] = None,
) -> ndarray:
    from numpy import isfinite, ones_like

    mask = isfinite(a) & isfinite(b) \
        if mask is None \
        else mask
    
    diff = fill * ones_like(a)
    diff[mask] = a[mask] - b[mask]

    return diff

def nan_residuals(
    y: ndarray, 
    f: ndarray, 
    dy: ndarray,
    z_fill: float = 0,
    mask: Optional[ndarray[bool]] = None,
) -> ndarray:
    from numpy import isfinite, ones_like

    mask = isfinite(y) & isfinite(f) & (dy > 0) \
        if mask is None \
        else mask
    
    if mask.all():
        z = (y - f) / dy
    else:
        z = z_fill * ones_like(y)
        z[mask] = (y[mask] - f[mask]) / dy[mask]

    return z