### Load 'is under-fitted models'

import os
from pathlib import Path
from pickle import load
from functools import lru_cache
from collections.abc import Callable
from numpy import dot, zeros_like, minimum, maximum, nan_to_num, nan, float64
from numpy.typing import NDArray
from scipy.stats import chi2

from quasar_utils.absorption.absorption import fft

directory = os.path.join(
    Path(__file__).parent, 
    'models',
)

models = {}
for fname in os.listdir(directory):

    if fname.split('.')[-1] == 'pkl':
        _fname = fname.split('.')[0]
        model_type = _fname.split('_')[0]

        if model_type == 'under':
            n = int(_fname.split('_')[1])

            path = os.path.join(directory, fname)
            with open(path, 'rb') as f:
                models[n] = load(f)

###


def _number_of_free_params(model):
    n = 0
    for submodel in model:

        for param_name in submodel.param_names:
            param = getattr(submodel, param_name)
            
            if not (param.fixed or bool(param.tied)):
                n += 1

    return n

class Under:
    measures = [
        'SNR',
        'LLH',
        'Chi2',
        'FFT',
        'Cov',
    ]

    feature_names = [
        r"$SNR$",
        r"$LLH_{\text{mod.}}$",
        r"$p(\chi^2)$",
        r"$p(\text{FFT})$",
        r"$r_{\text{cov.}}$",
    ]

    models = models

    def __init__(
        self,
        x: NDArray[float64],
        y: NDArray[float64],
        dy: NDArray[float64],
        fit: Callable|None = None,
        snr: float|int|None = None,
    ):
        self.N = len(x)

        self.x = x
        self.y = y
        self.dy = dy

        self.fit = fit or zeros_like

        self.n = 0 \
            if (fit is None) \
            else self.fit.n_submodels
        
        self.f = fit(x)
        self.z = (self.y - self.f) / self.dy
        self.chi2 = dot(self.z, self.z)

        self.snr = snr or 10

    def __call__(
        self,
        n: int|None = None,
        n_default: int = 1,
    ) -> bool:

        classifier = self.models.get(
            n or self.n,
            self.models[n_default]
        )
        X = self.getFeatures()
        y = classifier.predict(X)[0]

        return bool(y)

    @lru_cache(maxsize=1)
    def getSNR(self) -> float:
        return self.snr

    @lru_cache(maxsize=1)
    def getLLH(self) -> float:
        return (1 - self.chi2 / self.N) / 2
    
    @lru_cache(maxsize=1)
    def getChi2(self) -> float:
        return chi2(self.N).sf(self.chi2)

    @lru_cache(maxsize=1)
    def getFFT(self) -> float:
        return fft(self.z, log=False)[1]

    @lru_cache(maxsize=1)
    def getCov(self) -> float:
        den = maximum(self.f, self.y).sum()
        return nan if den == 0 else minimum(self.f, self.y).sum() / den

    @lru_cache(maxsize=1)
    def getFeatures(
        self,
        as_dict: bool = False,
        ) -> NDArray[float64]:
        if as_dict:
            return {m: f for m, f in zip(self.measures, self.getFeatures()[0])}
        
        return nan_to_num(
            [getattr(self, f"get{measure}")() for measure in self.measures],
            neginf = -1e8,
            posinf = 1e8
        )[None,:]