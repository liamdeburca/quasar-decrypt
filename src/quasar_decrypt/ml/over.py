from logging import getLogger
import os
from pathlib import Path
from pickle import load
from typing import Optional, Union, Callable
from scipy.stats import chi2
from numpy import float64, dot, zeros_like, isfinite, log, minimum, maximum, nan_to_num, inf, diff, array
from numpy.typing import NDArray
from functools import lru_cache

from quasar_utils.absorption.absorption import fft

logger = getLogger(__name__)
logger.disabled = not getLogger().hasHandlers()

### Load 'is over-fitted models'

directory = os.path.join(
    Path(__file__).parents[0], 
    'models',
)

models = {}
for fname in os.listdir(directory):

    if fname.split('.')[-1] == 'pkl':
        _fname = fname.split('.')[0]
        model_type = _fname.split('_')[0]

        if model_type == 'over':
            n = int(_fname.split('_')[1])

            path = os.path.join(directory, fname)
            with open(path, 'rb') as f:
                models[n] = load(f)

###

def _number_of_free_params(model):
    n = 0
    for submodel in [model] if model.n_submodels == 1 else model:
        for param_name in submodel.param_names:
            param = getattr(submodel, param_name)
            
            if not (param.fixed or bool(param.tied)):
                n += 1
    return n

class Over:
    measures = [
        'SNR',
        'LLH',
        'LogChi2',
        'Chi2n',
        'LogFFT',
        'LogCov'
    ]

    feature_names = [
        r"$SNR$", 
        r"$\Delta LLH_{\text{mod.}}$", 
        r"$\Delta\log\left(p(\chi^2)\right)$", 
        r"$\Delta\chi^2_{\nu}$", 
        r"$\Delta\log\left(p(FFT)\right)$", 
        r"$\Delta\log\left(r_{\text{cov.}}\right)$"
    ]

    models = models

    def __init__(
        self,
        x: NDArray[float64],
        y: NDArray[float64],
        dy: NDArray[float64],
        fit_initial: Callable | None = None,
        fit_final: Callable | None = None,
        snr: float | int | None = None,
    ):
        self.N = len(x)

        self.x: NDArray[float64] = x
        self.y: NDArray[float64] = y
        self.dy: NDArray[float64] = dy

        ### Fits

        self.fit_initial = self.fit_final = zeros_like
        self.n_initial = self.n_final = 0
        self.ddof_initial = self.ddof_final = 0

        if fit_initial is not None:
            self.fit_initial = fit_initial
            self.n_initial = fit_initial.n_submodels
            self.ddof_initial = _number_of_free_params(fit_initial)

        if fit_final is not None:
            self.fit_final = fit_final
            self.n_final = fit_final.n_submodels
            self.ddof_final = _number_of_free_params(fit_final)

        ###

        self.f_initial = self.fit_initial(self.x)
        self.f_final = self.fit_final(self.x)

        self.z_initial = (y - self.f_initial) / dy
        self.z_final = (y - self.f_final) / dy
        self.chi2_initial = dot(self.z_initial, self.z_initial)
        self.chi2_final = dot(self.z_final, self.z_final)

        self.snr = snr or 10

        logger.debug(
            "Instantiated: {}".format(self)
        )

    def __str__(self):
        s = "'Over' class [{} ({}) -> {} ({})] w/ {}/{} (fittable).".format(
            self.ddof_initial,
            self.n_initial,
            self.ddof_final,
            self.n_final,
            int(isfinite(self.y).sum()),
            len(self.y),
        )

        return s

    def __call__(
        self,
        n: Optional[int] = None,
        n_default: int = 2,
    ) -> bool:

        classifier = self.models.get(
            n or self.n_final,
            self.models[n_default]
        )
        X = self.getFeatures()
        y = classifier.predict(X)[0]

        msg = ">>> Is over-fitted? '{}' ".format(bool(y)) \
            + "(snr: {:.1f}) ".format(self.snr) \
            + "(llh: {:.1f} -> {:.1f}) ".format(*self.getLLH()) \
            + "(log-chi2: {:.1e} -> {:.1e}) ".format(*self.getLogChi2()) \
            + "(chi2n: {:.2f} -> {:.2f}) ".format(*self.getChi2n()) \
            + "(log-fft: {:.1e} -> {:.1e}) ".format(*self.getLogFFT()) \
            + "(log-cov: {:.1e} -> {:.1e}).".format(*self.getLogCov())
        logger.debug(msg)

        return bool(y)
    
    @lru_cache(maxsize=1)
    def getSNR(self) -> float:
        return self.snr
    
    @lru_cache(maxsize=1)
    def getLLH(self) -> tuple[float, float]:
        return (
            (1 - self.chi2_initial / self.N) / 2,
            (1 - self.chi2_final / self.N) / 2
        )
        
    @lru_cache(maxsize=1)
    def getLogChi2(self) -> tuple[float, float]:
        return (
            chi2(self.N).logsf(self.chi2_initial),
            chi2(self.N).logsf(self.chi2_final)
        )

    @lru_cache(maxsize=1)
    def getChi2n(self) -> tuple[float, float]:
        return (
            self.chi2_initial / (self.N - self.ddof_initial),
            self.chi2_final / (self.N - self.ddof_final)
        )

    @lru_cache(maxsize=1)
    def getLogFFT(self) -> tuple[float, float]:
        return (
            log(10) * fft(self.z_initial, log=True)[1],
            log(10) * fft(self.z_final,   log=True)[1],
        )
    
    @lru_cache(maxsize=1)
    def getLogCov(self) -> tuple[float, float]:
        def func(f: NDArray[float64]) -> float:
            a = minimum(f, self.y).sum()
            b = maximum(f, self.y).sum()
            return -inf if a <= 0 else log(a) - log(b)
        
        return func(self.f_initial), func(self.f_final)

    @lru_cache(maxsize=1)
    def getFeatures(
        self,
        as_dict: bool = False
    ) -> NDArray[float64]:
        if as_dict:
            return {m: f for m, f in zip(self.measures, self.getFeatures()[0])}

        features = [self.getSNR()]
        for measure in self.measures[1:]:
            features.append(
                diff(
                    nan_to_num(
                        list(getattr(self, f"get{measure}")()),
                        neginf = -1e8,
                        posinf = 1e8
                    )
                )[0]
            )

        return array(features)[None,:]    