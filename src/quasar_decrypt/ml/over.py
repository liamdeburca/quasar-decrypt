__all__ = ['Over']

from logging import getLogger
from pathlib import Path
from pickle import load
from typing import Callable, ClassVar
from scipy.stats import chi2
from numpy import float64, dot, zeros_like, log, minimum, maximum, nan_to_num, inf, empty
from functools import lru_cache, cached_property

from dataclasses import dataclass, field

from quasar_typing.numpy import FloatVector
from quasar_typing.astropy import Model_, CompoundModel_
from quasar_utils.absorption.absorption import fft

logger = getLogger(__name__)

### Load 'is over-fitted models'

_this_file: Path = Path(__file__).resolve()
PATH_TO_MODELS = _this_file.parents[0] / 'models'

@lru_cache(maxsize=1)
def _get_classifiers() -> dict[int, Callable]:
    classifiers = {}
    for fname in PATH_TO_MODELS.glob('over_*.pkl'):
        n = int(fname.name.removesuffix('.pkl').split('_')[1])
        with open(fname, 'rb') as f:
            classifiers[n] = load(f)
    return classifiers

###

def _number_of_free_params(model: Model_ | CompoundModel_) -> int:
    n = 0
    ms = (model,) if model.n_submodels == 1 else model
    for m in ms:
        for paran_name in m.param_names:
            param = getattr(m, paran_name)
            if not (param.fixed or bool(param.tied)):
                n += 1
    return n

@dataclass    
class Over:
    # Coordinates
    x: FloatVector
    y: FloatVector
    dy: FloatVector

    # Fits
    fit_initial: Callable[[FloatVector], FloatVector] | None = None
    fit_final: Callable[[FloatVector], FloatVector] | None = None

    snr: float = field(default=10, kw_only=True)

    # Size of spectrum
    n_pix: int = field(init=False)
    # Baseline fit
    n_initial: int = field(default=0, init=False)
    ddof_initial: int = field(default=0, init=False)
    f_initial: FloatVector = field(init=False)
    z_initial: FloatVector = field(init=False)
    chi2_initial: float = field(init=False)
    # New fit
    n_final: int = field(default=0, init=False)
    ddof_final: int = field(default=0, init=False)
    f_final: FloatVector = field(init=False)
    z_final: FloatVector = field(init=False)
    chi2_final: float = field(init=False)

    measures: ClassVar[frozenset[str]] = frozenset({
        'snr', 'llh', 'log_chi2', 'chi2n', 'log_fft', 'log_cov',
    })
    feature_names: ClassVar[frozenset[str]] = frozenset({
        r"$SNR$", 
        r"$\Delta LLH_{\text{mod.}}$", 
        r"$\Delta\log\left(p(\chi^2)\right)$", 
        r"$\Delta\chi^2_{\nu}$", 
        r"$\Delta\log\left(p(FFT)\right)$", 
        r"$\Delta\log\left(r_{\text{cov.}}\right)$",
    })
    classifiers: ClassVar[dict[int, Callable]] = _get_classifiers()

    def __post_init__(self):
        self.n_pix = self.x.size

        if (model := self.fit_initial) is not None:
            self.n_initial = model.n_submodels
            self.ddof_initial = _number_of_free_params(model)
            self.f_initial = model(self.x)
        else:
            self.f_initial = zeros_like(self.x)

        self.z_initial = (self.y - self.f_initial) / self.dy
        self.chi2_initial = dot(self.z_initial, self.z_initial)

        if (model := self.fit_final) is not None:
            self.n_final = model.n_submodels
            self.ddof_final = _number_of_free_params(model)
            self.f_final = model(self.x)
        else:
            self.f_final = zeros_like(self.x)

        self.z_final = (self.y - self.f_final) / self.dy
        self.chi2_final = dot(self.z_final, self.z_final)

    @cached_property
    def llh(self) -> tuple[float, float]:
        return (
            (1 - self.chi2_initial / self.n_pix) / 2,
            (1 - self.chi2_final / self.n_pix) / 2,
        )
    
    @cached_property
    def log_chi2(self) -> tuple[float, float]:
        return (
            chi2(self.n_pix).logsf(self.chi2_initial),
            chi2(self.n_pix).logsf(self.chi2_final),
        )
    
    @cached_property
    def chi2n(self) -> tuple[float, float]:
        chi2n_initial = inf
        if (num := (self.n_pix - self.ddof_initial)) > 0:
            chi2n_initial = self.chi2_initial / num

        chi2n_final = inf
        if (num := (self.n_pix - self.ddof_final)) > 0:
            chi2n_final = self.chi2_final / num

        return chi2n_initial, chi2n_final
    
    @cached_property
    def log_fft(self) -> tuple[float, float]:
        return (
            log(10) * fft(self.z_initial, log=True)[1],
            log(10) * fft(self.z_final, log=True)[1],
        )
    
    @cached_property
    def log_cov(self) -> tuple[float, float]:
        def func(f: FloatVector) -> float:
            a = minimum(f, self.y).sum()
            b = maximum(f, self.y).sum()
            return -inf if a <= 0 else log(a) - log(b)
        return func(self.f_initial), func(self.f_final)
    
    def getFeatures(self, as_dict: bool = False) -> FloatVector:
        if as_dict:
            return dict(zip(self.measures, self.getFeatures(as_dict=False)[0]))

        features = empty(len(self.measures), dtype=float64)
        for i, measure in enumerate(self.measures):
            feature = getattr(self, measure)
            if measure == 'snr':
                features[i] = feature
            else:
                val_i, val_f = nan_to_num(feature, neginf=-1e8, posinf=1e8)
                features[i] = val_f - val_i

        return features[None,:]

    def __call__(self, n: int | None = None, n_default: int = 2) -> bool:
        classifier = self.classifiers.get(
            n or self.n_final,
            self.classifiers[n_default]
        )
        X = self.getFeatures(as_dict=False)

        dchi2n = self.chi2n[1] - self.chi2n[0]
        y = not (dchi2n < 0)
        # y = classifier.predict(X)[0]

        msg = ">>> Is over-fitted? '{}' ".format(bool(y)) \
            + "(snr: {:.1f}) ".format(self.snr) \
            + "(llh: {:.1f} -> {:.1f}) ".format(*self.llh) \
            + "(log-chi2: {:.1e} -> {:.1e}) ".format(*self.log_chi2) \
            + "(chi2n: {:.2f} -> {:.2f}) ".format(*self.chi2n) \
            + "(log-fft: {:.1e} -> {:.1e}) ".format(*self.log_fft) \
            + "(log-cov: {:.1e} -> {:.1e}).".format(*self.log_cov)
        logger.debug(msg)

        return bool(y)
