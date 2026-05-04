__all__ = ['Under']

from pathlib import Path
from pickle import load
from functools import lru_cache, cached_property
from typing import Callable, ClassVar
from numpy import dot, zeros_like, minimum, maximum, nan_to_num, nan, float64, empty
from scipy.stats import chi2

from quasar_typing.numpy import FloatVector
from quasar_utils.absorption.absorption import fft
from dataclasses import dataclass, field

### Load 'is under-fitted models'

_this_file: Path = Path(__file__).resolve()
PATH_TO_MODELS = _this_file.parents[0] / 'models'

@lru_cache(maxsize=1)
def _get_classifiers() -> dict[int, Callable]:
    classifiers = {}
    for fname in PATH_TO_MODELS.glob('under_*.pkl'):
        n = int(fname.name.removesuffix('.pkl').split('_')[1])
        with open(fname, 'rb') as f:
            classifiers[n] = load(f)
    return classifiers

###

@dataclass
class Under:
    # Coordinates
    x: FloatVector
    y: FloatVector
    dy: FloatVector

    # Fit
    fit: Callable[[FloatVector], FloatVector] | None = None

    snr: float = field(default=10, kw_only=True)

    # Size of spectrum
    n_pix: int = field(init=False)
    # Fit parameters
    n: int = field(default=0, init=False)
    f: FloatVector = field(init=False)
    z: FloatVector = field(init=False)
    chi2: float = field(init=False)

    measures: ClassVar[frozenset[str]] = frozenset({
        'snr', 'llh', 'chi2', 'fft', 'cov',
    })
    feature_names: ClassVar[frozenset[str]] = frozenset({
        r"$SNR$",
        r"$LLH_{\text{mod.}}$",
        r"$p(\chi^2)$",
        r"$p(\text{FFT})$",
        r"$r_{\text{cov.}}$",
    })
    classifiers: ClassVar[dict[int, Callable]] = _get_classifiers()

    def __post_init__(self):
        self.n_pix = self.x.size

        if (model := self.fit) is not None:
            self.n = model.n_submodels
            self.f = model(self.x)
        else:
            self.f = zeros_like(self.x)

        self.z = (self.y - self.f) / self.dy
        self.chi2 = dot(self.z, self.z)

    @cached_property
    def llh(self) -> float:
        return (1 - self.chi2 / self.n_pix) / 2
    
    @cached_property
    def chi2_sf(self) -> float:
        return chi2(self.n_pix).sf(self.chi2)

    @cached_property
    def fft(self) -> float:
        return fft(self.z, log=False)[1]

    @cached_property
    def cov(self) -> float:
        den = maximum(self.f, self.y).sum()
        return nan if den == 0 else minimum(self.f, self.y).sum() / den

    def getFeatures(self, as_dict: bool = False) -> FloatVector:
        if as_dict:
            return dict(zip(self.measures, self.getFeatures()[0]))

        features = empty(len(self.measures), dtype=float64)
        for i, measure in enumerate(self.measures):
            if measure == 'snr':
                features[i] = self.snr
            elif measure == 'chi2':
                features[i] = self.chi2_sf
            else:
                features[i] = getattr(self, measure)

        return nan_to_num(features, neginf=-1e8, posinf=1e8)[None,:]

    def __call__(self, n: int | None = None, n_default: int = 1) -> bool:
        classifier = self.classifiers.get(
            n or self.n,
            self.classifiers[n_default]
        )
        X = self.getFeatures(as_dict=False)
        y = classifier.predict(X)[0]

        return bool(y)