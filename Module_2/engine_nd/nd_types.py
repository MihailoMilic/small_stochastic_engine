
from typing import Protocol, Callable, Dict, Any, Tuple, Optional, Sequence, List
import numpy as np
from dataclasses import dataclass
from engine_nd.driver import StochasticDriver
# Note: avoid importing CalibrationResult here to prevent circular import.
Array = np.ndarray
Params = Dict[str, Any]
Path = Array  # convenience alias; interpreted as the full path up to time t

# Drift & loading signatures now accept the realized path rather than a single state.
DriftFn = Callable[[Path, float, Params], Array]      # (path,t,θ) -> R^N
LoadFn  = Callable[[Path, float, Params], Array]      # (path,t,θ) -> R^{N×K}
LevyFn = Callable[[Path, float, float, Params, int, Optional[np.random.Generator]], Array]
# Optional model callbacks (kept loose for flexibility)
CorrelationFn = Callable[[Path, float, Params], Array]  # expected to return Cholesky L (K×K) or correlation
SkewFn        = Callable[[Path, float, Params], Array]  # user-defined higher-moment control

@dataclass
class Cholsky:
    dim : int = 0
    matrix: Array = np.diag(np.ones(dim))
    static: bool = True
    function: Optional[CorrelationFn] = None
    def __post_init__(self):
        L = self.matrix
        assert L.ndim == 2 and L.shape[0] == L.shape[1], "L must be square"
        # Require lower-triangular with positive diagonal (Cholesky factor)
        assert np.allclose(L, np.tril(L)), "L must be lower-triangular"
        assert np.all(np.diag(L) > 0.0), "L must have positive diagonal"
    def get(self, path: Path, t: float, θ: Params) -> Array:
        if self.static:
            return self.matrix
        elif self.function is not None:
            matrix = self.function(path, t, θ)
            return matrix 
        else:
            raise ValueError("Cholsky factor is not static and no function provided.")
    
class LevyDriver(Protocol):
    dim: int
    antithetic: bool

    def sample(
        self,
        path: Path,
        t: float,
        dt: float,
        θ: Params | None = None,
        size: int = 1,
        rng: np.random.Generator | None = None,
    ) -> Array:
        ...

    def from_standard(
        self,
        Z: np.ndarray,
        dt: float,
        path: Path | None = None,
        t: float | None = None,
        θ: Params | None = None,
    ) -> np.ndarray:
        ...

    @property
    def L(self) -> Cholsky: ...

    @L.setter
    def L(self, L: Cholsky): ...
    # Note: Some drivers (e.g., Brownian) may expose an optional L property; do not require it here.

# Alias for potential separate jump driver typing
JumpDriver = LevyDriver
class Integrator(Protocol):
    def step(
        self,
        path: Path,
        t: float,
        dt: float,
        a: DriftFn,
        B: LoadFn,
        L:  StochasticDriver,
        θ: Params,
        rng: np.random.Generator | None = None,
    ) -> tuple[Array, float]: ...
class Markov(Protocol):
    a : DriftFn
    B : LoadFn
    L: StochasticDriver
    θ : Params
    x0 : Array
    t0 : float
    integrator : Integrator | None = None

@dataclass
class CalibrationResult:
    x: Array
    success: bool
    fun: float
    nit: int
    message: str
    θ: Optional[Params] = None
    extra: Optional[Dict[str, Any]] = None


class Calibrator(Protocol):
    def fit(self, *kwargs) -> "CalibrationResult":
        ...

class MonteCarlo(Protocol):
    def simulate(self, process: Markov,integrator:Integrator,  n_paths: int, n_steps: int, dt: float,seed:int , **kwargs) -> Array:
        ...
