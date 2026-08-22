from typing import Protocol, Callable, Dict, Any, Tuple, Optional, Sequence, List
import numpy as np
from dataclasses import dataclass, field
from engine_nd.driver import StochasticDriver

Array = np.ndarray
Params = Dict[str, Any]
Path = Array  # (M, N) array of realized path up to time t

DriftFn = Callable[[Path, float, Params], Array]      # (path, t, θ) -> R^N
LoadFn  = Callable[[Path, float, Params], Array]      # (path, t, θ) -> R^{N×K}
LevyFn  = Callable[[Path, float, float, Params, int, Optional[np.random.Generator]], Array]
CorrelationFn = Callable[[Path, float, Params], Array]
SkewFn        = Callable[[Path, float, Params], Array]


@dataclass
class Cholsky:
    """Lower-triangular Cholesky factor for correlated Brownian drivers.

    Either static (supply ``matrix``) or dynamic (supply ``function``).
    If ``matrix`` is omitted, defaults to the identity of size ``dim``.
    """
    dim: int
    matrix: Optional[Array] = field(default=None)
    static: bool = True
    function: Optional[CorrelationFn] = None

    def __post_init__(self):
        if self.matrix is None:
            self.matrix = np.eye(self.dim)
        L = np.asarray(self.matrix, dtype=float)
        self.matrix = L
        assert L.ndim == 2 and L.shape[0] == L.shape[1], "L must be square"
        assert np.allclose(L, np.tril(L)), "L must be lower-triangular"
        assert np.all(np.diag(L) > 0.0), "L must have positive diagonal"

    def get(self, path: Path, t: float, θ: Params) -> Array:
        if self.static:
            return self.matrix
        if self.function is not None:
            return self.function(path, t, θ)
        raise ValueError("Cholsky: not static and no function provided.")


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
    ) -> Array: ...

    def from_standard(
        self,
        Z: np.ndarray,
        dt: float,
        path: Path | None = None,
        t: float | None = None,
        θ: Params | None = None,
    ) -> np.ndarray: ...

    @property
    def L(self) -> Cholsky: ...

    @L.setter
    def L(self, L: Cholsky): ...


JumpDriver = LevyDriver


class Integrator(Protocol):
    def step(
        self,
        path: Path,
        t: float,
        dt: float,
        a: DriftFn,
        B: LoadFn,
        L: StochasticDriver,
        θ: Params,
        rng: np.random.Generator | None = None,
    ) -> tuple[Array, float]: ...


class Markov(Protocol):
    a: DriftFn
    B: LoadFn
    L: StochasticDriver
    θ: Params
    x0: Array
    t0: float
    integrator: Integrator | None


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
    def fit(self, *kwargs) -> "CalibrationResult": ...


class MonteCarlo(Protocol):
    def simulate(
        self,
        process: Markov,
        integrator: Integrator,
        n_paths: int,
        n_steps: int,
        dt: float,
        seed: int,
        **kwargs,
    ) -> Array: ...
