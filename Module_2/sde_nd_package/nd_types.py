
from typing import Protocol, Callable, Dict, Any
import numpy as np
from sde_nd_package.markov import MarkovProcessND
from sde_nd_package.calibration_nd import CalibrationResult
Array = np.ndarray
Params = Dict[str, Any]

# Drift & loading signatures
DriftFn = Callable[[Array, float, Params], Array]      # (x,t,θ) -> R^N
LoadFn  = Callable[[Array, float, Params], Array]      # (x,t,θ) -> R^{N×K}

class LevyDriver(Protocol):
    dim: int
    def sample(self, dt: float, size: int = 1, rng: np.random.Generator | None = None) -> Array: ...

class Integrator(Protocol):
    def step(self, x: Array, t: float, dt: float,
             a: DriftFn, B: LoadFn, drv: LevyDriver,
             θ: Params, rng: np.random.Generator | None = None) -> tuple[Array, float]: ...
class Calibrator(Protocol):
    def fit(self, process:MarkovProcessND, data:Array, dt, **kwargs) -> CalibrationResult: ...