from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from engine_nd.driver import StochasticDriver
from .nd_types import Array, DriftFn, LoadFn, Integrator, Params, Path


@dataclass
class MarkovProcess:
    """N-dimensional Markov process driven by a StochasticDriver.

    Fields
    ------
    N : state dimension
    K : driver dimension (number of noise channels)
    a : drift function  (path, t, θ) -> R^N
    B : loading matrix  (path, t, θ) -> R^{N×K}
    L : stochastic driver producing K-dimensional increments
    θ : parameter dict
    x0 : initial state  R^N
    t0 : initial time (also acts as current time — advanced by step())
    integrator : default integrator used by step(); may be overridden per call
    """
    N: int
    K: int
    a: DriftFn
    B: LoadFn
    L: StochasticDriver
    θ: Params
    x0: Array
    t0: float
    integrator: Optional[Integrator] = None

    # Internal mutable state — not constructor arguments
    S: Array = field(init=False, repr=False)
    path: Path = field(init=False, repr=False)

    def __post_init__(self):
        self.S = self.x0.copy()
        self.path = np.atleast_2d(self.x0.copy())

    def step(self, dt: float, integrator: Optional[Integrator] = None) -> Array:
        """Advance the process by one time step of size ``dt``.

        Returns the new state vector.
        """
        intg = integrator or self.integrator
        if intg is None:
            raise ValueError("Provide an integrator either at construction or per-call.")
        x_next, _ = intg.step(self.path, self.t0, dt, self.a, self.B, self.L, self.θ)
        self.S = x_next
        self.path = np.vstack([self.path, self.S])
        self.t0 += dt
        return self.S
