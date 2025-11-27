
from typing import Callable, List
import numpy as np
from dataclasses import dataclass

from engine_nd.driver import StochasticDriver
from .nd_types import Array, DriftFn, LoadFn, LevyDriver, Integrator, Calibrator, Params, MonteCarlo, SkewFn, CorrelationFn, JumpDriver, LevyFn, Path

@dataclass
class MarkovProcess:
    N: int
    K: int
    a: DriftFn        # Drift functions for each dimension
    B: LoadFn          # Load functions for each dimension
    # change: Array  # Implement if required to perform state changes across more than just a vriable of time
    L: StochasticDriver
    θ: Params
    x0: Array         # Initial state
    S: Array
    path: Path
    t:float
    

    def __post_init__(self):
        self.S = self.x0.copy()
    

    def step(self, integrator: Integrator, dt:float) -> Array:
        dS = integrator.step(self.path, self.t, dt, self.a, self.B, self.L, self.θ)
        self.S += dS
        self.path = np.vstack([self.path, self.S])
        self.t += dt
        return self.S
    

    # def simulate(self,dt: float, n_steps: int, simulator: Simulator):pass


    # def calibrate(self, calibrator: Calibrator, *args, **kwargs) -> Params:
    #     return calibrator.fit(self, *args, **kwargs)