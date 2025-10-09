
import numpy as np
from dataclasses import dataclass
from .nd_types import Array, DriftFn, LoadFn, LevyDriver, Integrator, Calibrator

@dataclass
class MarkovProcessND:
    N: int
    K: int
    a: DriftFn
    B: LoadFn
    levy_driver: LevyDriver
    θ: dict
    x0: Array
    t0: float = 0.0
    integrator: Integrator | None = None

    def __post_init__(self):
        if self.integrator is None:
            from .integrators import EulerIntegrator
            self.integrator = EulerIntegrator()
        self.x0 = np.asarray(self.x0, float).reshape(self.N)
        if self.K != self.levy_driver.dim:
            raise ValueError("K mismatch with driver dimension")

    def simulate(self, T: float, dt: float, n_paths: int = 1, rng=None):
        M = int(np.ceil(T/dt))
        t = np.linspace(self.t0, self.t0 + M*dt, M+1)
        X = np.empty((n_paths, M+1, self.N), dtype=float)
        for p in range(n_paths):
            x = self.x0.copy()
            X[p,0] = x
            tp = self.t0
            for m in range(M):
                if self.integrator is None:
                    raise ValueError("Integrator not set")
                x, tp = self.integrator.step(x, tp, dt, self.a, self.B, self.levy_driver, self.θ, rng=rng)
                X[p,m+1] = x
        return {"t": t, "X": X}
    
    def fit(self, calibrator:Calibrator, data:Array, **kwargs):
        return calibrator.fit(self, data, **kwargs)

