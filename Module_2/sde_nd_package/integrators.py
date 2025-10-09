
import numpy as np
from .nd_types import Array, DriftFn, LoadFn, LevyDriver

class EulerIntegrator:
    """Euler–Maruyama integrator; works for diffusion and jump increments provided by the driver."""
    def step(self, x: Array, t: float, dt: float,
             a: DriftFn, B: LoadFn, drv: LevyDriver, θ, rng=None):
        dL = drv.sample(dt, size=1, rng=rng).reshape(-1)       # (K,)
        drift = a(x, t, θ)                                     # (N,)
        load  = B(x, t, θ)                                     # (N,K)
        x_new = x + drift * dt + load @ dL
        return x_new, t + dt

class FullTruncation:
    """Euler with full truncation for nonneg states (e.g., CIR/Heston variance). Supply index list."""
    def __init__(self, nonneg_indices: list[int]):
        self.nonneg_indices = list(nonneg_indices)

    def step(self, x: Array, t: float, dt: float,
             a: DriftFn, B: LoadFn, drv: LevyDriver, θ, rng=None):
        dL = drv.sample(dt, size=1, rng=rng).reshape(-1)
        x_eff = x.copy()
        x_eff[self.nonneg_indices] = np.maximum(x_eff[self.nonneg_indices], 0.0)
        drift = a(x_eff, t, θ)
        load  = B(x_eff, t, θ)
        x_new = x + drift * dt + load @ dL
        x_new[self.nonneg_indices] = np.maximum(x_new[self.nonneg_indices], 0.0)
        return x_new, t + dt
