# levy.py
from dataclasses import dataclass
import numpy as np
from typing import Callable, Protocol, Optional, Tuple
from scipy.stats import norm
from math import erf, sqrt
Array = np.ndarray

class LevyIncrement(Protocol):
    def __call__(self, dt: float, size: Tuple[int, ...]) -> Array:...
    """Sample Lévy increments ΔX over time step dt; returns array with given size."""

@dataclass(frozen=True)
class LevyTriplet:
    drift: float                
    sigma: float                 
    # ν is represented operationally via samplers / integrals below

class LevyMeasure(Protocol):
    def large_jump_rate(self, eps: float) -> float: ...
    def sample_large_jumps(self, n: int) -> Array: ...
    def small_jump_var(self, eps: float) -> float: ...


@dataclass(frozen=True)
# For production, implement exact truncated normal moments & conditional sampling; under is a readable sketch.
class GaussianJumpMeasure(LevyMeasure):
    lam: float
    muJ: float
    sigJ: float
    def large_jump_rate(self, eps: float)->float:

        p_small = norm.cdf((eps - self.muJ)/self.sigJ) - norm.cdf((-eps-self.muJ)/self.sigJ)
        return float(self.lam * (1.0 - p_small))
    def sample_large_jumps(self, n:int)-> np.ndarray:
        return np.random.normal(self.muJ, self.sigJ, size = (n, ))
    def small_jump_var(self, eps: float) -> float:
        grid = np.linspace(-eps, eps, 4096)
        pdf = norm.pdf((grid - self.muJ)/self.sigJ) / self.sigJ
        # Use np.trapz for broad NumPy compatibility
        m2 = np.trapezoid(grid**2 * pdf, grid)
        return float(self.lam * m2)

# levy.py (cont.)
@dataclass
class TruncatedLevyIncrement:
    triplet: LevyTriplet
    measure: LevyMeasure
    eps: float = 1e-3 

    def __call__(self, dt: float, size: Tuple[int, ...]) -> Array:
        # Brownian part
        dW = np.random.normal(0.0, np.sqrt(dt), size=size)
        cont = self.triplet.drift * dt + self.triplet.sigma * dW


        v_eps = self.measure.small_jump_var(self.eps)     
        small = np.random.normal(0.0, np.sqrt(v_eps * dt), size=size)


        lam = self.measure.large_jump_rate(self.eps)       
      
        N = np.random.poisson(lam * dt, size=size).astype(int)

        out = cont + small
        flat = out.reshape(-1)
        N_flat = N.reshape(-1)

        # Vectorized accumulation of large jumps:
        # build repeated indices per element count and add sampled jumps in one go
        total_jumps = int(N_flat.sum())
        if total_jumps > 0:
            # indices repeated by number of jumps per entry
            idx_rep = np.repeat(np.arange(N_flat.size, dtype=int), N_flat)
            J = self.measure.sample_large_jumps(total_jumps)
            # accumulate sums into flat using in-place indexed addition
            np.add.at(flat, idx_rep, J)
        return flat.reshape(size)
