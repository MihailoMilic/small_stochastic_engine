# sde/rng.py
from dataclasses import dataclass
import numpy as np
from scipy.stats import qmc, norm
from typing import Optional
import math

@dataclass(frozen=True)
class PRNG:
    """Deterministic RNG wrapper, vectorized draws, reproducible streams."""
    seed: Optional[int] = None

    def normal(self, shape, mean=0.0, std=1.0) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.normal(loc=mean, scale=std, size=shape)

    def antithetic_normals(self, shape, mean=0.0, std=1.0) -> np.ndarray:
        """Return Z and -Z stacked along axis 1 if shape=(N, d)."""
        rng = np.random.default_rng(self.seed)
        halfN = (shape[0] + 1)//2
        z = rng.normal(loc=mean, scale=std, size=(halfN, *shape[1:]))
        z_full = np.concatenate([z, -z[:shape[0]-halfN]], axis=0)
        return z_full

class SobolNormals:
    """Sobol -> N(0,1) with optional Owen scrambling."""
    def __init__(self, dim: int, scramble: bool = True, ):
        self.engine = qmc.Sobol(d=dim, scramble=scramble)
        self.dim = dim

    def normals(self, n: int) -> np.ndarray:
        m = math.ceil(math.log2(n))
        u = self.engine.random_base2(m)  
        u = u[:n]                       
        z = norm.ppf(u)                  
        # guard against inf at 0/1:
        z = np.clip(z, -10.0, 10.0)
        return z
