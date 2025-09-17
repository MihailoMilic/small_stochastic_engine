# sde/simulate.py
from dataclasses import dataclass
import numpy as np
from numpy.random import Generator, PCG64
from .model import RegimeSwitchingModel
from typing import Optional
from .types import Model, Discretization, Array
from .rng import PRNG
@dataclass
class Simulator:
    model: Model
    scheme: Discretization
    dt: float
    num_steps: int
    num_paths: int
    seed: int
    cholesky: Optional[Array] = None  # (dimW, dimW) for correlated Brownian
    # If model.diffusion outputs per-dim, set dimW=model.dim unless model needs >dim drivers.

    def run(self, dW_supplier) -> Array:
        """
        dW_supplier(T, N, dimW) -> (T, N, dimW) Brownian increments with variance dt.
        Returns paths shape (T+1, N, dim).
        """
        dim = self.model.dim
        paths = np.zeros((self.num_steps+1, self.num_paths, dim), dtype=np.float64)
        x = np.broadcast_to(self.model.init, (self.num_paths, dim)).copy()
        if isinstance(self.model, RegimeSwitchingModel):
            self.model.prepare_regimes(self.num_paths, self.num_steps,Generator(PCG64(self.seed)))

        paths[0] = x

        dW = dW_supplier(self.num_steps, self.num_paths)  # (T, N, dimW)
        if self.cholesky is not None:
            # apply correlation per step/path
            dW = np.einsum('ij,tNj->tNi', self.cholesky, dW)

        t = 0.0
        for k in range(self.num_steps):
            # scheme expects x (N, dim) and dW (N, dimW or dim)
            if isinstance(self.model, RegimeSwitchingModel):
                self.model.advance_regime()
            x = self.scheme.step(self.model, x, t, self.dt, dW[k])
            paths[k+1] = x
            t += self.dt
        return paths
