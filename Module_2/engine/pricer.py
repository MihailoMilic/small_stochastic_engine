# sde/pricer.py
from dataclasses import dataclass
import numpy as np
from typing import Optional, Callable
from .types import Array, Model, Discretization, Payoff
from .simulate import Simulator
from .rng import PRNG, SobolNormals
from .brownian import brownian_bridge_increments

Control = Callable[[Array], Array]  # control payoff on same paths

@dataclass
class MonteCarloPricer:
    sim: Simulator
    r: float               # risk-free
    prng: PRNG
    antithetic: bool = False
    control: Optional[Control] = None
    control_expectation: Optional[float] = None  # E[X]

    def _dW_mc(self, T: int, N: int) -> Array:
        dimW = self.sim.model.dim
        if self.antithetic:
            z = self.prng.antithetic_normals((N, T*dimW)).reshape(N, T, dimW)
        else:
            z = self.prng.normal((N, T, dimW))
        return np.transpose(z, (1,0,2)) * np.sqrt(self.sim.dt)  # (T, N, dimW)

    def price(self, payoff: Payoff) -> dict:
        paths = self.sim.run(self._dW_mc)
        Y = payoff(paths)
        if self.control is not None and self.control_expectation is not None:
            X = self.control(paths)
            beta = np.cov(Y, X, ddof=1)[0,1] / np.var(X, ddof=1)
            Y = Y - beta*(X - self.control_expectation)
        disc = np.exp(-self.r * self.sim.num_steps * self.sim.dt)
        val = disc * np.mean(Y)
        err = disc * (np.std(Y, ddof=1) / np.sqrt(self.sim.num_paths))
        return {"price": float(val), "stderr": float(err)}

@dataclass
class QuasiMonteCarloPricer(MonteCarloPricer):
    sobol: Optional[SobolNormals] = None
    use_bridge: bool = True

    def _dW_qmc(self, T: int, N: int) -> Array:
        dimW = self.sim.model.dim
        if self.sobol is None:
            self.sobol = SobolNormals(dim=dimW, scramble=True)
        elif self.sobol.dim != dimW:
            raise ValueError(
                f"SobolNormals dimension changed from {self.sobol.dim} to {dimW}. "
                "Changing dimension mid-run resets the Sobol sequence and reduces QMC effectiveness."
            )
        z = self.sobol.normals(N*T*dimW).reshape(N, T, dimW)
        if self.use_bridge:
            return brownian_bridge_increments(z, T, self.sim.dt)  # (T, N, dimW)
        return np.transpose(z, (1,0,2)) * np.sqrt(self.sim.dt)

    def price(self, payoff: Payoff) -> dict:
        self._dW_mc = self._dW_qmc  # override supplier
        return super().price(payoff)
