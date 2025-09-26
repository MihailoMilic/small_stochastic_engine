# sde/pricer.py
from dataclasses import dataclass
import numpy as np
from typing import Optional, Callable, Protocol, Tuple
from .types import Array, Model, Discretization, Payoff
from .simulate import Simulator
from .rng import PRNG, SobolNormals
from .brownian import brownian_bridge_increments
from scipy.stats import norm
Control = Callable[[Array], Array]  # control payoff on same paths
class PricerProtocol(Protocol):
    def price(self, model: Model, K: float, T: float, S0: float, r: float) -> Tuple[float, float]:
        raise NotImplementedError("Each model must implement price method")

class CharFuncProtocol(Protocol):
    def charfunc(self, u: complex, T: float, S0: float, r: float = 0.0) -> complex:
        raise NotImplementedError("Each model must implement charfunc")
class ImpliedVolProtocol(Protocol):
    def implied_vol(self, F: float, K: float, T: float) -> float:
        raise NotImplementedError("Each model must implement implied_vol method")
@dataclass
class CarrMadanPricer(PricerProtocol):
    alpha: float = 1.5
    N: int = 256
    B: float = 500.0

    def price(self, model:CharFuncProtocol, K, T, S0, r=0.0):
        u = np.arange(self.N) * np.pi / self.B
        phi = lambda u: model.charfunc(u - (self.alpha+1)*1j, T, S0, r) / \
                        (self.alpha**2 + self.alpha - u**2 + 1j*(2*self.alpha+1)*u)
        integrand = np.exp(-1j*u*np.log(K)) * phi(u) * (np.pi/self.B)
        price = np.exp(-self.alpha*np.log(K)) / np.pi * np.sum(np.real(integrand))
        return float(np.exp(-r*T)*price), 0.0  # price, error (error not estimated)

@dataclass
class BlackPricer:
    def price(self, model:ImpliedVolProtocol, K, T, S0, r=0.0, call=True) -> tuple[float, float]:
        F = S0*np.exp(r*T)
        iv = model.implied_vol(F, K, T)
        if iv <= 0 or T <= 0:
            return max((F-K) if call else (K-F), 0.0) * np.exp(-r*T), 0.0
        sig = iv*np.sqrt(T)
        d1 = (np.log(F/K) + 0.5*sig*sig)/sig
        d2 = d1 - sig
        call_price = np.exp(-r*T)*(F*norm.cdf(d1) - K*norm.cdf(d2))
        price = call_price if call else call_price - np.exp(-r*T)*(F-K)
        return price, 0.0
@dataclass
class MonteCarloPricer(PricerProtocol):
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

    def price(self, payoff: Payoff) -> Tuple[float, float]:
        paths = self.sim.run(self._dW_mc)
        Y = payoff(paths)
        if self.control is not None and self.control_expectation is not None:
            X = self.control(paths)
            beta = np.cov(Y, X, ddof=1)[0,1] / np.var(X, ddof=1)
            Y = Y - beta*(X - self.control_expectation)
        disc = np.exp(-self.r * self.sim.num_steps * self.sim.dt)
        val = disc * np.mean(Y)
        err = disc * (np.std(Y, ddof=1) / np.sqrt(self.sim.num_paths))
        return float(val), float(err)

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

    def price(self, payoff: Payoff) -> Tuple[float, float]:
        self._dW_mc = self._dW_qmc  # override supplier
        return super().price(payoff)
