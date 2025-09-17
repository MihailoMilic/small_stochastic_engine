# sde/model.py
from dataclasses import dataclass
import numpy as np
from .types import Array, Model
from typing import Optional
from .levy import TruncatedLevyIncrement, LevyTriplet
from scipy.linalg import expm
@dataclass(frozen=True)
class GBM(Model):
    mu: float
    sigma: float
    s0: float

    @property
    def dim(self) -> int: return 1

    @property
    def init(self) -> Array: return np.array([self.s0], dtype=float)

    def drift(self, x: Array, t: float) -> Array:
        return self.mu * x

    def diffusion(self, x: Array, t: float) -> Array:
        return self.sigma * x  # scalar diffusion, 1D

    def diffusion_dx(self, x: Array, t: float) -> Array:
        # derivative wrt x of sigma*x is sigma
        return np.full_like(x, self.sigma, dtype=float)
    def jumps(self, x: Array, dt: float) -> Array:
        return np.zeros_like(x, dtype=np.float64)  # no jumps
    def extra_terms(self, x: Array, t: float, dt: float, dW: Array) -> Array:
        return np.zeros_like(x, dtype=np.float64)  # no extra terms

@dataclass(frozen=True)
class MertonJD(Model):
    mu: float
    sigma: float
    lam: float      # jump intensity
    muJ: float      # jump mean
    sigmaJ: float   # jump stddev
    s0: float
    @property
    def dim(self) -> int: return 1
    @property
    def init(self) -> Array: return np.array([self.s0], dtype=float)

    def drift(self, x: Array, t: float) -> Array:
        # risk-neutral drift adjustment for jumps
        k = np.exp(self.muJ + 0.5*self.sigmaJ**2) - 1.0
        return (self.mu - self.lam*k) * x
    def diffusion(self, x: Array, t: float) -> Array:
        return self.sigma * x  # scalar diffusion, 1D
    
    def diffusion_dx(self, x: Array, t: float) -> Array:
        return np.full_like(x, self.sigma, dtype=np.float64)
    
    def jumps(self, x:Array, dt:float) -> Array:
        n_jumps = np.random.poisson(self.lam*dt, size = x.shape)
        if np.any(n_jumps > 0):
            Y = np.random.lognormal(self.muJ, self.sigmaJ, size = x.shape)
            return (Y-1.0) * x * n_jumps
        return np.zeros_like(x, dtype=np.float64)
    def extra_terms(self, x: Array, t: float, dt: float, dW: Array) -> Array:
        return np.zeros_like(x, dtype=np.float64)  # no extra terms


@dataclass(frozen=True)
class AdditiveLevyModel(Model):
    """1D additive Lévy process: dX_t = dL_t with triplet (b,σ,ν)."""
    triplet: LevyTriplet
    increment: TruncatedLevyIncrement
    x0: float

    @property
    def dim(self): return 1
    @property
    def init(self): return np.array([self.x0], float)

    # No state dependence: drift/diffusion are zero; all action is in 'jumps' hook.
    def drift(self, x: Array, t: float) -> Array: return np.zeros_like(x)
    def diffusion(self, x: Array, t: float) -> Array: return np.zeros_like(x)
    def diffusion_dx(self, x: Array, t: float) -> Array: return np.zeros_like(x)

    def jumps(self, x: Array, t: float, dt: float) -> Array:
        # Return ΔL for each x in the batch (shape matches x)
        return self.increment(dt, size=x.shape)

    def extra_terms(self, x: Array, t: float, dt: float) -> Array:
        return np.zeros_like(x)


@dataclass(frozen=True)
class Heston(Model):
    # Risk-neutral Heston:
    # dS = r S dt + sqrt(v) S dW1
    # dv = kappa (theta - v) dt + xi sqrt(v) dW2
    r: float
    kappa: float
    theta: float
    xi: float
    rho: float
    S0: float
    v0: float

    @property
    def dim(self) -> int:
        return 2  # [S, v]

    @property
    def init(self) -> Array:
        return np.array([self.S0, self.v0], dtype=float)

    def drift(self, x: Array, t: float) -> Array:
        # Broadcast-safe for (N,2) or (2,)
        if x.ndim == 1:
            S, v = x[0], x[1]
            return np.array([self.r * S,
                             self.kappa * (self.theta - max(v, 0.0))], dtype=float)
        S = x[:, 0]
        v = np.maximum(x[:, 1], 0.0)
        aS = self.r * S
        av = self.kappa * (self.theta - v)
        return np.stack([aS, av], axis=-1)

    def diffusion(self, x: Array, t: float) -> Array:
        """Diagonal diffusion with two drivers; correlation applied via Simulator.cholesky."""
        if x.ndim == 1:
            S, v = x[0], x[1]
            sv = np.sqrt(max(v, 0.0))
            # shape (2,2): columns are drivers W1, W2
            return np.array([[S * sv, 0.0],
                             [0.0,     self.xi * sv]], dtype=float)
        S = x[:, 0]
        v = np.maximum(x[:, 1], 0.0)
        sv = np.sqrt(v)
        N = x.shape[0]
        B = np.zeros((N, 2, 2), dtype=float)
        B[:, 0, 0] = S * sv           # S driver
        B[:, 1, 1] = self.xi * sv     # v driver
        return B

    # Keep these no-ops so generic schemes won’t fail if called.
    def diffusion_dx(self, x: Array, t: float) -> Array:
        return np.zeros_like(x)

    def jumps(self, x: Array, dt: float) -> Array:
        return np.zeros_like(x)

    def extra_terms(self, x: Array, t: float, dt: float, dW: Array) -> Array:
        return np.zeros_like(x)
    
    @property
    def cholsky(self):
        L = np.array([[1.0, 0.0],
              [self.rho, np.sqrt(1.0 - self.rho**2)]])
        return L
    # For barriers or tight accuracy, reduce dt or add Brownian-bridge logic for crossings on S (you already have a bridge generator in brownian.py for QMC—nice).
# sde/model.py
from dataclasses import dataclass
import numpy as np
from .types import Array, Model

@dataclass(frozen=True)
class SABR(Model):
    # dF = mu * F dt + alpha * F^beta dW1
    # dalpha = nu * alpha dW2
    # corr(dW1, dW2) = rho
    beta: float        # in [0,1] (0: normal/Bachelier, 1: lognormal/Black, in-between: CEV)
    nu: float          # vol-of-vol (ν)
    rho: float         # correlation
    F0: float          # initial forward/spot
    alpha0: float      # initial instantaneous vol
    mu: float = 0.0    # drift for F (0 under forward measure; use r-q if spot)

    @property
    def dim(self) -> int:
        return 2  # [F, alpha]

    @property
    def init(self) -> Array:
        return np.array([self.F0, self.alpha0], dtype=float)

    def drift(self, x: Array, t: float) -> Array:
        # (N,2) or (2,)
        if x.ndim == 1:
            F, a = x[0], x[1]
            return np.array([self.mu * F, 0.0], dtype=float)
        F = x[:, 0]
        a = x[:, 1]
        return np.stack([self.mu * F, np.zeros_like(a)], axis=-1)

    def diffusion(self, x: Array, t: float) -> Array:
        """
        Return (N,2,2) diffusion matrix mapping (dW1, dW2) -> (dF, dalpha).
        Correlation is applied by Simulator.cholesky; we keep this diagonal.
        """
        if x.ndim == 1:
            F, a = x[0], max(x[1], 0.0)
            Fp = max(F, 1e-16)  # guard for beta>0
            return np.array([[a * (Fp**self.beta), 0.0],
                             [0.0,                  self.nu * a]], dtype=float)
        F = x[:, 0]
        a = np.maximum(x[:, 1], 0.0)
        Fp = np.maximum(F, 1e-16)
        N = x.shape[0]
        B = np.zeros((N, 2, 2), dtype=float)
        B[:, 0, 0] = a * (Fp ** self.beta)   # F driver
        B[:, 1, 1] = self.nu * a            # alpha driver
        return B

    def diffusion_dx(self, x: Array, t: float) -> Array:
        # Scalar Milstein doesn’t apply; return zeros to keep generic schemes happy.
        return np.zeros_like(x)

    def jumps(self, x: Array, dt: float) -> Array:
        return np.zeros_like(x)

    def extra_terms(self, x: Array, t: float, dt: float, dW: Array) -> Array:
        return np.zeros_like(x)
    
    @property
    def cholsky(self):
        # Cholesky for (W1, W2)
        L = np.array([[1.0, 0.0],
                [self.rho, np.sqrt(1.0 - self.rho**2)]])
        return L
    

@dataclass
class RegimeSwitchingModel(Model):
    models: list[Model]            
    Q: np.ndarray                
    pi0: np.ndarray               
    dt: float

    # internal state
    _regimes: np.ndarray | None = None
    _t_index: int = 0
    current_regime: np.ndarray | None = None

    @property
    def dim(self) -> int:
        return self.models[0].dim

    @property
    def init(self) -> Array:
        return self.models[0].init

    def prepare_regimes(self, N: int, T: int, rng: np.random.Generator):
        K = len(self.models)
 
        P = expm(self.Q * self.dt)
 
        P = np.clip(P, 0.0, None)
        row_sums = P.sum(axis=1, keepdims=True)
     
        row_sums[row_sums == 0.0] = 1.0
        P = P / row_sums
        self._regimes = np.zeros((T + 1, N), dtype=int)
        self._regimes[0] = rng.choice(K, size=N, p=self.pi0)

        for t in range(1, T + 1):
            for n in range(N):
                i = self._regimes[t - 1, n]
                self._regimes[t, n] = rng.choice(K, p=P[i])

        self._t_index = 0
        self.current_regime = self._regimes[0]

    def advance_regime(self):
        if self._regimes is None:
            raise RuntimeError("Call prepare_regimes before simulation.")
        self._t_index += 1
        self.current_regime = self._regimes[self._t_index]

    def drift(self, x: Array, t: float) -> Array:
        out = np.zeros_like(x)
        for k, m in enumerate(self.models):
            mask = (self.current_regime == k)
            if np.any(mask):
                out[mask] = m.drift(x[mask], t)
        return out

    def diffusion(self, x: Array, t: float) -> Array:
        out = np.zeros_like(x)
        for k, m in enumerate(self.models):
            mask = (self.current_regime == k)
            if np.any(mask):
                out[mask] = m.diffusion(x[mask], t)
        return out

    def diffusion_dx(self, x, t): return np.zeros_like(x)
    def jumps(self, x, dt): return np.zeros_like(x)
    def extra_terms(self, x, t, dt, dW): return np.zeros_like(x)
