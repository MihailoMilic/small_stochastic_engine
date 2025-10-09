# sde/model.py
from dataclasses import dataclass, replace
import numpy as np

from engine.calibration import MLECalibrator, SurfaceCalibrator, MLEProtocol, SurfaceProtocol
from engine.types import Array, Model
from typing import Any, Dict, Optional, TYPE_CHECKING, Tuple, Callable
if TYPE_CHECKING:
    from engine.pricer import PricerProtocol
from engine.levy import TruncatedLevyIncrement, LevyTriplet
from scipy.linalg import expm
from engine.types import DriftFn, DiffusionFn
from scipy.stats import norm
from scipy.special import factorial
import pandas as pd

@dataclass(frozen=True)
class GBM(Model, MLEProtocol):
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
    def loglike(self, theta, returns, dt):
        mu, sigma = theta["mu"], theta["sigma"]
        mean = mu * dt
        var = sigma**2 * dt
        if var <=0:
            return -np.inf
        resid = returns - mean
        return -0.5*(len(returns)*np.log(2*np.pi*var) + np.sum(resid**2)/var)
    def _default_calibrator(self):
        return MLECalibrator()


@dataclass(frozen=True)
class MertonJD(Model, MLEProtocol, SurfaceProtocol):
    mu: float
    sigma: float
    lam: float      # jump intensity
    muJ: float      # jump mean
    sigmaJ: float   # jump stddev
    s0: float
    max_jumps: int = 10  # truncate Poisson at k=max_jumps for likelihood
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
    def loglike(self, theta, returns, dt):
        mu, sigma, lam, m, delta = (theta[k] for k in ["mu","sigma","lam","muJ","sigmaJ"])
        ll = 0.0
        for r in returns:
            # mixture over small k (truncated Poisson)
            mix = 0.0
            for k in range(0, self.max_jumps):  # truncate at k=max_jumps
                weight = np.exp(-lam*dt) * (lam*dt)**k / factorial(k)
                mean = mu*dt + k*m
                var  = sigma**2*dt + k*delta**2
                mix += weight * (1/np.sqrt(2*np.pi*var)) * np.exp(-(r-mean)**2/(2*var))
            ll += np.log(mix + 1e-12)  # numerical stability
        return ll
    def charfunc(self, u, T, S0, r=0.0):
        """Characteristic function of ln S_T for Merton jump-diffusion.
        Supports scalar or vector `u` inputs.
        """
        u = np.asarray(u, dtype=np.complex128)
        mu, sigma, lam, m, delta = self.mu, self.sigma, self.lam, self.muJ, self.sigmaJ
        k = np.exp(m + 0.5 * delta**2) - 1.0  # E[e^J] - 1, J ~ N(m, delta^2)
        drift = r - lam * k - 0.5 * sigma**2
        return np.exp(
            1j * u * (np.log(S0) + drift * T)
            - 0.5 * sigma**2 * u**2 * T
            + lam * T * (np.exp(1j * u * m - 0.5 * delta**2 * u**2) - 1.0)
        )
    def extra_terms(self, x: Array, t: float, dt: float, dW: Array) -> Array:
        return np.zeros_like(x, dtype=np.float64)  # no extra terms
       # === Option Pricing (Carr–Madan) ===

    def option_prices(self, theta, strikes, maturities, S0, r=0.0, pricer: Optional["PricerProtocol"] = None) -> np.ndarray:
        prices = []
        for K, T in zip(strikes, maturities):
            if pricer is not None:
                price, _ = pricer.price(self, K, T, S0, r)
                prices.append(price)
            else:
                from engine.pricer import CarrMadanPricer
                cm_pricer = CarrMadanPricer()
                price, _ = cm_pricer.price(self, K, T, S0, r)
                prices.append(price)
        return np.array(prices)
    def _default_calibrator(self):
        return MLECalibrator()



@dataclass(frozen=True)
class Heston(Model, SurfaceProtocol):
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
    # === Characteristic Function of log-price ===
    def charfunc(self, u: complex, T: float, S0: float, r: float = 0.0) -> complex:
        kappa, theta, xi, rho, v0 = self.kappa, self.theta, self.xi, self.rho, self.v0

        d = np.sqrt((rho * xi * 1j * u - kappa)**2 + (xi**2) * (1j * u + u**2))
        g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)

        C = (r * 1j * u * T +
             (kappa * theta / (xi**2)) *
             ((kappa - rho * xi * 1j * u - d) * T -
              2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))))
        D = ((kappa - rho * xi * 1j * u - d) / (xi**2)) * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))

        return np.exp(C + D * v0 + 1j * u * np.log(S0))

    def _default_calibrator(self):
        # Use a derivative-free method for robustness on nonconvex SABR MSE surfaces
        return SurfaceCalibrator(method="Powell")

    # === Required for SurfaceInterface ===
    def option_prices(self, theta: dict, strikes: np.ndarray, maturities: np.ndarray,
                      S0: float, r: float = 0.0, pricer: Optional["PricerProtocol"] = None) -> np.ndarray:
        # replace self with new params for consistency
        model = replace(self, **theta)
        prices = []
        for K, T in zip(strikes, maturities):
            if pricer is not None:
                price, _ = pricer.price(model, K, T, S0, r)
                prices.append(price)
            else:
                from engine.pricer import CarrMadanPricer
                cm_pricer = CarrMadanPricer()
                price, _ = cm_pricer.price(model, K, T, S0, r)
                prices.append(price)
        return np.array(prices)
    # For barriers or tight accuracy, reduce dt or add Brownian-bridge logic for crossings on S (you already have a bridge generator in brownian.py for QMC—nice).
# sde/model.py

@dataclass(frozen=True)
class SABR(Model, SurfaceProtocol):
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
    # --- Hagan et al. (2002) lognormal implied vol approximation ---
    def implied_vol(self, F: float, K: float, T: float) -> float:
        # handle ATM limit robustly
        if F <= 0 or K <= 0 or T <= 0:
            return np.nan
        a, b, r, n = self.alpha0, self.beta, self.rho, self.nu
        if a <= 0 or n <= 0 or not (-0.999 < r < 0.999) or not (0.0 <= b <= 1.0):
            return np.nan

        logFK = np.log(F/K)
        # power mean factor
        FK_beta = (F * K) ** ((1.0 - b) / 2.0)

        # z and x(z)
        z = (n / a) * FK_beta * logFK if abs(logFK) > 1e-12 else 0.0
        # x(z) as in Hagan
        if abs(z) < 1e-12:
            x_z = 1.0
        else:
            tmp = np.sqrt(1.0 - 2.0 * r * z + z * z)
            x_z = np.log((tmp + z - r) / (1.0 - r))

        # Denominator correction in log-strike
        log_corr = 1.0 + ((1 - b) ** 2 / 24.0) * (logFK ** 2) + ((1 - b) ** 4 / 1920.0) * (logFK ** 4)

        # Time-dependent correction
        A = ((1 - b) ** 2 / 24.0) * (a * a) / (FK_beta ** 2)
        B = (r * b * n * a) / (4.0 * FK_beta)
        C = ((2.0 - 3.0 * r * r) / 24.0) * (n * n)
        time_corr = 1.0 + (A + B + C) * T

        # Final implied vol
        vol = (a / (FK_beta * log_corr)) * (z / x_z if x_z != 0 else 1.0) * time_corr
        return max(vol, 1e-12)

    # --- Vectorized surface pricing hook for SurfaceCalibrator ---
    def option_prices(
        self,
        theta: dict[str, float],
        strikes: np.ndarray,
        maturities: np.ndarray,
        s0: float,
        r: float = 0.0,
        call_put: np.ndarray | None = None,  # +1 for calls, -1 for puts; if None assume calls
        pricer: Optional["PricerProtocol"] = None
    ) -> np.ndarray:
        model = replace(self, **theta) if theta else self
        strikes = np.asarray(strikes, dtype=float)
        maturities = np.asarray(maturities, dtype=float)
        if call_put is None:
            call_put = np.ones_like(strikes)
        call_put = np.asarray(call_put, dtype=float)

        prices = np.empty_like(strikes, dtype=float)
        for i, (K, T, cp) in enumerate(zip(strikes, maturities, call_put)):
            df = np.exp(-r * T)
            F = s0 * np.exp(r * T)  # forward (no carry)
            # Option price using provided pricer or Black with model's implied vol
            if pricer is not None:
                call, _ = pricer.price(model, K, T, s0, r)
            else:
                from engine.pricer import BlackPricer
                black_pricer = BlackPricer()
                call, _ = black_pricer.price(model, K, T, s0, r)
            if cp >= 0:
                prices[i] = call
            else:
                # put from call via Black put-call parity on forwards
                prices[i] = call - df * (F - K)
        return prices
    def _default_calibrator(self):
        # Derivative-free is often more robust for SABR surface MSE
        return SurfaceCalibrator(method="Powell")

@dataclass
class RegimeSwitchingModel:
    models: list[Model]            
    Q: np.ndarray                
    pi0: np.ndarray               
    dt: float
    K: int = 2                 # number of regimes (models)

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

        # Transition matrix and normalization
        P = expm(self.Q * self.dt)
        P = np.clip(P, 0.0, None)
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        P = P / row_sums
        Pc = np.cumsum(P, axis=1)  # cumulative probabilities per current state

        # Allocate regime path matrix
        self._regimes = np.zeros((T + 1, N), dtype=int)
        # Initial regimes from initial distribution
        self._regimes[0] = rng.choice(K, size=N, p=self.pi0)

        # Vectorized per-step categorical sampling using cumulative probabilities
        for t in range(1, T + 1):
            prev = self._regimes[t - 1]                 # shape (N,)
            U = rng.random(N)                           # shape (N,)
            # Select cumulative row for each path's current state
            cum_rows = Pc[prev]                         # shape (N, K)
            # Next state = number of entries strictly less than U
            next_states = (cum_rows < U[:, None]).sum(axis=1).astype(int)
            self._regimes[t] = next_states

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
    def n_states(self) -> int:
        return self.K

    def param_names(self) -> Tuple[str, ...]:
        K = self.K
        names = [f"mu{i+1}" for i in range(K)] + [f"log_sigma{i+1}" for i in range(K)]
        for i in range(K):
            for j in range(K):
                names.append(f"logit_P{i+1}{j+1}")
        return tuple(names)

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = x - np.max(x, axis=axis, keepdims=True)
        expx = np.exp(x)
        return expx / np.sum(expx, axis=axis, keepdims=True)

    def unpack_params(self, theta: Array) -> Dict[str, Any]:
        theta = np.asarray(theta, dtype=float)
        K = self.K
        expected_len = 2*K + K*K
        if theta.shape[0] != expected_len:
            raise ValueError(f"Expected {expected_len} parameters, got {theta.shape[0]}")
        mu = theta[:K]
        sigma = np.exp(theta[K:2*K])
        logits = theta[2*K:].reshape(K, K)
        P = self._softmax(logits, axis=1)
        return {"mu": mu, "sigma": sigma, "P": P}

    def stationary_distribution(self, P: np.ndarray) -> np.ndarray:
        w, v = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(w - 1.0))
        pi = np.real(v[:, idx])
        pi = np.maximum(pi, 0.0)
        pi /= np.sum(pi)
        return pi

    @staticmethod
    def _log_normal_pdf(x: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        X = x[:, None]
        M = mean[None, :]
        S = sigma[None, :]
        return -0.5*((X-M)**2)/(S**2) - np.log(S) - 0.5*np.log(2*np.pi)

    def log_likelihood(self, returns: np.ndarray, theta: Array, pi0: Optional[np.ndarray]=None) -> float:
        pars = self.unpack_params(theta)
        mu, sigma, P = pars['mu'], pars['sigma'], pars['P']
        K = self.K
        T = len(returns)
        if pi0 is None:
            pi = self.stationary_distribution(P)
        else:
            pi = np.maximum(pi0,1e-300); pi/=np.sum(pi)
        logB = self._log_normal_pdf(returns, mu, sigma)
        logP = np.log(np.maximum(P,1e-300))
        log_pi = np.log(np.maximum(pi,1e-300))

        def logsumexp(a, axis=None):
            m=np.max(a,axis=axis,keepdims=True)
            return np.squeeze(np.log(np.sum(np.exp(a-m),axis=axis,keepdims=True))+m,axis=axis)

        alpha=np.empty((T,K))
        alpha[0,:]=log_pi+logB[0,:]
        for t in range(1,T):
            temp=alpha[t-1,:][:,None]+logP
            alpha[t,:]=logB[t,:]+logsumexp(temp,axis=0)
        ll=logsumexp(alpha[-1,:])
        return float(ll) if np.isfinite(ll) else -1e20




