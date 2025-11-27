from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Tuple
import numpy as np
from .nd_types import Markov, Calibrator, Array, Params, CalibrationResult



# ======================================================
# Core utilities
# ======================================================

def default_param_vec_unvec(θ0: Params) -> Tuple[Callable[[Params], Array], Callable[[Array], Params]]:
    keys = list(θ0.keys())
    def vec(th: Params) -> Array:
        return np.array([float(th[k]) for k in keys], dtype=float)
    def unvec(u: Array) -> Params:
        out = dict(θ0)
        for k, v in zip(keys, np.asarray(u, float)):
            out[k] = float(v)
        return out
    return vec, unvec


def driver_covariance(drv: Any, dt: float) -> Array:
    K = getattr(drv, "dim", None)
    if K is None and hasattr(drv, "L"):
        K = drv.L.shape[0]
    if hasattr(drv, "L"):
        L = np.asarray(drv.L, float)
        return (L @ L.T) * dt
    if hasattr(drv, "cov"):
        return np.asarray(drv.cov(dt), float)
    if K is None:
        raise ValueError("Cannot infer driver dimension; please supply a driver with `.dim` or `.L`.")
    return np.eye(K) * dt



class GaussianMLECalibrator:
    """Gaussian MLE (diffusion-based) calibrator."""

    def __init__(self,
        process: Markov,
        data: Array,
        dt: float,
        param_vec: Callable[[Params], Array],
        unvec: Callable[[Array], Params],
        bounds: Optional[Tuple[Array, Array]] = None,
        options: Optional[Dict[str, Any]] = None,
        method: str = "L-BFGS-B"):
        self.method = method
        self.process = process
        self.data = data
        self.dt = dt
        self.param_vec = param_vec
        self.unvec = unvec
        self.bounds = bounds
        self.options = options

    def fit( self) -> CalibrationResult:
        from scipy.optimize import minimize

        X = np.asarray(self.data, float)
        if X.ndim != 2:
            raise ValueError("data must be (T, N) array of observed states")
        ΔX = np.diff(X, axis=0)
        T_obs, N = ΔX.shape

        param_vec, unvec = self.param_vec, self.unvec or default_param_vec_unvec(self.process.θ)
        Q = driver_covariance(self.process.L, self.dt)

        def neg_ll(u: Array) -> float:
            θ = unvec(u)
            ll = 0.0
            for t_idx in range(T_obs):
                path_t = X[:t_idx + 1]
                μ_t = self.process.a(path_t, t_idx * self.dt, θ) * self.dt
                B_t = self.process.B(path_t, t_idx * self.dt, θ)
                Σ_t = B_t @ Q @ B_t.T
                r_t = ΔX[t_idx] - μ_t
                Σ_t = 0.5 * (Σ_t + Σ_t.T) + 1e-10 * np.eye(N)
                try:
                    sign, logdet = np.linalg.slogdet(Σ_t)
                    if sign <= 0:
                        raise np.linalg.LinAlgError
                    alpha = np.linalg.solve(Σ_t, r_t)
                    ll += 0.5 * (logdet + r_t @ alpha + N * np.log(2 * np.pi))
                except np.linalg.LinAlgError:
                    return 1e9
            return ll

        u0 = param_vec(self.process.θ)
        res = minimize(neg_ll, u0, method=self.method, bounds=self.bounds, options=self.options)
        θ_hat = unvec(res.x)
        return CalibrationResult(res.x, res.success, float(res.fun), res.nit,
                                 str(getattr(res, "message", "")), θ=θ_hat)


class GMMCalibrator:
    """Generalized Method of Moments calibrator."""

    def __init__(self, 
                process: Markov,
                data: Array,
                dt: float,
                param_vec: Callable[[Params], Array],
                unvec: Callable[[Array], Params],
                bounds: Optional[Tuple[Array, Array]] = None,
                options: Optional[Dict[str, Any]] = None,
                weights: Tuple[float, float] = (1.0, 1.0),
                method: str = "L-BFGS-B"):
        self.method = method
        self.method = method
        self.process = process
        self.data = data
        self.dt = dt
        self.param_vec = param_vec
        self.unvec = unvec
        self.bounds = bounds
        self.options = options
        self.weights=  weights
    def fit(self) -> CalibrationResult:
        from scipy.optimize import minimize

        X = np.asarray(self.data, float)
        ΔX = np.diff(X, axis=0)
        T_obs, N = ΔX.shape
        m_emp = ΔX.mean(axis=0)
        S_emp = np.cov(ΔX.T, bias=True)
        param_vec, unvec = self.param_vec, self.unvec or default_param_vec_unvec(self.process.θ)
        w_m, w_S = self.weights
        Q = driver_covariance(self.process.L, self.dt)

        def gmm_loss(u: Array) -> float:
            θ = unvec(u)
            m_mod = np.zeros(N)
            S_mod = np.zeros((N, N))
            for t_idx in range(T_obs):
                path_t = X[:t_idx + 1]
                m_mod += self.process.a(path_t, t_idx * self.dt, θ) * self.dt
                B_t = self.process.B(path_t, t_idx * self.dt, θ)
                S_mod += B_t @ Q @ B_t.T
            m_mod /= T_obs
            S_mod /= T_obs
            mr = m_emp - m_mod
            Sr = S_emp - S_mod
            return float(w_m * (mr @ mr) + w_S * np.sum(Sr * Sr))

        u0 = param_vec(self.process.θ)
        res = minimize(gmm_loss, u0, method=self.method, bounds=self.bounds, options=self.options)
        θ_hat = unvec(res.x)
        return CalibrationResult(res.x, res.success, float(res.fun), res.nit,
                                 str(getattr(res, "message", "")), θ=θ_hat)


class ParticleFilterMLE:
    """Particle filter-based maximum likelihood calibrator."""

    def __init__(self, 
                 process: Markov,
                data: Array,
                dt: float,
                param_vec: Callable[[Params], Array],
                unvec: Callable[[Array], Params],
                bounds: Optional[Tuple[Array, Array]],
                options: Optional[Dict[str, Any]] = None,
                loglik_y: Callable[[Array, Array, Params], float] = None,
                n_particles: int = 200,
                rng: Optional[np.random.Generator] = None,
                 method: str = "L-BFGS-B"):
        self.method = method
        self.method = method
        self.process = process
        self.data = data
        self.dt = dt
        self.param_vec = param_vec
        self.unvec = unvec
        self.bounds = bounds
        self.options = options
        self.loglik_y = loglik_y
        self.n_particles = n_particles
        self.rng = rng

    @staticmethod
    def _pf_negloglik(
        process: Markov,
        θ: Params,
        y: Array,
        dt: float,
        n_particles: int,
        loglik_y: Callable[[Array, Array, Params], float] = None,
        rng: Optional[np.random.Generator]= None,
    ) -> float:
        rng = rng or np.random.default_rng()
        T_obs = y.shape[0]
        Xp = np.tile(process.x0, (n_particles, 1)) + 1e-6 * rng.normal(size=(n_particles, process.x0.size))
        paths = [np.atleast_2d(process.x0.copy()) for _ in range(n_particles)]
        loglik = 0.0
        for t_idx in range(T_obs):
            current_t = process.t0 + t_idx * dt
            for i in range(n_particles):
                assert process.integrator is not None, "Process integrator is not set"
                next_state, _ = process.integrator.step(paths[i], current_t, dt, process.a, process.B, process.L, θ, rng=rng)
                paths[i] = np.vstack([paths[i], next_state])
                Xp[i] = next_state
            assert loglik_y is not None, "loglik_y callback must be provided"
            lw = np.array([loglik_y(Xp[i], y[t_idx], θ) for i in range(n_particles)])
            m = np.max(lw)
            w = np.exp(lw - m)
            W = w / np.sum(w)
            loglik += m + np.log(np.sum(w)) - np.log(n_particles)
            cdf = np.cumsum(W)
            u0 = rng.random() / n_particles
            us = u0 + np.arange(n_particles) / n_particles
            idx = np.searchsorted(cdf, us, side="right")
            Xp = Xp[idx]
            paths = [paths[j].copy() for j in idx]
        return -loglik

    def fit(
        self,
        
    ) -> CalibrationResult:
        from scipy.optimize import minimize

        if self.loglik_y is None:
            raise ValueError("ParticleFilterMLE requires loglik_y(x, y, θ) callback")
        param_vec, unvec = self.param_vec, self.unvec or default_param_vec_unvec(self.process.θ)

        def obj(u: Array) -> float:
            θ = unvec(u)
            return self._pf_negloglik(self.process, θ, self.data, self.dt, self.n_particles, self.loglik_y, self.rng)

        u0 = param_vec(self.process.θ)
        res = minimize(obj, u0, method=self.method, bounds=self.bounds, options=self.options)
        θ_hat = unvec(res.x)
        return CalibrationResult(res.x, res.success, float(res.fun), res.nit,
                                 str(getattr(res, "message", "")), θ=θ_hat)


class HybridCalibrator:
    """Combine time-series loss with pricing loss via weighting α."""

    def __init__(self, 
                process: Markov,
                data: Array,
                dt: float,
                param_vec: Callable[[Params], Array],
                unvec: Callable[[Array], Params],
                bounds: Optional[Tuple[Array, Array]] = None,
                options: Optional[Dict[str, Any]] = None,
                base_calibrator: Calibrator = None,
                pricing_loss: Callable[[Markov, Params], float] = None,
                alpha: float = 1.0,
                 method: str = "L-BFGS-B"):
        self.method = method
        self.process = process
        self.data = data
        self.dt = dt
        self.param_vec = param_vec
        self.unvec = unvec
        self.bounds = bounds
        self.options = options
        self.base_calibrator = base_calibrator
        self.pricing_loss = pricing_loss
        self.alpha = alpha

    def fit(self) -> CalibrationResult:
        from scipy.optimize import minimize

        base_cal = self.base_calibrator or GaussianMLECalibrator( self.process, self.data, self.dt, self.param_vec, self.unvec, self.bounds, self.options, self.method,)
        if self.pricing_loss is None:
            return base_cal.fit()

        param_vec, unvec = self.param_vec, self.unvec or default_param_vec_unvec(self.process.θ)

        def hybrid_obj(u: Array) -> float:
            θ = unvec(u)
            base_res = base_cal.fit()
            assert self.pricing_loss is not None; "Must provide pricing_loss callback for HybridCalibrator"
            return base_res.fun + self.alpha * float(self.pricing_loss(self.process, θ))

        u0 = param_vec(self.process.θ)
        res = minimize(hybrid_obj, u0, method=self.method, bounds=self.bounds, options=self.options)
        θ_hat = unvec(res.x)
        return CalibrationResult(res.x, res.success, float(res.fun), res.nit,
                                 str(getattr(res, "message", "")), θ=θ_hat)


class OptionSurfaceCalibrator:
    """Calibration based purely on option/pricing loss."""

    def __init__(self, 
                 process: Markov,
                data: Array,
                dt: float,
                param_vec: Callable[[Params], Array],
                unvec: Callable[[Array], Params],
                bounds: Optional[Tuple[Array, Array]] = None,
                options: Optional[Dict[str, Any]] = None,
                pricing_loss: Callable[[Markov, Params], float] = None,
                 method: str = "L-BFGS-B"):
        self.method = method
        self.process = process
        self.data = data
        self.dt = dt
        self.param_vec = param_vec
        self.unvec = unvec
        self.bounds = bounds
        self.options = options
        self.pricing_loss = pricing_loss

    def fit(self,) -> CalibrationResult:
        from scipy.optimize import minimize

        param_vec, unvec = self.param_vec, self.unvec or default_param_vec_unvec(self.process.θ)

        def obj(u: Array) -> float:
            θ = unvec(u)
            assert self.pricing_loss is not None; "Must provide pricing_loss callback for OptionSurfaceCalibrator"
            return float(self.pricing_loss(self.process, θ))

        u0 = param_vec(self.process.θ)
        res = minimize(obj, u0, method=self.method, bounds=self.bounds, options=self.options)
        θ_hat = unvec(res.x)
        return CalibrationResult(res.x, res.success, float(res.fun), res.nit,
                                 str(getattr(res, "message", "")), θ=θ_hat)
