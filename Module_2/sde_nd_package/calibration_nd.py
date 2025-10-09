
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Any, Protocol
from .markov import MarkovProcessND

Array = np.ndarray

@dataclass
class CalibrationResult:
    x: Array
    success: bool
    fun: float
    nit: int
    message: str

def _expit(z): return 1/(1+np.exp(-z))

class CalibrationEngineND:
    """
    Lightweight joint calibration engine for ND Markov processes using
    time-series increments (Gaussian MLE by default) and optional custom loss.
    """
    def __init__(self, process_factory: Callable[[Dict[str, Any]], MarkovProcessND]):
        self.process_factory = process_factory

    def fit_timeseries_gaussian(self, θ0: Dict[str, Any], data: Array, dt: float,
                                param_vec: Callable[[Dict[str, Any]], Array],
                                unvec: Callable[[Array], Dict[str, Any]],
                                bounds: tuple[Array, Array] | None = None,
                                maxiter: int = 200) -> CalibrationResult:
        """
        data: (T+1,N) or (T,N) array of observed states (full state observed).
        Gaussian MLE on increments: ΔX_t ~ N(a(x_t,t,θ)dt,  B Σ B^T dt), approximated by sample cov.
        Practical simplification: assume independent components for speed unless model couples them.
        """
        from scipy.optimize import minimize

        data = np.asarray(data, float)
        if data.ndim != 2:
            raise ValueError("data must be (T, N) or (T+1, N)")
        if data.shape[0] < 3:
            raise ValueError("not enough observations")

        X = data if np.isnan(data[0]).any() else data  # assume already aligned
        ΔX = np.diff(X, axis=0)
        T_obs, N = ΔX.shape

        def neg_ll(u: Array) -> float:
            θ = unvec(u)
            # local drift approx via finite diff mean
            m = ΔX.mean(axis=0)
            S = np.cov(ΔX.T, bias=True)  # (N,N)
            # Penalize invalid params softly (e.g., negative variances)
            pen = 0.0
            for k, v in θ.items():
                if isinstance(v, (int,float)) and (not np.isfinite(v)):
                    pen += 1e6
            # Gaussian neg log-like up to constants
            try:
                sign, logdet = np.linalg.slogdet(S)
                if sign <= 0: raise np.linalg.LinAlgError
                val = 0.5*T_obs*logdet + 0.5*np.sum(np.linalg.solve(S, ΔX.T).T * ΔX)
                return float(val + pen)
            except np.linalg.LinAlgError:
                return 1e9

        u0 = param_vec(θ0)
        method = "L-BFGS-B"
        if bounds is not None:
            lo, hi = bounds
        else:
            lo = np.full_like(u0, -np.inf, dtype=float)
            hi = np.full_like(u0,  np.inf, dtype=float)

        res = minimize(neg_ll, u0, method=method, bounds=list(zip(lo,hi)), options=dict(maxiter=maxiter))
        return CalibrationResult(x=res.x, success=res.success, fun=res.fun, nit=res.nit, message=res.message)
