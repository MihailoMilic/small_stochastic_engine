
import numpy as np
from .nd_types import Array, Params

def _last_state(path: Array) -> Array:
    arr = np.atleast_2d(np.asarray(path, float))
    return arr[-1]

# -------- GBM (1D) --------
def gbm_a(path: Array, t: float, θ: Params) -> Array:
    x = _last_state(path)
    μ = θ["mu"]
    return np.array([μ * x[0]])

def gbm_B(path: Array, t: float, θ: Params) -> Array:
    x = _last_state(path)
    s = θ["sigma"]
    return np.array([[s * x[0]]])  # N×K = 1×1

# -------- Heston (2D) --------
def heston_a(path: Array, t: float, θ: Params) -> Array:
    x = _last_state(path)
    S, v = x[0], x[1]
    μ, κ, θv = θ["mu"], θ["kappa"], θ["theta_v"]
    return np.array([μ*S, κ*(θv - v)])

def heston_B(path: Array, t: float, θ: Params) -> Array:
    x = _last_state(path)
    S, v = x[0], x[1]
    ξ = θ["xi"]
    v_rt = np.sqrt(max(v, 0.0))
    # Assumes Brownian driver already encodes correlation via Cholesky L (K=2)
    return np.array([[S*v_rt, 0.0],
                     [0.0,    ξ*v_rt]])

# -------- Merton (log-price) --------
def merton_log_a(path: Array, t: float, θ: Params) -> Array:
    x = _last_state(path)
    # x[0] = log S
    μ, s, λ, mJ, sJ = θ["mu"], θ["sigma"], θ["lam"], θ["mJ"], θ["sJ"]
    # compensate jumps in log domain
    kappa = np.exp(mJ + 0.5*sJ*sJ) - 1.0
    drift_log = μ - 0.5*s*s - λ * np.log(1.0 + kappa)
    return np.array([drift_log])

def merton_log_B(path: Array, t: float, θ: Params) -> Array:
    # K=2: [Brownian, JumpChannel]
    s = θ["sigma"]
    return np.array([[s, 1.0]])
