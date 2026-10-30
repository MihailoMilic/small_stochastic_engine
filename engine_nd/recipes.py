
import numpy as np
from .nd_types import Array, Params

def _last_state(path: Array) -> Array:
    arr = np.atleast_2d(np.asarray(path, float))
    return arr[-1]

# -------- GBM (1D) --------
def gbm_a(path: Array, t: float, θ: Params) -> Array:
    x = _last_state(path)
    mu = θ["mu"]
    return np.array([mu * x[0]])

def gbm_B(path: Array, t: float, θ: Params) -> Array:
    x = _last_state(path)
    s = θ["sigma"]
    return np.array([[s * x[0]]])  # N×K = 1×1

# -------- Heston (2D) --------
def heston_a(path: Array, t: float, θ: Params) -> Array:
    x = _last_state(path)
    S, v = x[0], x[1]
    mu, kappa, theta_v = θ["mu"], θ["kappa"], θ["theta_v"]
    return np.array([mu*S, kappa*(theta_v - v)])

def heston_B(path: Array, t: float, θ: Params) -> Array:
    x = _last_state(path)
    S, v = x[0], x[1]
    xi = θ["xi"]
    v_rt = np.sqrt(max(v, 0.0))
    # Assumes Brownian driver already encodes correlation via Cholesky L (K=2)
    return np.array([[S*v_rt, 0.0],
                     [0.0,    xi*v_rt]])

# -------- Merton (log-price) --------
def merton_log_a(path: Array, t: float, θ: Params) -> Array:
    x = _last_state(path)
    # x[0] = log S
    mu, s, λ, mJ, sJ = θ["mu"], θ["sigma"], θ["lam"], θ["mJ"], θ["sJ"]
    # compensate jumps in log domain
    kappa = np.exp(mJ + 0.5*sJ*sJ) - 1.0
    drift_log = mu - 0.5*s*s - λ * np.log(kappa)
    return np.array([drift_log])

def merton_log_B(path: Array, t: float, θ: Params) -> Array:
    # K=2: [Brownian, JumpChannel]
    s = θ["sigma"]
    return np.array([[s, 1.0]])
