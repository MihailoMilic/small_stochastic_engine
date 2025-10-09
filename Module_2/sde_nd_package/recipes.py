
import numpy as np
from .nd_types import Array, Params

# -------- GBM (1D) --------
def gbm_a(x: Array, t: float, θ: Params) -> Array:
    μ = θ["mu"]
    return np.array([μ * x[0]])

def gbm_B(x: Array, t: float, θ: Params) -> Array:
    s = θ["sigma"]
    return np.array([[s * x[0]]])  # N×K = 1×1

# -------- Heston (2D) --------
def heston_a(x: Array, t: float, θ: Params) -> Array:
    S, v = x
    μ, κ, θv = θ["mu"], θ["kappa"], θ["theta_v"]
    return np.array([μ*S, κ*(θv - v)])

def heston_B(x: Array, t: float, θ: Params) -> Array:
    S, v = x
    ξ = θ["xi"]
    v_rt = np.sqrt(max(v, 0.0))
    # Assumes Brownian driver already encodes correlation via Cholesky L (K=2)
    return np.array([[S*v_rt, 0.0],
                     [0.0,    ξ*v_rt]])

# -------- Merton (log-price) --------
def merton_log_a(x: Array, t: float, θ: Params) -> Array:
    # x[0] = log S
    μ, s, λ, mJ, sJ = θ["mu"], θ["sigma"], θ["lam"], θ["mJ"], θ["sJ"]
    # compensate jumps in log domain
    kappa = np.exp(mJ + 0.5*sJ*sJ) - 1.0
    drift_log = μ - 0.5*s*s - λ * np.log(1.0 + kappa)
    return np.array([drift_log])

def merton_log_B(x: Array, t: float, θ: Params) -> Array:
    # K=2: [Brownian, JumpChannel]
    s = θ["sigma"]
    return np.array([[s, 1.0]])
