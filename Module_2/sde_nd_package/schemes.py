
# sde/schemes.py  — Integrator-Protocol compliant schemes
import numpy as np
from typing import Dict, Any
from engine.types import Array  # keep your Array alias
# NOTE: We intentionally do NOT depend on Model/Discretization here anymore.
# These classes implement the "Integrator" protocol used by the ND Markov framework:
# 
# def step(self, x: Array, t: float, dt: float,
#          a, B, drv, θ: Dict[str, Any], rng=None) -> tuple[Array, float]
#     - x: (N,) current state vector
#     - a: drift function   a(x,t,θ) -> (N,)
#     - B: loading matrix   B(x,t,θ) -> (N,K)
#     - drv: Lévy driver with .sample(dt,size,rng)->(size,K)
#     - θ: parameter dict (contains any model-specific params)
#     - returns (x_next, t+dt)

# ----------------------------
# Generic Euler–Maruyama
# ----------------------------
class Euler:
    def step(self, x: Array, t: float, dt: float, a, B, drv, θ: Dict[str, Any], rng=None):
        dL = drv.sample(dt, size=1, rng=rng).reshape(-1)   # (K,)
        drift = a(x, t, θ)                                 # (N,)
        load  = B(x, t, θ)                                 # (N,K)
        x_new = x + drift * dt + load @ dL
        return x_new, t + dt


# -------------------------------------------
# Euler with full truncation for nonneg state
# -------------------------------------------
class FullTruncation:
    """Clamp specific indices to be non-negative (e.g., variance in Heston).
    Usage:
        integrator = FullTruncation(nonneg_indices=[1])  # if x[1] must stay >= 0
    """
    def __init__(self, nonneg_indices):
        self.nonneg_indices = list(nonneg_indices)

    def step(self, x: Array, t: float, dt: float, a, B, drv, θ: Dict[str, Any], rng=None):
        dL = drv.sample(dt, size=1, rng=rng).reshape(-1)
        x_eff = x.copy()
        x_eff[self.nonneg_indices] = np.maximum(x_eff[self.nonneg_indices], 0.0)
        drift = a(x_eff, t, θ)
        load  = B(x_eff, t, θ)
        x_new = x + drift * dt + load @ dL
        x_new[self.nonneg_indices] = np.maximum(x_new[self.nonneg_indices], 0.0)
        return x_new, t + dt


# ----------------------------
# Heston QE (variance-only)
# ----------------------------
class HestonQE:
    """Andersen QE for the variance state + log-Euler for S.
    Assumes state x = [S, v]; driver is 2D Brownian with correlation encoded in the driver.
    Required θ keys: { 'r', 'kappa', 'theta_v', 'xi' }.
    - Uses driver[:,0] for S and a *standard normal* Z2 for QE via driver[:,1]/sqrt(dt).
    Note: This integrator bypasses a/B for v-update (uses closed-form moments), but still
    satisfies the Integrator signature.
    """
    def __init__(self, psi_c: float = 1.5):
        self.psi_c = float(psi_c)

    @staticmethod
    def _norm_cdf(z: Array) -> Array:
        t = 1.0 / (1.0 + 0.2316419 * np.abs(z))
        a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
        poly = (((a5*t + a4)*t + a3)*t + a2)*t + a1
        pdf = (1.0/np.sqrt(2.0*np.pi)) * np.exp(-0.5*z*z)
        cdf = 1.0 - pdf * poly
        return np.where(z >= 0.0, cdf, 1.0 - cdf)

    def step(self, x: Array, t: float, dt: float, a, B, drv, θ: Dict[str, Any], rng=None):
        # Sample a 2D Brownian increment and convert second channel to N(0,1) for QE
        dL = drv.sample(dt, size=1, rng=rng).reshape(-1)    # (K,), need K>=2
        if dL.shape[0] < 2:
            raise ValueError("HestonQE requires a 2D Brownian driver (K>=2).")
        dW1 = dL[0]
        Z2  = dL[1] / np.sqrt(max(dt, 1e-16))               # ~ N(0,1)

        S = x[0]
        v = max(x[1], 0.0)

        k = float(θ['kappa']); th = float(θ['theta_v']); xi = float(θ['xi']); r = float(θ['r'])
        ek = np.exp(-k * dt)
        m  = th + (v - th) * ek
        k_eps = max(k, 1e-12)
        s2 = (v * xi*xi * ek * (1.0 - ek) / k_eps) + (th * xi*xi) * ((1.0 - ek)**2) / (2.0 * k_eps)

        m = max(m, 1e-16)
        psi = s2 / (m*m)
        psi_c = self.psi_c

        if psi <= psi_c:
            two_over_psi = 2.0 / max(psi, 1e-16)
            b2 = two_over_psi - 1.0 + np.sqrt(max(two_over_psi*(two_over_psi - 1.0), 0.0))
            b  = np.sqrt(max(b2, 1e-16))
            a_q  = m / (1.0 + b2)
            v_next = a_q * (b + Z2)**2
        else:
            p    = np.clip((psi - 1.0) / (psi + 1.0), 0.0, 0.999999)
            beta = (1.0 - p) / m
            U    = self._norm_cdf(Z2)
            v_next = 0.0 if U <= p else -np.log((1.0 - p) / max(1.0 - U, 1e-16)) / max(beta, 1e-16)

        v_next = max(v_next, 0.0)
        v_star = 0.5 * (v + v_next)

        S_next = S * np.exp((r - 0.5 * v_star) * dt + np.sqrt(max(v_star, 0.0)) * dW1)

        return np.array([S_next, v_next], float), t + dt


# ----------------------------
# SABR: exact alpha, robust F
# ----------------------------
class SABRExactAlpha:
    """Exact lognormal for alpha; robust step for F using mid-step alpha.
    Assumes state x = [F, alpha]; 2D Brownian driver with correlation encoded in driver.
    Required θ keys: { 'beta', 'nu', 'mu' (optional, default 0.0) }.
    Uses driver[:,0] for F and standard normal Z2 from driver[:,1]/sqrt(dt) for exact alpha.
    """
    def __init__(self, floor_F: float = 1e-16):
        self.floor_F = float(floor_F)

    def step(self, x: Array, t: float, dt: float, a, B, drv, θ: Dict[str, Any], rng=None):
        dL = drv.sample(dt, size=1, rng=rng).reshape(-1)
        if dL.shape[0] < 2:
            raise ValueError("SABRExactAlpha requires a 2D Brownian driver (K>=2).")
        dW1 = dL[0]
        Z2  = dL[1] / np.sqrt(max(dt, 1e-16))  # ~ N(0,1)

        F  = x[0]
        a0 = max(x[1], 0.0)

        nu   = float(θ['nu'])
        beta = float(θ['beta'])
        mu   = float(θ.get('mu', 0.0))

        # Exact alpha update (lognormal)
        a1 = a0 * np.exp(-0.5 * nu*nu * dt + nu * np.sqrt(dt) * Z2)
        a1 = max(a1, 0.0)

        a_star = 0.5 * (a0 + a1)

        if abs(beta - 1.0) < 1e-12:
            F_next = F * np.exp((mu - 0.5 * a_star*a_star) * dt + a_star * dW1)
        else:
            F_base = F if beta <= 0.0 else max(F, self.floor_F)
            F_next = F + mu * F * dt + a_star * (F_base ** beta) * dW1
            if beta > 0.0:
                F_next = max(F_next, self.floor_F)

        return np.array([F_next, a1], float), t + dt
