
# sde/schemes.py  — Integrator-Protocol compliant schemes
import numpy as np
from typing import Dict, Any, Callable, Literal, Optional

from engine_nd.driver import StochasticDriver   
from .nd_types import DriftFn, LoadFn, Integrator, Params, Path
from .nd_types import Array  # keep your Array alias
# NOTE: We intentionally do NOT depend on Model/Discretization here anymore.
# These classes implement the "Integrator" protocol used by the ND Markov framework:
#
# def step(self, path: Array, t: float, dt: float,
#          a, B, levy_fn, theta: Dict[str, Any], rng=None) -> tuple[Array, float]
#     - path: (M,N) array of the realized path up to time t (latest row is current state)
#     - a: drift function   a(path,t,theta) -> (N,)
#     - B: loading matrix   B(path,t,theta) -> (N,K)
#     - levy_fn: callable returning Lévy increments based on (path,t,dt,theta)
#     - theta: parameter dict (contains any model-specific params)
#     - returns (x_next, t+dt)

# ----------------------------
# Generic Euler–Maruyama
# ----------------------------
class Euler:
    def step(self, path: Path, t: float, dt: float, a: DriftFn, B: LoadFn,
             driver, theta: Dict[str, Any], rng: Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng()
        path_arr = np.atleast_2d(np.asarray(path, float))
        x = path_arr[-1]
        drift = a(path_arr, t, theta)
        load  = B(path_arr, t, theta)
        dL, meta = driver.step(t, dt, path_arr, rng = rng)
        dL = dL.reshape(-1)
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

    def step(self, path: Path, t: float, dt: float, a: DriftFn, B: LoadFn,
             driver:StochasticDriver, theta: Dict[str, Any], rng: Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng()
        path_arr = np.atleast_2d(np.asarray(path, float))
        x = path_arr[-1]
        dL, meta = driver.step(t, dt, path_arr, rng = rng)
        dL = dL.reshape(-1)
        x_eff = x.copy()
        x_eff[self.nonneg_indices] = np.maximum(x_eff[self.nonneg_indices], 0.0)
        path_eff = path_arr.copy()
        path_eff[-1] = x_eff
        drift = a(path_eff, t, theta)
        load  = B(path_eff, t, theta)
        x_new = x + drift * dt + load @ dL
        x_new[self.nonneg_indices] = np.maximum(x_new[self.nonneg_indices], 0.0)
        return x_new, t + dt


# ----------------------------
# Heston QE (variance-only)
# ----------------------------
class HestonQE:
    """Andersen QE for the variance state + log-Euler for S.
    Assumes state x = [S, v]; driver is 2D Brownian with correlation encoded in the driver.
    Required theta keys: { 'r', 'kappa', 'theta_v', 'xi' }.
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

    def step(self, path: Path, t: float, dt: float, a: DriftFn, B: LoadFn,
             driver : StochasticDriver, theta: Dict[str, Any], rng:Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng()
        path_arr = np.atleast_2d(np.asarray(path, float))
        # Sample a 2D Brownian increment and convert second channel to N(0,1) for QE
        dL, meta = driver.step(t, dt, path_arr, rng=rng)
        dL = dL.reshape(-1)
        if dL.shape[0] < 2:
            raise ValueError("HestonQE requires a 2D Brownian driver (K>=2).")
        dW1 = dL[0]
        Z2  = dL[1] / np.sqrt(max(dt, 1e-16))               # ~ N(0,1)

        state = path_arr[-1]
        S = state[0]
        v = max(state[1], 0.0)

        k = float(theta['kappa']); th = float(theta['theta_v']); xi = float(theta['xi']); r = float(theta['r'])
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
    Required theta keys: { 'beta', 'nu', 'mu' (optional, default 0.0) }.
    Uses driver[:,0] for F and standard normal Z2 from driver[:,1]/sqrt(dt) for exact alpha.
    """
    def __init__(self, floor_F: float = 1e-16):
        self.floor_F = float(floor_F)

    def step(self, path: Path, t: float, dt: float, a: DriftFn, B: LoadFn,
             driver: StochasticDriver, theta: Dict[str, Any], rng: Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng()
        path_arr = np.atleast_2d(np.asarray(path, float))
        dL, meta = driver.step(t, dt, path_arr, rng=rng)
        dL = dL.reshape(-1)
        if dL.shape[0] < 2:
            raise ValueError("SABRExactAlpha requires a 2D Brownian driver (K>=2).")
        dW1 = dL[0]
        Z2  = dL[1] / np.sqrt(max(dt, 1e-16))  # ~ N(0,1)

        state = path_arr[-1]
        F  = state[0]
        a0 = max(state[1], 0.0)

        nu   = float(theta['nu'])
        beta = float(theta['beta'])
        mu   = float(theta.get('mu', 0.0))

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


class SRKIntegrator:
    """
    General n-dimensional Stochastic Runge–Kutta Integrator using the provided Lévy driver.

    - SRK1 works with any driver by using the sampled increment dL directly.
    - SRK2 uses a Platen-style correction that assumes Brownian increments; if the driver does
      not expose a Brownian structure (no L), it automatically falls back to SRK1.
    - SRK3/SRK4 sample additional independent increments from the driver (half-steps for SRK4);
      these are Brownian-consistent. For jump drivers they act as higher-order heuristics.

    SDE form: dX = μ(X, t) dt + σ(X, t) dL, with dL provided by `drv.sample`.
    """

    def __init__(
        self,
        scheme: Literal["SRK1", "SRK2", "SRK3", "SRK4"] = "SRK1",
        seed: Optional[int] = None
    ):
        self.scheme = scheme.upper()
        # Keep an internal RNG only as a fallback; primary randomness comes from drv.sample
        self.rng = np.random.default_rng(seed)

    # Step function (core integrator)
    def step(self, path: Path, t: float, dt: float,
             a: DriftFn, B: LoadFn, drv,
             theta: Params, rng: Optional[np.random.Generator] = None) -> tuple[Array, float]:
        path_arr = np.atleast_2d(np.asarray(path, float))
        
        rng = rng or self.rng
        if self.scheme == "SRK1":
            return self._srk1(path_arr, t, dt, a, B, drv, theta, rng)
        elif self.scheme == "SRK2":
            return self._srk2(path_arr, t, dt, a, B, drv, theta, rng)
        elif self.scheme == "SRK3":
            return self._srk3(path_arr, t, dt, a, B, drv, theta, rng)
        elif self.scheme == "SRK4":
            return self._srk4(path_arr, t, dt, a, B, drv, theta, rng)
        else:
            raise NotImplementedError(f"Scheme {self.scheme} not implemented yet")


    # SRK-1 (Heun) — strong order 1.0

    def _srk1(self, path: Array, t: float, dt: float, a, B, drv:StochasticDriver, theta: Dict[str, Any], rng) -> tuple[Array, float]:
        x = path[-1]
        sigma_t = B(path, t, theta)                                 # (N,K)
        dL, meta = drv.step(t, dt, path, rng=rng)     # (K,)
        dL = dL.reshape(-1)
        mu_t = a(path, t, theta)
        X_star = x + mu_t * dt + sigma_t @ dL
        path_star = np.vstack([path, X_star])
        mu_star = a(path_star, t + dt, theta)
        x_next = x + 0.5 * (mu_t + mu_star) * dt + sigma_t @ dL
        return x_next, t + dt

    # SRK-2 (Platen) — strong order 1.5
    def _srk2(self, path: Array, t: float, dt: float, a, B, drv:StochasticDriver, theta: Dict[str, Any], rng) -> tuple[Array, float]:
        # SRK2 relies on Brownian calculus for the dZ term. If the driver does not look
        # Brownian-like (no L), fall back to SRK1 using the Lévy increment directly.
        brownian_like = hasattr(drv, "L")
        if not brownian_like:
            return self._srk1(path, t, dt, a, B, drv, theta, rng)

        x = path[-1]
        sigma_t = B(path, t, theta)
        dW, meta = drv.step(t, dt, path, rng=rng)
        dW = dW.reshape(-1)
        # Per-channel SRK2 correction term using Brownian moments
        sqrt_dt = np.sqrt(max(dt, 1e-16))
        dZ = 0.5 * (dW**2 - dt) / sqrt_dt

        mu_t = a(path, t, theta)
        X_star = x + mu_t * dt + sigma_t @ dW
        path_star = np.vstack([path, X_star])
        mu_star = a(path_star, t + dt, theta)
        sigma_star = B(path_star, t + dt, theta)

        correction = 0.5 * (sigma_t + sigma_star) @ dZ
        x_next = x + 0.5 * (mu_t + mu_star) * dt + sigma_t @ dW + correction
        return x_next, t + dt

    # SRK-3 (weak order 2.0, Kloeden–Platen)
    def _srk3(self, path: Array, t: float, dt: float, a, B, drv:StochasticDriver, theta: Dict[str, Any], rng) -> tuple[Array, float]:
        x = path[-1]
        sigma_t = B(path, t, theta)
        dL , meta = drv.step(t, dt, path, rng=rng)
        dL = dL.reshape(-1)
        mu_t = a(path, t, theta)
        X1 = x + mu_t * dt + sigma_t @ dL
        path1 = np.vstack([path, X1])
        mu1 = a(path1, t + dt, theta)
        sigma1 = B(path1, t + dt, theta)
        X2 = x + 0.5 * (mu_t + mu1) * dt + 0.5 * (sigma_t + sigma1) @ dL
        return X2, t + dt
    # SRK-4 (weak order 2.0–2.5 variant)
    def _srk4(self, path: Array, t: float, dt: float, a, B, drv, theta: Dict[str, Any], rng) -> tuple[Array, float]:
        x = path[-1]
        sigma_t = B(path, t, theta)
        # Use two independent half-step increments from the driver (Brownian-consistent)
        dL1, meta1 = drv.step(t, dt / 2.0, path, rng=rng)
        dL1 = dL1.reshape(-1)
        path_half = np.vstack([path, x + sigma_t @ (dL1 * 0.5)])
        dL2, meta2 = drv.step(t + dt / 2.0, dt / 2.0, path_half, rng=rng)
        dL2 = dL2.reshape(-1)
        mu_t = a(path, t, theta)

        k1 = mu_t
        path_k2 = np.vstack([path, x + sigma_t @ (dL1 * 0.5)])
        k2 = a(path_k2, t + 0.5 * dt, theta)
        path_k3 = np.vstack([path, x + sigma_t @ (dL2 * 0.5)])
        k3 = a(path_k3, t + 0.5 * dt, theta)
        path_k4 = np.vstack([path, x + sigma_t @ (dL1 + dL2)])
        k4 = a(path_k4, t + dt, theta)

        drift_term = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        x_next = x + drift_term * dt + sigma_t @ (dL1 + dL2)
        return x_next, t + dt
