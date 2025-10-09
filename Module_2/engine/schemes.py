# sde/schemes.py
import numpy as np
from .types import Model, Array, Discretization
from .model import Heston, SABR
class Euler(Discretization):
    def step(self, model: Model, x: Array, t: float, dt: float, dW: Array) -> Array:
        a = model.drift(x, t)
        b = model.diffusion(x, t)
        jumps = model.jumps(x, dt)
        extra = model.extra_terms(x, t, dt, dW)
        return x + a*dt + b*dW + jumps + extra

class Milstein(Discretization):
    def step(self, model: Model, x: Array, t: float, dt: float, dW: Array) -> Array:
        a = model.drift(x, t)
        b = model.diffusion(x, t)
        jumps = model.jumps(x, dt)
        extra = model.extra_terms(x, t, dt, dW)
        bp = model.diffusion_dx(x, t)
        return x + a*dt + b*dW + 0.5*b*bp*(dW**2 - dt) + jumps + extra

class SRKScalar(Discretization):
    """Strong ~1.5 for scalar SDE; use for 1D only unless extended."""
    def step(self, model: Model, x: Array, t: float, dt: float, dW: Array) -> Array:
        a = model.drift(x, t)
        b = model.diffusion(x, t)
        bp = model.diffusion_dx(x, t)
        jumps = model.jumps(x, dt)
        extra = model.extra_terms(x, t, dt, dW)
        x_new = x + a*dt + b*dW + 0.5*b*bp*(dW**2 - dt)
        a_tilde = model.drift(x + a*dt + b*np.sqrt(dt), t+dt)
        x_new += 0.5*(a_tilde - a)*dt 
        return x_new + jumps + extra



class HestonQE(Discretization):
    """Andersen QE for Heston variance + log-Euler for S.
       Expects a Heston-like model with attributes: r, kappa, theta, xi.
       Uses dW[:,0] for S and dW[:,1] (→ Z2) for variance update."""
    def __init__(self, psi_c: float = 1.5):
        self.psi_c = psi_c

    @staticmethod
    def _norm_cdf(z: Array) -> Array:
        # Fast Abramowitz–Stegun approximation; vectorized
        # P(Z<=z), Z~N(0,1)
        t = 1.0 / (1.0 + 0.2316419 * np.abs(z))
        a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
        poly = (((a5*t + a4)*t + a3)*t + a2)*t + a1
        pdf = (1.0/np.sqrt(2.0*np.pi)) * np.exp(-0.5*z*z)
        cdf = 1.0 - pdf * poly
        return np.where(z >= 0.0, cdf, 1.0 - cdf)

    def step(self, model: Heston, x: Array, t: float, dt: float, dW: Array) -> Array:
        # state & drivers
        S = x[:, 0]
        v = np.maximum(x[:, 1], 0.0)
        dW1 = dW[:, 0]                           # correlated asset driver
        Z2  = dW[:, 1] / np.sqrt(dt)             # standard normal driver for QE

        # QE moments for CIR variance
        k, th, xi = model.kappa, model.theta, model.xi
        ek = np.exp(-k * dt)

        m  = th + (v - th) * ek
        k_eps = np.maximum(k, 1e-12)  # avoid div-by-zero
        s2 = (v * xi*xi * ek * (1.0 - ek) / k_eps) + (th * xi*xi) * ((1.0 - ek)**2) / (2.0 * k_eps)

        m = np.maximum(m, 1e-16)      # numeric guard
        psi = s2 / (m*m)
        psi_c = self.psi_c

        # Regime 1: Quadratic
        use_quad = psi <= psi_c
        psi_q = np.where(use_quad, psi, 1.0)     # dummy for shape
        two_over_psi = 2.0 / np.maximum(psi_q, 1e-16)
        b2 = two_over_psi - 1.0 + np.sqrt(np.maximum(two_over_psi*(two_over_psi - 1.0), 0.0))
        b  = np.sqrt(np.maximum(b2, 1e-16))
        a  = m / (1.0 + b2)
        v_quad = a * (b + Z2)**2

        # Regime 2: Exponential
        psi_e = np.where(~use_quad, psi, 2.0)    # dummy for shape
        p    = (psi_e - 1.0) / (psi_e + 1.0)     # in [0,1)
        p    = np.clip(p, 0.0, 0.999999)
        beta = (1.0 - p) / m
        U    = self._norm_cdf(Z2)                # map Z2 -> U to preserve QMC structure
        v_exp = np.where(U <= p, 0.0, -np.log((1.0 - p) / np.maximum(1.0 - U, 1e-16)) / np.maximum(beta, 1e-16))

        v_next = np.where(use_quad, v_quad, v_exp)
        v_next = np.maximum(v_next, 0.0)

        # Effective variance over the step
        v_star = 0.5 * (v + v_next)

        # Asset update (log-Euler with correlated dW1)
        S_next = S * np.exp((model.r - 0.5 * v_star) * dt + np.sqrt(v_star) * dW1)

        # Return stacked state (N,2)
        out = np.empty_like(x)
        out[:, 0] = S_next
        out[:, 1] = v_next
        return out
# sde/schemes.py
import numpy as np
from .types import Model, Array, Discretization

class HestonCIRExact(Discretization):
    """Exact CIR transition for v; log-Euler for S with v* = (v+v_next)/2.
       Uses only dW[:,0] for S. Correlation enters via Simulator.cholesky,
       but v update itself doesn’t consume dW[:,1] (not full BK exact)."""

    def step(self, model: Heston, x: Array, t: float, dt: float, dW: Array) -> Array:
        # State
        S = x[:, 0]
        v = np.maximum(x[:, 1], 0.0)

        # Parameters
        k, th, xi = model.kappa, model.theta, model.xi

        # Exact CIR(v) transition: v_{t+dt} = c * χ'^2_d(λ)
        # c = (ξ^2 (1 - e^{-κΔ})) / (4κ)
        # d = 4κθ / ξ^2
        # λ = (4κ e^{-κΔ} v_t) / (ξ^2 (1 - e^{-κΔ}))
        ek = np.exp(-k * dt)
        one_minus_ek = 1.0 - ek
        # Guard against division by zero for tiny k or dt
        k_pos = np.maximum(k, 1e-16)
        xi2 = xi * xi

        c = (xi2 * one_minus_ek) / (4.0 * k_pos)
        d = (4.0 * k_pos * th) / np.maximum(xi2, 1e-16)
        lam = (4.0 * k_pos * ek * v) / np.maximum(xi2 * one_minus_ek, 1e-16)

        v_next = c * np.random.noncentral_chisquare(df=d, nonc=lam, size=v.shape)
        v_next = np.maximum(v_next, 0.0)

        # Effective variance over the step (trapezoid)
        v_star = 0.5 * (v + v_next)

        # Asset update: log-Euler with correlated driver dW[:,0]
        dW1 = dW[:, 0]
        S_next = S * np.exp((model.r - 0.5 * v_star) * dt + np.sqrt(np.maximum(v_star, 0.0)) * dW1)

        out = np.empty_like(x)
        out[:, 0] = S_next
        out[:, 1] = v_next
        return out
# sde/schemes.py


class SABRExactAlpha(Discretization):
    """Exact lognormal step for alpha; robust step for F using mid-step alpha."""
    def __init__(self, floor_F: float = 1e-16):
        self.floor_F = float(floor_F)

    def step(self, model: SABR, x: Array, t: float, dt: float, dW: Array) -> Array:
        F  = x[:, 0]
        a0 = np.maximum(x[:, 1], 0.0)

        # Drivers (already correlated by Simulator.cholesky)
        dW1 = dW[:, 0]
        Z2  = dW[:, 1] / np.sqrt(dt)  # standard normal for exact alpha

        # 1) Exact alpha update (lognormal)
        nu = model.nu
        a1 = a0 * np.exp(-0.5 * nu*nu * dt + nu * np.sqrt(dt) * Z2)
        a1 = np.maximum(a1, 0.0)

        # Mid-step alpha
        a_star = 0.5 * (a0 + a1)

        beta = model.beta
        mu   = getattr(model, "mu", 0.0)

        # 2) F update
        if abs(beta - 1.0) < 1e-12:
            # Log-Euler for beta=1 (Black-type)
            F_next = F * np.exp((mu - 0.5 * a_star*a_star) * dt + a_star * dW1)
        else:
            # Euler for general beta (includes normal beta=0)
            F_base = F if beta <= 0.0 else np.maximum(F, self.floor_F)
            F_next = F + mu * F * dt + a_star * (F_base ** beta) * dW1
            # Optional clamp for beta>0 to avoid negative due to Euler
            if beta > 0.0:
                F_next = np.maximum(F_next, self.floor_F)

        out = np.empty_like(x)
        out[:, 0] = F_next
        out[:, 1] = a1
        return out
