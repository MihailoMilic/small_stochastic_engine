
import numpy as np
from typing import Callable
from .nd_types import Array, LevyDriver

class BrownianDriver:
    """K-dim Brownian increments with covariance L L^T; if L is identity → iid N(0, dt)."""
    def __init__(self, L: Array):
        self.L = np.asarray(L, float)
        assert self.L.ndim == 2 and self.L.shape[0] == self.L.shape[1], "L must be square"
        self.dim = self.L.shape[0]

    def sample(self, dt: float, size: int = 1, rng: np.random.Generator | None = None) -> Array:
        rng = rng or np.random.default_rng()
        Z = rng.normal(size=(size, self.dim))
        return (Z @ self.L.T) * np.sqrt(dt)

class CompoundPoissonDriver:
    """Finite-activity jumps in R^K with intensity λ and vector jump sampler `rvs(n, rng)->(n,K)`."""
    def __init__(self, lam: float, jump_sampler: Callable[[int, np.random.Generator], Array], K: int):
        self.lam = float(lam)
        self.jump_sampler = jump_sampler
        self.dim = int(K)

    def sample(self, dt: float, size: int = 1, rng: np.random.Generator | None = None) -> Array:
        rng = rng or np.random.default_rng()
        out = np.zeros((size, self.dim))
        Ns = rng.poisson(self.lam * dt, size=size)
        for i, n in enumerate(Ns):
            if n > 0:
                J = self.jump_sampler(n, rng).reshape(n, self.dim)
                out[i] = J.sum(axis=0)
        return out

class VarianceGammaDriver:
    """K-dim VG via time-change: θ*G + σ*sqrt(G)*N, G ~ Gamma(dt/ν, scale=ν). Per-dim params allowed."""
    def __init__(self, theta: Array, sigma: Array, nu: Array):
        self.theta = np.asarray(theta, float).ravel()
        self.sigma = np.asarray(sigma, float).ravel()
        self.nu    = np.asarray(nu, float).ravel()
        assert self.theta.shape == self.sigma.shape == self.nu.shape
        self.dim = self.theta.size

    def sample(self, dt: float, size: int = 1, rng: np.random.Generator | None = None) -> Array:
        rng = rng or np.random.default_rng()
        alpha = dt / self.nu
        G = rng.gamma(shape=alpha, scale=self.nu, size=(size, self.dim))
        Z = rng.normal(size=(size, self.dim))
        return self.theta * G + self.sigma * np.sqrt(G) * Z

class MixtureDriver:
    """Sum outputs of sub-drivers (must share the same dimension K)."""
    def __init__(self, *drivers: LevyDriver):
        if not drivers:
            raise ValueError("Provide at least one driver")
        K = drivers[0].dim
        for d in drivers:
            if d.dim != K:
                raise ValueError("All drivers must share same dimension")
        self._drivers = drivers
        self.dim = K

    def sample(self, dt: float, size: int = 1, rng: np.random.Generator | None = None) -> Array:
        parts = [d.sample(dt, size=size, rng=rng) for d in self._drivers]
        return np.add.reduce(parts)
