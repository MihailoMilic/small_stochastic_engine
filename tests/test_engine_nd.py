"""
Smoke tests for engine_nd.

Tests verify that the core pipeline (build a process → simulate → check output shape
and numerical sanity) works for GBM, Heston (QE), SABR (exact alpha), and a
Merton jump-diffusion.

Calibration tests are intentionally NOT included here: the calibrators depend on
scipy.optimize and are slow; they belong in a separate integration-test suite.
"""

import numpy as np
import pytest

from engine_nd import (
    SemimartingaleDriver,
    MarkovProcess,
    Euler,
    FullTruncation,
    HestonQE,
    SABRExactAlpha,
    SRKIntegrator,
    BasicMonteCarlo,
)
from engine_nd.recipes import gbm_a, gbm_B, heston_a, heston_B, merton_log_a, merton_log_B


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_brownian_driver(dim: int) -> SemimartingaleDriver:
    return SemimartingaleDriver(
        dim=dim,
        drift_fn=lambda t, path: np.zeros(dim),
        cov_fn=lambda t, path: np.eye(dim),
    )


def make_correlated_brownian(dim: int, rho: float) -> SemimartingaleDriver:
    C = np.array([[1.0, rho], [rho, 1.0]])
    return SemimartingaleDriver(
        dim=dim,
        drift_fn=lambda t, path: np.zeros(dim),
        cov_fn=lambda t, path: C,
    )


# ---------------------------------------------------------------------------
# GBM
# ---------------------------------------------------------------------------

def test_gbm_shape():
    driver = make_brownian_driver(1)
    proc = MarkovProcess(
        N=1, K=1,
        a=gbm_a, B=gbm_B,
        L=driver,
        θ={"mu": 0.08, "sigma": 0.2},
        x0=np.array([100.0]),
        t0=0.0,
        integrator=Euler(),
    )
    mc = BasicMonteCarlo()
    X = mc.simulate(proc, Euler(), n_paths=50, n_steps=252, dt=1/252, seed=0)
    assert X.shape == (50, 253, 1)
    assert np.all(np.isfinite(X))


def test_gbm_positive():
    driver = make_brownian_driver(1)
    proc = MarkovProcess(
        N=1, K=1,
        a=gbm_a, B=gbm_B,
        L=driver,
        θ={"mu": 0.05, "sigma": 0.3},
        x0=np.array([100.0]),
        t0=0.0,
    )
    X = BasicMonteCarlo().simulate(proc, Euler(), n_paths=20, n_steps=100, dt=1/252, seed=1)
    assert np.all(X > 0), "GBM prices should stay positive"


# ---------------------------------------------------------------------------
# Heston (QE integrator)
# ---------------------------------------------------------------------------

def test_heston_qe_shape():
    driver = make_correlated_brownian(2, rho=-0.7)
    proc = MarkovProcess(
        N=2, K=2,
        a=heston_a, B=heston_B,
        L=driver,
        θ={"mu": 0.05, "r": 0.02, "kappa": 1.5, "theta_v": 0.04, "xi": 0.3},
        x0=np.array([100.0, 0.04]),
        t0=0.0,
    )
    X = BasicMonteCarlo().simulate(proc, HestonQE(), n_paths=30, n_steps=50, dt=1/52, seed=2)
    assert X.shape == (30, 51, 2)
    assert np.all(np.isfinite(X))
    assert np.all(X[:, :, 1] >= 0.0), "Heston variance must be non-negative"


# ---------------------------------------------------------------------------
# SABR (exact alpha integrator)
# ---------------------------------------------------------------------------

def test_sabr_shape():
    driver = make_correlated_brownian(2, rho=-0.3)
    proc = MarkovProcess(
        N=2, K=2,
        a=lambda path, t, θ: np.zeros(2),
        B=lambda path, t, θ: np.eye(2),
        L=driver,
        θ={"beta": 0.7, "nu": 0.5, "mu": 0.0},
        x0=np.array([100.0, 0.3]),
        t0=0.0,
    )
    X = BasicMonteCarlo().simulate(proc, SABRExactAlpha(), n_paths=20, n_steps=50, dt=1/52, seed=3)
    assert X.shape == (20, 51, 2)
    assert np.all(np.isfinite(X))


# ---------------------------------------------------------------------------
# Merton log-price jump-diffusion
# ---------------------------------------------------------------------------

def _merton_driver(θ):
    def jump_fn(t, dt, path, rng):
        lam, mJ, sJ = θ["lam"], θ["mJ"], θ["sJ"]
        n = rng.poisson(lam * dt)
        jump = np.sum(rng.normal(mJ, sJ, size=n)) if n > 0 else 0.0
        return np.array([jump]), {}
    return SemimartingaleDriver(
        dim=2,
        drift_fn=lambda t, path: np.zeros(2),
        cov_fn=lambda t, path: np.diag([1.0, 0.0]),
        jump_fn=jump_fn,
    )


def test_merton_shape():
    θ = {"mu": 0.05, "sigma": 0.2, "lam": 2.0, "mJ": -0.05, "sJ": 0.1}
    driver = _merton_driver(θ)
    proc = MarkovProcess(
        N=1, K=2,
        a=merton_log_a, B=merton_log_B,
        L=driver,
        θ=θ,
        x0=np.array([np.log(100.0)]),
        t0=0.0,
    )
    X = BasicMonteCarlo().simulate(proc, Euler(), n_paths=20, n_steps=252, dt=1/252, seed=4)
    assert X.shape == (20, 253, 1)
    assert np.all(np.isfinite(X))


# ---------------------------------------------------------------------------
# SRK integrator
# ---------------------------------------------------------------------------

def test_srk1_gbm():
    driver = make_brownian_driver(1)
    proc = MarkovProcess(
        N=1, K=1,
        a=gbm_a, B=gbm_B,
        L=driver,
        θ={"mu": 0.05, "sigma": 0.2},
        x0=np.array([100.0]),
        t0=0.0,
    )
    for scheme in ("SRK1", "SRK2", "SRK3", "SRK4"):
        X = BasicMonteCarlo().simulate(proc, SRKIntegrator(scheme), n_paths=10, n_steps=50, dt=1/252, seed=5)
        assert X.shape == (10, 51, 1), f"{scheme} wrong shape"
        assert np.all(np.isfinite(X)), f"{scheme} produced non-finite values"


if __name__ == "__main__":
    test_gbm_shape()
    test_gbm_positive()
    test_heston_qe_shape()
    test_sabr_shape()
    test_merton_shape()
    test_srk1_gbm()
    print("All tests passed.")
