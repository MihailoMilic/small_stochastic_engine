# psi-enterprise
#Learning project, not validated.
A modular quantitative trading research platform. The core library is **`engine_nd`** — an N-dimensional stochastic process engine that underpins strategy development, simulation, and calibration.

## engine_nd

`engine_nd` models any asset price or state vector as a Markov process of the form:

```
dX_t = a(X_{≤t}, t) dt + B(X_{≤t}, t) dV_t
```

where `V_t` is a fully modular **stochastic driver** that can represent Brownian motion, Lévy jump-diffusions, fractional Brownian motion, Hawkes processes, or any composite of these.

### Architecture

```
engine_nd/
├── driver.py            — Abstract StochasticDriver + 4 concrete types
│                          (SemimartingaleDriver, GaussianKernelDriver,
│                           PointProcessDriver, CompositeDriver)
├── nd_types.py          — Type aliases, Protocols (Integrator, Markov,
│                          Calibrator, MonteCarlo), CalibrationResult, Cholsky
├── markov.py            — MarkovProcess dataclass (state, path, step)
├── schemes.py           — Numerical integrators: Euler, FullTruncation,
│                          HestonQE, SABRExactAlpha, SRKIntegrator (SRK1–SRK4)
├── simulators.py        — BasicMonteCarlo, QuasiMonteCarlo (placeholder)
├── calibration_nd.py    — GaussianMLECalibrator, GMMCalibrator,
│                          ParticleFilterMLE, HybridCalibrator,
│                          OptionSurfaceCalibrator
├── calibration_callbacks.py — loglik_gaussian/poisson/student, pricing losses
└── recipes.py           — Pre-built drift/loading functions for GBM, Heston,
                           Merton log-price
```

### Quickstart

```python
import numpy as np
from engine_nd import (
    SemimartingaleDriver, MarkovProcess,
    Euler, BasicMonteCarlo,
)
from engine_nd.recipes import gbm_a, gbm_B

driver = SemimartingaleDriver(
    dim=1,
    drift_fn=lambda t, path: np.zeros(1),
    cov_fn=lambda t, path: np.eye(1),
)

proc = MarkovProcess(
    N=1, K=1,
    a=gbm_a, B=gbm_B,
    L=driver,
    θ={"mu": 0.08, "sigma": 0.2},
    x0=np.array([100.0]),
    t0=0.0,
    integrator=Euler(),
)

X = BasicMonteCarlo().simulate(proc, Euler(), n_paths=1000, n_steps=252, dt=1/252, seed=0)
# X.shape == (1000, 253, 1)
```

See `docs/DRIVER_FRAMEWORK.md` for the full mathematical specification of the driver abstraction.

### What is implemented

- [x] Stochastic driver abstraction (`StochasticDriver`, `CompositeDriver`)
- [x] Semimartingale drivers (Brownian, correlated Brownian, Lévy jumps)
- [x] Gaussian kernel drivers (fBM, rough volatility)
- [x] Point process drivers (Poisson, Hawkes, marked)
- [x] Numerical integrators: Euler–Maruyama, Full Truncation, Heston QE, SABR exact-alpha, SRK1–SRK4
- [x] Basic Monte Carlo simulator
- [x] Calibrators: Gaussian MLE, GMM, Particle Filter MLE, Hybrid, Option Surface
- [x] Recipes: GBM, Heston, Merton log-price

### What is NOT yet implemented / incomplete

- [ ] `QuasiMonteCarlo`: Sobol normals are generated but not injected into the integrator — needs a `driver.from_standard` interface or a dedicated QMC integrator wrapper; currently falls back to independent sampling
- [ ] `AntitheticMonteCarlo`: placeholder only
- [ ] `ParallelMonteCarlo` / `GpuMonteCarlo`: not started
- [ ] `RegimeSwitchingModel` and its calibrator: class body is incomplete in the old `test.py` — needs a proper `RegimeSwitchingCalibratable` Protocol and forward-algorithm implementation
- [ ] Pricing / implied-vol surface methods (`process.pricer()`) referenced in `calibration_callbacks.py` — these assume an attached pricer object that does not exist yet
- [ ] `Cholsky` dynamic correlation: `function` path is wired but no examples exist
- [ ] Tests for calibration routines (slow; need a separate integration test suite)

## Platform modules (spec only)

The five-module platform spec lives in `docs/`. Only Module 2 (the SDE engine above) has code.

| Module | Description | Status |
|--------|-------------|--------|
| 1 | Data, news, sentiment, alternative signals | Spec only |
| **2** | **SDE engine, strategy builder, signal processing** | **Active — see `engine_nd/`** |
| 3 | Execution & risk layer (KYC, VaR, pre-trade checks) | Spec only |
| 4 | Backtesting & simulation framework | Spec only |
| 5 | Portfolio management & orchestration | Spec only |

## Installation

```bash
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```
