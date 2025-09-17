# Engine API Overview

This folder hosts the lightweight SDE engine that implements the practical API described in `workflow.md`. It focuses on composable models, numerical schemes, and a simple simulation loop to support pricing and strategy prototyping.

## What’s Implemented Now

- Models (`engine/model.py`)
  - `GBM`, `MertonJD` (jump–diffusion), `AdditiveLevyModel`
  - `Heston`, `SABR`
  - `RegimeSwitchingModel` with continuous-time generator `Q` and `expm(Q·dt)` transitions
- Lévy components (`engine/levy.py`)
  - `LevyTriplet`, `GaussianJumpMeasure`, `TruncatedLevyIncrement`
- Schemes (`engine/schemes.py`)
  - Generic: `Euler`, `Milstein`, `SRKScalar` 
  - Model-specific: `HestonQE`, `HestonCIRExact`, `SABRExactAlpha`
- Simulation (`engine/simulate.py`)
  - `Simulator` orchestrates paths with pluggable scheme and optional Brownian correlation
- Randomness (`engine/rng.py`, `engine/brownian.py`)
  - `PRNG` wrapper; `SobolNormals` for QMC; Brownian bridge mapping for low effective dimension
- Payoffs (`engine/payoff.py`)
  - `EuropeanCall` (extend with custom payoffs as needed)
- Pricing (`engine/pricer.py`)
  - `MonteCarloPricer` and `QuasiMonteCarloPricer` (with optional antithetics and control variates)

These components align with the Module 2 goals in `workflow.md`: flexible SDE modeling, regime switching, and MC/QMC simulation for pricing and experiment workflows.

## Core API (quick map)

- Model protocol (`engine/types.py`)
  - Requires: `dim`, `init`, and methods `drift(x,t)`, `diffusion(x,t)`, `diffusion_dx(x,t)`, `jumps(x,dt)`, `extra_terms(x,t,dt,dW)`
- Discretization protocol (`engine/types.py`)
  - `step(model, x, t, dt, dW) -> x_next`
- Simulator (`engine/simulate.py`)
  - `Simulator(model, scheme, dt, num_steps, num_paths, seed, cholesky=None)`
  - `run(dW_supplier)` where `dW_supplier(T,N) -> (T,N,dimW)` with variance `dt`
- Pricers (`engine/pricer.py`)
  - `MonteCarloPricer(sim, r, prng, antithetic=False, control=None, control_expectation=None)`
  - `QuasiMonteCarloPricer(sim, r, prng, sobol=None, use_bridge=True)`

## Minimal Examples

### 1) Simulate GBM paths

```python
import numpy as np
from engine.model import GBM
from engine.schemes import Euler
from engine.simulate import Simulator
from engine.rng import PRNG

dt, T, N = 1/252, 252, 1000
model = GBM(mu=0.05, sigma=0.20, s0=100.0)
sim = Simulator(model=model, scheme=Euler(), dt=dt, num_steps=T, num_paths=N, seed=42, cholesky=None)

# Brownian increments: shape (T, N, dimW) with variance dt
dW_supplier = lambda T,N: PRNG(123).normal((T, N, 1)) * np.sqrt(dt)
paths = sim.run(dW_supplier)
print(paths.shape)  # (T+1, N, 1)
```

### 2) Price a European call with MC

```python
import numpy as np
from engine.model import GBM
from engine.schemes import Euler
from engine.simulate import Simulator
from engine.pricer import MonteCarloPricer
from engine.payoff import EuropeanCall
from engine.rng import PRNG

dt, T, N = 1/252, 252, 20000
model = GBM(mu=0.05, sigma=0.2, s0=100.0)
sim = Simulator(model=model, scheme=Euler(), dt=dt, num_steps=T, num_paths=N, seed=1, cholesky=None)
pricer = MonteCarloPricer(sim=sim, r=0.02, prng=PRNG(7), antithetic=True)
res = pricer.price(EuropeanCall(strike=100.0))
print(res)  # {'price': ..., 'stderr': ...}
```

### 3) Regime-switching GBM

```python
import numpy as np
from engine.model import GBM, RegimeSwitchingModel
from engine.schemes import Euler
from engine.simulate import Simulator
from engine.rng import PRNG

dt, T, N = 1/252, 252, 5000
gbm_lo = GBM(mu=0.05, sigma=0.20, s0=100.0)
gbm_hi = GBM(mu=0.02, sigma=0.50, s0=100.0)
Q = np.array([[-0.10, 0.10], [0.30, -0.30]], float)
pi0 = np.array([0.8, 0.2], float)

model = RegimeSwitchingModel(models=[gbm_lo, gbm_hi], Q=Q, pi0=pi0, dt=dt)
sim = Simulator(model=model, scheme=Euler(), dt=dt, num_steps=T, num_paths=N, seed=2, cholesky=None)
paths = sim.run(lambda T,N: PRNG(11).normal((T, N, 1)) * np.sqrt(dt))
print(paths.shape)
```

### 4) Heston with QE scheme

```python
import numpy as np
from engine.model import Heston
from engine.schemes import HestonQE
from engine.simulate import Simulator
from engine.rng import PRNG

dt, T, N = 1/252, 252, 10000
model = Heston(r=0.0, kappa=2.0, theta=0.04, xi=1.0, rho=-0.7, S0=100.0, v0=0.04)
scheme = HestonQE()
L = model.cholsky  # (2x2) correlation for Brownian drivers
sim = Simulator(model=model, scheme=scheme, dt=dt, num_steps=T, num_paths=N, seed=3, cholesky=L)
paths = sim.run(lambda T,N: PRNG(5).normal((T, N, 2)) * np.sqrt(dt))
print(paths.shape)  # (T+1, N, 2) for [S, v]
```

## Notes and Conventions

- Shapes
  - Paths: `(T+1, N, dim)`; Brownian increments: `(T, N, dimW)`
  - For 1D models, `dim = dimW = 1`. Multi-factor models may set `dimW > dim`.
- Correlation
  - Provide `cholesky` to `Simulator` to correlate Brownian drivers: `dW_t = L @ Z_t`.
- Regime Switching
  - Uses `P = expm(Q·dt)` with row normalization safeguards. Provide valid generator `Q` (off-diagonal ≥ 0, row sums 0).
- Lévy Models
  - `TruncatedLevyIncrement` approximates small jumps as Gaussian; large jumps via compound Poisson.

## Requirements

- Python 3.10+
- NumPy, SciPy

Install extras if needed:

```bash
pip install numpy scipy
```

## Extending

- Add new models by conforming to the `Model` protocol (`engine/types.py`).
- Add schemes by implementing the `Discretization` protocol.
- Add payoffs by implementing `__call__(paths) -> ndarray` over terminal or path-dependent values.

## Quick Tips

- Set RNG seeds (`Simulator.seed`, `PRNG(seed)`) for reproducibility.
- For QMC, use `QuasiMonteCarloPricer` with `use_bridge=True` for better convergence on path-dependent payoffs.
- For barriers, consider smaller `dt` or a Brownian bridge crossing correction.

