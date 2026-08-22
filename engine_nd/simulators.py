import numpy as np
from scipy.stats import qmc, norm
from .nd_types import Array, Markov, Params, Integrator


class BasicMonteCarlo:
    """Simple CPU Monte Carlo simulator using any Integrator."""

    def simulate(
        self,
        process: Markov,
        integrator: Integrator,
        n_paths: int,
        n_steps: int,
        dt: float,
        seed: int | None = None,
        **kwargs,
    ) -> Array:
        rng = np.random.default_rng(seed)
        state_dim = process.x0.size
        X = np.zeros((n_paths, n_steps + 1, state_dim))
        X[:, 0] = process.x0
        for i in range(n_steps):
            current_t = process.t0 + i * dt
            for p in range(n_paths):
                path_slice = X[p, :i + 1]
                next_state, _ = integrator.step(
                    path_slice, current_t, dt,
                    process.a, process.B,
                    process.L, process.θ, rng=rng,
                )
                X[p, i + 1] = next_state
        return X


class QuasiMonteCarlo:
    """Sobol-based quasi Monte Carlo simulator.

    NOTE — NOT FULLY IMPLEMENTED: pre-generated Sobol normals are not yet
    injected into the integrator. The current integrator API accepts an
    ``np.random.Generator`` rather than a pre-drawn sample, so QMC cannot
    bypass the integrator's internal sampling. A proper implementation requires
    either a ``from_standard`` path on the driver or a dedicated QMC integrator
    wrapper. For now this falls back to independent sampling and is kept as a
    placeholder.
    """

    def simulate(
        self,
        process: Markov,
        integrator: Integrator,
        n_paths: int,
        n_steps: int,
        dt: float,
        seed: int = 1,
        **kwargs,
    ) -> Array:
        # TODO: wire pre-generated Sobol normals through driver.from_standard
        # when that interface is available.
        rng = np.random.default_rng(seed)
        state_dim = process.x0.size
        X = np.zeros((n_paths, n_steps + 1, state_dim))
        X[:, 0] = process.x0
        for i in range(n_steps):
            current_t = process.t0 + i * dt
            for p in range(n_paths):
                path_slice = X[p, :i + 1]
                X[p, i + 1], _ = integrator.step(
                    path_slice, current_t, dt,
                    process.a, process.B,
                    process.L, process.θ, rng=rng,
                )
        return X


# Planned extensions (not yet implemented):
# - ParallelMonteCarlo  — concurrent.futures / Ray for distributed path generation
# - AntitheticMonteCarlo — variance reduction via antithetic pairs
# - ControlVariateMonteCarlo — control variate correction term
# - GpuMonteCarlo — CuPy / PyTorch backend
