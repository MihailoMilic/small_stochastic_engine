import numpy as np
from scipy.stats import qmc, norm
from .nd_types import Array, Markov, Params, Integrator



class BasicMonteCarlo:
    """Simple CPU Monte Carlo using any Integrator."""
    def simulate(
        self, process: Markov, integrator: Integrator,
        n_paths: int, n_steps: int, dt: float,
        seed: int | None = None, **kwargs
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
                    process.L, process.θ, rng=rng
                )
                X[p, i + 1] = next_state
        return X

# ?
# class AntitheticMonteCarlo: 
#     """Monte Carlo with antithetic variance reduction."""
#     def simulate(
#         self, process: Markov, integrator: Integrator,
#         n_paths: int, n_steps: int, dt: float,
#         seed: int | None = None, **kwargs
#     ) -> Array:
#         rng = np.random.default_rng(seed)
#         half = n_paths // 2

#         # Normal paths
#         process.levy_driver.antithetic = False
#         X_plus = BasicMonteCarlo().simulate(process, integrator, half, n_steps, dt, seed=seed)

#         # Antithetic paths (negated noise)
#         process.levy_driver.antithetic = True
#         X_minus = BasicMonteCarlo().simulate(process, integrator, half, n_steps, dt, seed=(seed + 1 if seed else None))

#         process.levy_driver.antithetic = False
#         return np.concatenate([X_plus, X_minus], axis=0)


class QuasiMonteCarlo:
    """Sobol-based quasi Monte Carlo."""
    def simulate(
        self, process: Markov, integrator: Integrator,
        n_paths: int, n_steps: int, dt: float,
        seed: int =1, **kwargs
    ) -> Array:
        dim = process.L.dim
        state_dim = process.x0.size

        sampler = qmc.Sobol(d=dim * n_steps, scramble=True, rng=np.random.Generator(np.random.PCG64(seed)))
        m = int(np.ceil(np.log2(max(1, n_paths))))
        u_full = sampler.random_base2(m)
        u = u_full[:n_paths]
        z = norm.ppf(u)
        z = np.clip(z, -10.0, 10.0)
        normals = z.reshape(n_paths, n_steps, dim)

        X = np.zeros((n_paths, n_steps + 1, state_dim))
        X[:, 0] = process.x0
        for i in range(n_steps):
            current_t = process.t0 + i * dt
            for p in range(n_paths):
                path_slice = X[p, :i + 1]
                X[p, i + 1], _ = integrator.step(
                    path_slice, current_t, dt,
                    process.a, process.B,
                    process.L, process.θ,
                )
        return X


"""
ParallelMonteCarlo	Use concurrent.futures or Ray for distributed path generation
GpuMonteCarlo	CuPy / PyTorch backend
ControlVariateMonteCarlo	Add control variate correction term
SobolAdaptive	Adaptive quasi-MC for path-dependent options
"""
