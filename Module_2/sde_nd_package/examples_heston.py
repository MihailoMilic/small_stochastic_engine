
import numpy as np
from sde_nd_package.drivers import BrownianDriver
from sde_nd_package.integrators import FullTruncation
from sde_nd_package.markov import MarkovProcessND
from sde_nd_package.recipes import heston_a, heston_B

# Heston example with correlated Brownian via L
rho = -0.6
L = np.array([[1.0, 0.0],
              [rho, np.sqrt(1-rho*rho)]])
drv = BrownianDriver(L)

θ = dict(mu=0.02, kappa=1.5, theta_v=0.04, xi=0.6, rho=rho)
x0 = np.array([100.0, 0.04])

proc = MarkovProcessND(N=2, K=2, a=heston_a, B=heston_B,
                       levy_driver=drv, θ=θ, x0=x0,
                       integrator=FullTruncation(nonneg_indices=[1]))

out = proc.simulate(T=1.0, dt=1/252, n_paths=2)
print("t shape:", out["t"].shape, "X shape:", out["X"].shape)
print("S_T (2 paths) =", out["X"][:, -1, 0])
