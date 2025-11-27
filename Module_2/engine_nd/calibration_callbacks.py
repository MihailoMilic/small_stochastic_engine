import numpy as np
from typing import Callable, Dict, Any
from scipy.special import gamma
Array = np.ndarray
Params = Dict[str, Any]

# Gaussian observation: y|x ~ N(h(x,θ), R)
def loglik_gaussian(x: Array, y: Array, θ: Params, h: Callable[[Array, Params], Array], R: np.ndarray) -> float:
    r = y - h(x, θ)
    R = 0.5 * (R + R.T)
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0:
        return -1e9
    alpha = np.linalg.solve(R, r)
    return -0.5 * (r @ alpha + logdet + len(r) * np.log(2 * np.pi))

# Poisson observation: y_i|x ~ Poisson(λ_i(x,θ))
def loglik_poisson(x: Array, y: Array, θ: Params, rate_fn: Callable[[Array, Params], Array]) -> float:
    lam = np.maximum(rate_fn(x, θ), 1e-12)
    return float(np.sum(y * np.log(lam) - lam - np.log(np.maximum(np.arange(1, int(y.max()) + 1), 1e-12))))

# Student-t observation (robust heavy-tail)
def loglik_student(x: Array, y: Array, θ: Params, h: Callable[[Array, Params], Array], R: np.ndarray, v: float = 5.0) -> float:
    r = y - h(x, θ)
    d = len(r)
    Rinv = np.linalg.inv(R)
    q = r @ Rinv @ r
    c = np.log(np.linalg.det(R)) + d * np.log(np.pi * v) + np.log(gamma(v/2))
    return float(np.log(gamma((v + d)/2)) - 0.5*(c + (v + d)*np.log1p(q/v)))


# Mean Squared Error between model and market implied vols
def pricing_loss_iv(process, θ, market_vols, strikes, maturities, weight=None) -> float:
    process.θ = θ
    model_vols = process.pricer().implied_vol_surface(strikes, maturities)
    diff = model_vols - market_vols
    if weight is not None:
        diff *= weight
    return float(np.mean(diff ** 2))

# Mean Squared Error between model and market prices
def pricing_loss_price(process, θ, market_prices, strikes, maturities, weight=None) -> float:
    process.θ = θ
    model_prices = process.pricer().price_surface(strikes, maturities)
    diff = model_prices - market_prices
    if weight is not None:
        diff *= weight
    return float(np.mean(diff ** 2))

# Relative RMSE (scale-invariant)
def pricing_loss_relative(process, θ, market_vols, strikes, maturities) -> float:
    process.θ = θ
    model_vols = process.pricer().implied_vol_surface(strikes, maturities)
    return float(np.sqrt(np.mean(((model_vols - market_vols) / np.maximum(market_vols, 1e-8)) ** 2)))
