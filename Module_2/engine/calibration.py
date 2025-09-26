from .types import  Model, Discretization, Array
from typing import Optional, Protocol
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import dataclasses
@dataclasses.dataclass(frozen=True)
class MLEProtocol(Protocol):
    def loglike(self, theta:dict, data:pd.DataFrame, dt: float) -> float:
        raise NotImplementedError("Each model must implement loglike for MLE calibration")

class MLECalibrator:
    def __init__(self, method="L-BFGS-B", dt=1/252, maxiter=500, random_state=None):
        self.method = method
        self.maxiter = maxiter
        self.dt = dt
        self.rng = np.random.default_rng(random_state)

    def fit(self, model, data, start_params, bounds=None):
        """
        model: object with model.loglike(theta, data) implemented
        data: array of observations
        start_params: dict {param_name: initial_value}
        bounds: dict {param_name: (low, high)} optional
        """
        names = list(start_params.keys())
        x0 = np.array([start_params[n] for n in names])

        # Objective = negative log-likelihood
        def neg_ll(x):
            theta = {n: x[i] for i, n in enumerate(names)}
            return -model.loglike(theta, data, self.dt)

        # Convert bounds dict to list for optimizer
        if bounds is not None:
            bnds = [bounds.get(n, (None, None)) for n in names]
        else:
            bnds = None

        res = minimize(
            neg_ll,
            x0,
            method=self.method,
            bounds=bnds,
            options={"maxiter": self.maxiter}
        )

        theta_hat = {n: res.x[i] for i, n in enumerate(names)}
        return theta_hat, res
@dataclasses.dataclass(frozen = True)
class SurfaceProtocol(Protocol):
    def option_prices(self, theta:dict, strikes: Array, maturities: Array, s0: float, r: float) -> Array:
        raise NotImplementedError("Each model must implement option_prices for surface calibration")


class SurfaceCalibrator:
    def __init__(self, method="L-BFGS-B", maxiter=500, loss: str = "mse", eps: float = 1e-12):
        """
        loss: "mse" or "cross_entropy". Cross-entropy is computed on
              risk-neutral densities recovered from call prices per maturity
              via call-spread finite differences (Breeden–Litzenberger).
        eps:  small positive number for numerical stability.
        """
        self.method = method
        self.maxiter = maxiter
        self.loss = loss
        self.eps = float(eps)

    def _mse(self, model_prices: np.ndarray, market_prices: np.ndarray) -> float:
        return float(np.mean((model_prices - market_prices) ** 2))

    def _cross_entropy_prices(self,
                              strikes: np.ndarray,
                              maturities: np.ndarray,
                              market_prices: np.ndarray,
                              model_prices: np.ndarray,
                              r: float) -> float:
        """Cross-entropy between market and model risk-neutral densities per maturity.
        We approximate density mass across strike intervals using call-spread
        digital approximations:
          T(K_j) ≈ (C(K_j) - C(K_{j+1})) / (K_{j+1}-K_j) / df
          p_j (mass on [K_j, K_{j+1}]) ≈ T(K_j) - T(K_{j+1})
        Cross-entropy H(P_mkt, P_mod) = -Σ p_mkt ln(p_mod).
        Returns average over maturities where at least 3 strikes are present; otherwise falls back to MSE.
        """
        eps = self.eps
        Ts = np.asarray(maturities, dtype=float)
        Ks = np.asarray(strikes, dtype=float)
        Pm = np.asarray(market_prices, dtype=float)
        Pn = np.asarray(model_prices, dtype=float)

        uniq_T = np.unique(Ts)
        ce_vals = []
        for T in uniq_T:
            mask = (Ts == T)
            if np.count_nonzero(mask) < 3:
                continue
            k = Ks[mask]
            cm = Pm[mask]
            cn = Pn[mask]
            # sort by strike
            order = np.argsort(k)
            k = k[order]
            cm = cm[order]
            cn = cn[order]
            df = np.exp(-r * T)
            # first differences -> digitals (tail probs) at grid points (length n-1)
            dk = np.diff(k)
            if np.any(dk <= 0):
                continue
            dig_mkt = (cm[:-1] - cm[1:]) / dk / max(df, eps)
            dig_mod = (cn[:-1] - cn[1:]) / dk / max(df, eps)
            # clamp to [0,1]
            dig_mkt = np.clip(dig_mkt, 0.0, 1.0)
            dig_mod = np.clip(dig_mod, 0.0, 1.0)
            # interval masses between consecutive grid points (length n-2)
            pmass = dig_mkt[:-1] - dig_mkt[1:]
            qmass = dig_mod[:-1] - dig_mod[1:]
            pmass = np.clip(pmass, 0.0, None)
            qmass = np.clip(qmass, 0.0, None)
            s_p = pmass.sum()
            s_q = qmass.sum()
            if s_p <= eps or s_q <= eps:
                # degenerate; skip this maturity
                continue
            pmass /= s_p
            qmass = qmass / s_q
            ce = -float(np.sum(pmass * np.log(qmass + eps)))
            if np.isfinite(ce):
                ce_vals.append(ce)
        if ce_vals:
            return float(np.mean(ce_vals))
        # Fallback if insufficient grid
        return self._mse(model_prices, market_prices)

    def fit(self, model: SurfaceProtocol, market_data, start_params, bounds = None):
        """
        Market data: dict with keys {strikes, maturities, prices, spot, rate}
        """

        strikes, maturities, prices, s0, r = (
            market_data["strikes"],
            market_data["maturities"],
            market_data["prices"],
            market_data["spot"],
            market_data.get("rate", 0.0),
        )
        names = list(start_params.keys())
        x0 = np.array([start_params[n] for n in names])
        def objective(x):
            theta = {n: x[i] for i, n in enumerate(names)}
            model_new = dataclasses.replace(model, **theta)
            model_prices= model_new.option_prices(theta, strikes, maturities, s0, r)
            if self.loss == "mse":
                return self._mse(model_prices, prices)
            elif self.loss in ("cross_entropy", "xe"):
                return self._cross_entropy_prices(strikes, maturities, prices, model_prices, r)
            else:
                # default
                return self._mse(model_prices, prices)
        
        bounds= [bounds.get(n, (None, None)) for n in names] if bounds else None
        res = minimize(objective, x0, method = self.method, bounds=bounds, options={"maxiter": self.maxiter})
        theta_hat = {n: res.x[i] for i, n in enumerate(names)}
        return theta_hat, res
