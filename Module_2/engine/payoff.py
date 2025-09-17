# sde/payoff.py
import numpy as np
from .types import Array, Payoff

class EuropeanCall(Payoff):
    def __init__(self, strike: float, idx: int = 0):
        self.K = strike
        self.idx = idx

    def __call__(self, paths: Array) -> Array:
        ST = paths[-1, :, self.idx]
        return np.maximum(ST - self.K, 0.0)
