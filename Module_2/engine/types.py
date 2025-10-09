# sde/types.py
import pandas as pd
from typing import Protocol, Tuple, runtime_checkable, TypeVar, Any, ClassVar, Callable, Dict
import numpy as np
from dataclasses import replace

Array = np.ndarray


@runtime_checkable
class _DataclassLike(Protocol):
    # Match the structural shape expected by dataclasses.replace
    __dataclass_fields__: ClassVar[dict]


SelfModel = TypeVar("SelfModel", bound="Model")

DriftFn = Callable[[Array, float, Dict[str, Any]], Array]  # (x,t,θ)->R^N
DiffusionFn  = Callable[[Array, float, Dict[str, Any]], Array]  # (x,t,θ)->R^{N×K}
class Model(_DataclassLike, Protocol):
    # Required attributes
    dim: int
    init: Array

    # Likelihood and SDE hooks
    def drift(self, x: Array, t: float) -> Array: ...
    def diffusion(self, x: Array, t: float) -> Array: ...
    def diffusion_dx(self, x: Array, t: float) -> Array: ...
    def jumps(self, x: Array, dt: float) -> Array: ...
    def extra_terms(self, x: Array, t: float, dt: float, dW: Array) -> Array: ...

    # Default implementation for calibrator-based fit
    def fit(self: SelfModel, data, calibrator=None, **kwargs) -> tuple[SelfModel, Any]:
        if calibrator is None:
            calibrator = self._default_calibrator()
        params, fit_result = calibrator.fit(self, data, **kwargs)
        return replace(self, **params), fit_result

    def _default_calibrator(self):
        raise NotImplementedError("Each model must define a default calibrator")


class Payoff(Protocol):
    def __call__(self, paths: Array) -> Array: ...


class Discretization(Protocol):
    def step(self, model: Model, x: Array, t: float, dt: float, dW: Array) -> Array: ...


@runtime_checkable
class RegimeSwitching(Protocol):
    """Optional regime-switching hooks. Implement to enable regime simulation without
    importing concrete model types in the simulator (avoids circular imports)."""
    def prepare_regimes(self, N: int, T: int, rng) -> None: ...
    def advance_regime(self) -> None: ...
