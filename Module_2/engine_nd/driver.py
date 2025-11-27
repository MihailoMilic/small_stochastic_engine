from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, List
import numpy as np


class StochasticDriver(ABC):
    """
    Abstract base class for all stochastic drivers.
    
    Each driver must implement:
        - step(t, dt, X_path, rng): 
            returns a random increment ΔV_t for the interval [t, t+dt]
        
        - get_characteristics(t, X_path): 
            returns a structured description of the driver at time t
            (drift, covariance, Lévy measure, kernel, intensity, etc.)
    
    This abstraction is universal and covers:
        - Semimartingales (via local Lévy triplet)
        - Gaussian Volterra / fBM kernels
        - Hawkes & marked point-processes
        - Arbitrary path/time-dependent drivers
    """
    def __init__(self):
        self.dim: int = 0  # Dimension of the driver increment
    @abstractmethod
    def step(
        self, 
        t: float, 
        dt: float, 
        X_path: Any, 
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Sample the increment ΔV_t over the interval [t, t+dt].

        Parameters
        ----------
        t : float
            Current time.
        dt : float
            Time increment.
        X_path : Any
            History or state information (user-defined structure).
        rng : np.random.Generator
            Numpy random number generator.

        Returns
        -------
        ΔV : np.ndarray
            The increment vector for the driver at this time step.
        
        meta : dict
            Additional diagnostic information (jump times, marks, 
            intermediate quantities, etc.). Optional but useful.
        """
        pass

    @abstractmethod
    def get_characteristics(
        self, 
        t: float, 
        X_path: Any
    ) -> Dict[str, Any]:
        """
        Return a dictionary describing the mathematical structure 
        of the driver at time t. 
        
        Examples:
            Semimartingale:
                {"type": "semimartingale", "b": ..., "C": ..., "nu": ...}

            Gaussian kernel:
                {"type": "gaussian_kernel", "kernel": ..., "H": ...}

            Point process:
                {"type": "point_process", "λ": ..., "mark_law": ...}

        Parameters
        ----------
        t : float
            Current time.
        X_path : Any
            Path/history information.

        Returns
        -------
        characteristics : dict
            Fully structured description of the driver.
        """
        pass
    pass

class SemimartingaleDriver(StochasticDriver):
    """
    General semimartingale driver defined by a local Lévy triplet:
        - b_t  : drift vector
        - C_t  : covariance matrix (continuous martingale part)
        - nu_t : Lévy measure / jump specification (general jump part)

    b_t, C_t, nu_t may depend on:
        - time t
        - full path X_path
        - internal states
        - arbitrary user-defined context

    This class supports:
        - Brownian motion
        - Correlated Brownian motion
        - Local vol/stochastic vol continuous drivers
        - Merton/Kou jumps
        - Arbitrary Lévy jump structures
    """

    def __init__(
        self,
        dim: int,
        drift_fn: Callable[[float, Any], np.ndarray],
        cov_fn: Callable[[float, Any], np.ndarray],
        jump_fn: Optional[Callable[[float, float, Any, np.random.Generator], Tuple[np.ndarray, Dict[str, Any]]]] = None,
    ):
        """
        Parameters
        ----------
        dim : int
            Dimension of the driver (dim = number of coordinates in dV_t)

        drift_fn : function(t, X_path) -> np.ndarray shape (dim,)
            Drift vector b_t.

        cov_fn : function(t, X_path) -> np.ndarray shape (dim, dim)
            Instantaneous covariance matrix C_t for the continuous martingale part.

        jump_fn : optional function(t, dt, X_path, rng) -> (jump_vector, meta)
            A callable that simulates the jump increment over the interval [t, t+dt].
            If None, the process has NO jumps.
        """
        self.dim = dim
        self.drift_fn = drift_fn
        self.cov_fn = cov_fn
        self.jump_fn = jump_fn

    # ----------------------------------------------------------------------
    # STEP — simulate increment ΔV_t
    # ----------------------------------------------------------------------
    def step(
        self,
        t: float,
        dt: float,
        X_path: Any,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, Dict[str, Any]]:

        # 1) Drift contribution
        b_t = self.drift_fn(t, X_path)
        drift_part = b_t * dt

        # 2) Continuous martingale (Brownian / general diffusion)
        C_t = self.cov_fn(t, X_path)
        # Ensure symmetric positive semidefinite
        C_t = 0.5 * (C_t + C_t.T)

        # Cholesky factor for sampling normal increments
        # If matrix is not strictly PSD, np.linalg.cholesky would fail.
        # So we use eigh() -> clamp eigenvalues -> reconstruct.
        eigvals, eigvecs = np.linalg.eigh(C_t)
        eigvals_clamped = np.clip(eigvals, 0.0, None)
        sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals_clamped)) @ eigvecs.T

        # Standard normal vector scaled by sqrt(dt)
        dW = rng.normal(size=self.dim)
        continuous_part = sqrt_C @ dW * np.sqrt(dt)

        # 3) Jump part
        if self.jump_fn is None:
            jump_part = np.zeros(self.dim)
            jump_meta = {}
        else:
            jump_part, jump_meta = self.jump_fn(t, dt, X_path, rng)

        # Total increment
        ΔV = drift_part + continuous_part + jump_part

        # Metadata for diagnostics
        meta = {
            "drift": drift_part,
            "continuous": continuous_part,
            "jump": jump_meta,
        }

        return ΔV, meta

    # ----------------------------------------------------------------------
    # CHARACTERISTICS — return the local Lévy triplet
    # ----------------------------------------------------------------------
    def get_characteristics(
        self,
        t: float,
        X_path: Any
    ) -> Dict[str, Any]:
        """
        Return local Lévy triplet (b_t, C_t, ν_t).

        ν_t is represented abstractly because jump_fn may be anything.
        """

        b_t = self.drift_fn(t, X_path)
        C_t = self.cov_fn(t, X_path)

        return {
            "type": "semimartingale",
            "dim": self.dim,
            "b": b_t,
            "C": C_t,
            "nu": "implicit (via jump_fn)" if self.jump_fn else None,
        }




class GaussianKernelDriver(StochasticDriver):
    """
    Gaussian driver defined via a (possibly path/time-dependent) covariance
    structure for increments. This is meant for processes that may NOT be
    semimartingales, e.g.:

        - fractional Brownian motion (H ≠ 1/2)
        - rough volatility kernels
        - general Volterra Gaussian processes

    The key idea:
        Given (t, dt, X_path), the driver returns a Gaussian increment:
            ΔV ~ N(mean_fn(t, dt, X_path),
                   cov_fn(t, dt, X_path))

    Note:
        - cov_fn can encode ANY memory / kernel structure. It does NOT assume
          independent increments or linear scaling in dt.
        - You are free to store and update any internal state inside this class
          to implement fBM/rough-vol efficiently (e.g. Markovian lifts, precomputed
          covariance matrices, etc.).
    """

    def __init__(
        self,
        dim: int,
        cov_fn: Callable[[float, float, Any], np.ndarray],
        mean_fn: Optional[Callable[[float, float, Any], np.ndarray]] = None,
        kernel_descriptor: Optional[Dict[str, Any]] = None,
    ):
        """
        Parameters
        ----------
        dim : int
            Dimension of the driver (size of ΔV).

        cov_fn : function(t, dt, X_path) -> np.ndarray shape (dim, dim)
            Conditional covariance of the Gaussian increment over [t, t+dt].
            This can encode any kernel / Volterra structure you like.

        mean_fn : optional function(t, dt, X_path) -> np.ndarray shape (dim,)
            Conditional mean of the increment. If None, mean is zero.

        kernel_descriptor : optional dict
            Structural description of the kernel / process, e.g.:
                {"kind": "fbm", "H": 0.1, "sigma": 0.2}
                {"kind": "rough_volterra", "params": {...}}
            This is used only in get_characteristics for introspection and
            calibration metadata; simulation uses cov_fn / mean_fn.
        """
        self.dim = dim
        self.cov_fn = cov_fn
        self.mean_fn = mean_fn
        self.kernel_descriptor = kernel_descriptor or {}

    # ------------------------------------------------------------------
    # STEP — simulate increment ΔV_t
    # ------------------------------------------------------------------
    def step(
        self,
        t: float,
        dt: float,
        X_path: Any,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, Dict[str, Any]]:

        # 1) Mean of the increment
        if self.mean_fn is None:
            mean = np.zeros(self.dim)
        else:
            mean = self.mean_fn(t, dt, X_path)

        # 2) Covariance of the increment (may encode non-Markovian kernel)
        Sigma = self.cov_fn(t, dt, X_path)
        # Ensure symmetry
        Sigma = 0.5 * (Sigma + Sigma.T)

        # 3) Sample from N(mean, Sigma)
        #    Use eigen decomposition to handle semi-definite matrices robustly
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals_clamped = np.clip(eigvals, 0.0, None)
        sqrt_Sigma = eigvecs @ np.diag(np.sqrt(eigvals_clamped)) @ eigvecs.T

        z = rng.normal(size=self.dim)
        ΔV = mean + sqrt_Sigma @ z

        meta = {
            "mean": mean,
            "cov": Sigma,
        }

        return ΔV, meta

    # ------------------------------------------------------------------
    # CHARACTERISTICS — describe the Gaussian kernel
    # ------------------------------------------------------------------
    def get_characteristics(
        self,
        t: float,
        X_path: Any
    ) -> Dict[str, Any]:
        """
        Return a high-level description of the Gaussian kernel structure.
        We don't try to expose a Lévy triplet (it may not exist); instead we
        return whatever structural metadata was provided at construction.
        """

        return {
            "type": "gaussian_kernel",
            "dim": self.dim,
            "representation": "increment_covariance",
            "kernel_descriptor": self.kernel_descriptor,
        }



class PointProcessDriver(StochasticDriver):
    """
    General jump driver defined by an intensity λ(t, X_path) and a mark
    distribution M(t, X_path, rng).

    This includes:
        - Poisson processes
        - State-dependent Poisson
        - Compound Poisson jumps (random mark sizes)
        - Hawkes processes (self-exciting)
        - Marked point processes

    The driver returns ΔV_t, which is the *sum of all jump marks*
    on the interval [t, t+dt], and also returns metadata containing
    the individual events.
    """

    def __init__(
        self,
        dim: int,
        intensity_fn: Callable[[float, Any], float],
        mark_fn: Callable[[float, Any, np.random.Generator], np.ndarray],
        hawkes_update_fn: Optional[
            Callable[[float, float, Any], Any]
        ] = None,
        internal_state: Optional[Any] = None,
    ):
        """
        Parameters
        ----------
        dim : int
            Dimension of the jump increment ΔV (mark dimension).

        intensity_fn : function(t, X_path) -> float
            Instantaneous jump intensity λ(t).

        mark_fn : function(t, X_path, rng) -> np.ndarray shape (dim,)
            Random mark generator. This defines the jump size distribution.

        hawkes_update_fn : optional function(t, dt, internal_state) -> new_internal_state
            If provided, this is used to update internal intensity state
            (for Hawkes processes or general self-exciting dynamics).

        internal_state : optional
            Arbitrary structure to store state for self-exciting processes.
        """
        self.dim = dim
        self.intensity_fn = intensity_fn
        self.mark_fn = mark_fn
        self.hawkes_update_fn = hawkes_update_fn
        self.internal_state = internal_state

    # ------------------------------------------------------------------
    # STEP — simulate jump increment ΔV_t
    # ------------------------------------------------------------------
    def step(
        self,
        t: float,
        dt: float,
        X_path: Any,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, Dict[str, Any]]:

        # Update self-exciting intensity state if needed (Hawkes)
        if self.hawkes_update_fn is not None:
            self.internal_state = self.hawkes_update_fn(t, dt, self.internal_state)

        # Compute instantaneous intensity
        λ_t = self.intensity_fn(t, X_path)

        # Expected number of jumps on [t, t+dt]
        Λ = λ_t * dt

        # Number of jumps ~ Poisson(Λ)
        N = rng.poisson(Λ)

        jumps: List[np.ndarray] = []
        total = np.zeros(self.dim)

        # Sample each jump
        for _ in range(N):
            mark = self.mark_fn(t, X_path, rng)
            jumps.append(mark)
            total += mark

        meta = {
            "λ_t": λ_t,
            "expected_jumps": Λ,
            "num_jumps": N,
            "jumps": jumps,
        }

        return total, meta

    # ------------------------------------------------------------------
    # CHARACTERISTICS
    # ------------------------------------------------------------------
    def get_characteristics(
        self,
        t: float,
        X_path: Any
    ) -> Dict[str, Any]:

        λ_t = self.intensity_fn(t, X_path)

        return {
            "type": "point_process",
            "dim": self.dim,
            "λ_t": λ_t,
            "mark_law": self.mark_fn,
            "self_exciting": self.hawkes_update_fn is not None,
        }



class CompositeDriver(StochasticDriver):
    """
    Composite stochastic driver.

    Combines multiple StochasticDriver instances into a single
    multi-dimensional driver:

        ΔV = concat(ΔV^(1), ΔV^(2), ..., ΔV^(K))

    where each ΔV^(k) comes from one of:
        - SemimartingaleDriver
        - GaussianKernelDriver
        - PointProcessDriver
        - or any other StochasticDriver subclass

    The engine only sees this CompositeDriver and treats its
    output as one big dV_t vector.
    """

    def __init__(self, drivers: List[StochasticDriver]):
        """
        Parameters
        ----------
        drivers : list of StochasticDriver
            The atomic drivers to be combined. Each driver is assumed
            to have a `.dim` attribute giving the dimension of its
            increment vector.
        """
        if not drivers:
            raise ValueError("CompositeDriver requires at least one driver.")

        self.drivers = drivers

        # Infer global dimension from sub-drivers
        dims = []
        for i, d in enumerate(drivers):
            if not hasattr(d, "dim"):
                raise AttributeError(
                    f"Driver at index {i} has no 'dim' attribute; "
                    "all sub-drivers must expose 'dim'."
                )
            dims.append(d.dim)

        self.sub_dims: List[int] = dims
        self.dim: int = int(sum(dims))

        # Precompute index ranges for each sub-driver
        self._ranges: List[Tuple[int, int]] = []
        start = 0
        for d in dims:
            end = start + d
            self._ranges.append((start, end))
            start = end

    # ------------------------------------------------------------------
    # STEP — simulate composite increment ΔV_t
    # ------------------------------------------------------------------
    def step(
        self,
        t: float,
        dt: float,
        X_path: Any,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, Dict[str, Any]]:

        # Allocate global increment
        ΔV = np.zeros(self.dim)
        meta: Dict[str, Any] = {
            "sub": [],  # metadata per sub-driver
        }

        # For each inner driver, simulate and place into the right slice
        for (driver, (start, end)) in zip(self.drivers, self._ranges):
            sub_increment, sub_meta = driver.step(t, dt, X_path, rng)
            # safety: broadcast/shape check
            sub_increment = np.asarray(sub_increment)
            if sub_increment.shape != (end - start,):
                raise ValueError(
                    f"Sub-driver produced increment of shape {sub_increment.shape}, "
                    f"expected {(end - start,)} for range [{start}, {end})."
                )
            ΔV[start:end] = sub_increment
            meta["sub"].append({
                "range": (start, end),
                "meta": sub_meta,
            })

        return ΔV, meta

    # ------------------------------------------------------------------
    # CHARACTERISTICS — aggregate sub-driver characteristics
    # ------------------------------------------------------------------
    def get_characteristics(
        self,
        t: float,
        X_path: Any
    ) -> Dict[str, Any]:
        """
        Returns a list of sub-characteristics plus global dimension info.

        Example structure:
            {
                "type": "composite",
                "dim": total_dim,
                "components": [
                    {
                        "range": (0, 2),
                        "characteristics": {... from driver 0 ...}
                    },
                    ...
                ]
            }
        """

        components = []
        for (driver, (start, end)) in zip(self.drivers, self._ranges):
            char = driver.get_characteristics(t, X_path)
            components.append({
                "range": (start, end),
                "dim": end - start,
                "characteristics": char,
            })

        return {
            "type": "composite",
            "dim": self.dim,
            "components": components,
        }
