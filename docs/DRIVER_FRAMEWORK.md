Below is a clean, professional **Markdown summary** of your new stochastic **Driver Framework**.

You can save it as `DRIVER_FRAMEWORK.md`.

---

# **Stochastic Driver Framework — Summary**

This framework provides a **unified, extensible architecture** for building fully general stochastic processes used in financial modeling, simulation engines, and SDE systems of the form:

[
dX_t = A(X_{\le t}, t), dU_t ;+; B(X_{\le t}, t), dV_t.
]

It isolates all sources of randomness into modular **drivers**, allowing the construction of highly flexible, path-dependent, and mixed-type models, including:

* diffusions / semimartingales
* Lévy jump-diffusions
* fractional Brownian motion
* rough volatility kernels
* point processes (Poisson, Hawkes, marked processes)
* hybrids and correlated multi-factor models

---

## **1. Core Concept**

A **stochastic driver** is any object that, at each time-step, produces:

* a random increment
* metadata about the increment
* a structural description of its mathematical properties

Every driver satisfies the interface:

```python
class StochasticDriver(ABC):
    def step(t, dt, X_path, rng) -> (ΔV, meta): ...
    def get_characteristics(t, X_path) -> dict: ...
```

All drivers return increments `ΔV` as **NumPy vectors**, with dimension `dim`.

---

## **2. Atomic Driver Types**

The framework defines **three fundamental atomic driver classes**, each representing a major mathematical category of stochastic processes:

---

### **2.1 SemimartingaleDriver**

Represents general **Itô semimartingales** via a path/time-dependent **local Lévy triplet**:

[
(b_t,, C_t,, \nu_t).
]

Supports:

* Brownian motion (multi-dimensional, correlated)
* state-dependent volatility and drift
* Lévy jump-diffusions (Merton, Kou, CGMY, NIG, VG, etc.)
* local/stochastic volatility
* regime switching
* time-inhomogeneous or local Lévy models

**Key idea:**
`drift_fn` and `cov_fn` define the continuous part;
`jump_fn` simulates arbitrary jumps per time-step.

---

### **2.2 GaussianKernelDriver**

Represents **Gaussian processes not expressible as semimartingales**, defined via increment covariance:

[
\Delta V_t \sim \mathcal{N}(\mu_t,, \Sigma_t).
]

Supports:

* fractional Brownian motion (any Hurst H)
* rough volatility models (Volterra kernels)
* general Gaussian processes with memory
* multi-dimensional correlated Gaussian drivers

The covariance `Sigma(t, dt)` may encode **non-Markovian kernels**, long memory, or roughness.

---

### **2.3 PointProcessDriver**

Represents **jump/event processes** defined by:

* an intensity function ( \lambda(t, X_{\le t}) )
* a random mark distribution

Supports:

* Poisson and state-dependent Poisson
* Compound Poisson
* Hawkes / self-exciting processes
* marked point processes
* general jump dynamics with internal states

Returns the **sum of all jump marks** in the interval ([t, t+dt]) and metadata about individual events.

---

## **3. CompositeDriver**

The **CompositeDriver** is the master driver used by the SDE engine.
It takes a list of atomic drivers and **concatenates all their increments**:

```python
ΔV = concat(ΔV^(1), ΔV^(2), ..., ΔV^(K))
```

Thus, if you combine:

* a 2D Brownian driver
* a 1D fBM driver
* a 3D jump driver

the composite driver produces a **6-dimensional** increment vector.

CompositeDriver also aggregates metadata and characteristics from sub-drivers.

### **Important rule:**

**Correlated components must belong to the same atomic driver.**

For example:

* two correlated Brownian motions → one SemimartingaleDriver(dim=2)
* correlated fBM factors → one GaussianKernelDriver(dim=2)
* Brownian–fBM correlation → one GaussianKernelDriver(dim=2) with block covariance

CompositeDriver does **not** create correlation between different drivers.

---

## **4. Advantages of This Framework**

* **Modularity**
  Each source of randomness is isolated and swappable.

* **Generality**
  Supports almost any stochastic model used in modern quant finance, including rough volatility.

* **Path and time dependence**
  Drift, volatility, kernels, intensities, and jump laws can all depend on the full path.

* **Unified interface**
  The SDE engine only needs to interact with CompositeDriver and UDriver.

* **Correct handling of correlation**
  Correlation is encoded inside the atomic drivers, reflecting true Gaussian structure.

* **Extensible**
  Developers can add new drivers (e.g., regime-switching drivers, neural SDE drivers) without changing the engine.

---

## **5. High-Level Usage Flow**

1. **Define atomic drivers:**

   ```python
   brownian = SemimartingaleDriver(...)
   fbm      = GaussianKernelDriver(...)
   jumps    = PointProcessDriver(...)
   ```

2. **Merge them:**

   ```python
   V_driver = CompositeDriver([brownian, fbm, jumps])
   ```

3. **Use them in the SDE:**
   [
   dX = A(X,t),dU + B(X,t),dV.
   ]

4. The engine only sees:

   ```python
   ΔV, meta = V_driver.step(t, dt, X_path, rng)
   ```

---

## **6. What This Framework Enables**

This system can produce:

* GBM, Heston, SABR
* local & stochastic volatility
* rough Bergomi, rough Heston
* Merton, Kou, CGMY, Bates
* Hawkes-based volatility or jumps
* multi-factor correlated Brownian–fBM–Lévy drivers
* fully path-dependent and state-dependent dynamics

with the same unified SDE solver.

---

## **7. Next Extensions (Optional)**

The framework is ready for:

* Time-change drivers (e.g. subordinators)
* Regime-switching drivers (CTMC-driven)
* Neural SDE drivers (learned drift/cov/kernels/intensities)
* Calibration utilities using characteristics
* JSON specifications for model loading

