# Module 3: Execution & Risk Layer — specification (math + implementation notes)

1. Purpose and placement
Module 3 is the deterministic gate between the modeling sandbox (Module 2) and live execution: it enforces KYC-linked limits, performs per-order pre-trade checks, executes via the execution engine, and applies portfolio & systemic risk controls. All decisions are auditable and parameterized by mathematically defined thresholds.

2. High-level control flow (per order)
3. Receive order request from Module 2 (includes strategy id, account id, instrument, side, size, price type, latency tolerance, urgency).
4. Lookup KYC profile & account limits.
5. Compute pre-trade risk metrics (exposure, margin impact, incremental VaR, liquidity score, potential slippage estimate).
6. Evaluate hard constraints (margin, instrument permissions, max position, leverage). If any hard constraint violated → reject.
7. Apply soft constraints and adaptive rules (dynamic position sizing, volatility caps). May reduce order size or convert to limit/TWAP.
8. If passed, send to execution engine (smart-routing + chosen algo).
9. Monitor fill, slippage, partial fills. Apply real-time stop-loss adjustments or cancel remaining legs if conditions met.
10. Log full audit trail and feed execution metrics back into reliability models and Module 2.

11. KYC-linked risk mapping
- Each account has a KYC profile vector k containing parameters:
  - C = capital (free equity)
  - L_max = maximum allowed leverage
  - P_max = maximum single-position size (absolute / % of capital)
  - I_allow = allowed instrument set
  - Jurisdictional rules (tax/regulatory flags)
  - Behavioral/AML flags (suspicious trading thresholds)

Enforcement: before any order, verify:
- |pos_i + Δpos| ≤ P_max
- (notional exposure / C) ≤ L_max (or margin adequacy test below)
- instrument ∈ I_allow

4. Margin and collateral math
- For futures/derivatives: required margin M computed as:
  M = max(M_init, α * σ_instr * sqrt(T) * notional)
  where α is a multiplier based on instrument risk and σ_instr is realized/forecast vol.

- Margin adequacy check:
  C - M ≥ M_min_buffer

5. Exposure, portfolio variance and incremental risk
- Current portfolio exposure vector x (notional by asset). New candidate delta Δx.
- Portfolio variance:
  Var_p = x^T Σ x
- Incremental variance from order:
  ΔVar = 2 x^T Σ Δx + Δx^T Σ Δx
- Enforce exposure constraint:
  Var_p + ΔVar ≤ Var_max

6. Value-at-Risk (VaR) and CVaR (Expected Shortfall)
- Parametric (Gaussian):
  VaR_α = z_(1-α) * sqrt(x^T Σ x)
- Incremental VaR:
  ΔVaR = VaR(x+Δx) - VaR(x)
- CVaR:
  CVaR_α = E[L | L > VaR_α]

7. Liquidity, slippage and market impact models
- Almgren–Chriss expected cost:
  ExpectedCost = η * (X^2 / T) + γX
- Slippage estimator:
  SlippageEstimate = f(ADV, spread, depth, urgency)
- Fill probability:
  P(fill in Δt) = 1 - e^(-λΔt)

8. Execution optimization
- Minimize expected execution cost + impact + risk penalty:
  min Σ [p_t v_t + impact(v_t) + risk_penalty]

9. Real-time pre-trade check pseudocode
function pre_trade_check(order, account, portfolio, market_state):
    kyc = load_kyc(account)
    if order.instrument not in kyc.allowed_instruments: reject
    notional = order.size * market_state.mid_price
    if notional > kyc.P_max: reject or reduce_size()
    margin_req = compute_margin(order, market_state)
    if account.cash - margin_req < margin_buffer: reject
    delta_var = compute_incremental_var(portfolio.x, order.delta_x, Sigma)
    if delta_var > var_limit_per_order: reject
    slippage = estimate_slippage(order, market_state)
    if slippage > slippage_threshold: convert_to_twap_or_reject()
    decide_execution_algo(order, slippage, urgency, liquidity)
    return approved_with_algo

10. Circuit breakers and kill-switches
- Strategy-level kill: PnL loss > L_strategy_limit → disable
- Account-level: margin ratio < threshold → liquidate
- Global: system volatility/margin shortfall exceeds emergency threshold → pause execution

11. Statistical monitoring
- Update execution parameters with EWMA:
  θ_(t+1) = λθ_t + (1-λ)θ̂_t
- Bayesian update of source reliability:
  P(source good | data) ∝ P(data | source good) * P(source good)

12. Stress testing
- Scenario simulations: volatility shock, liquidity drought, jump diffusion events.
- Use jump-diffusion for shocks: price jump J with intensity λ_L.

13. Auditability and compliance
- Immutable logs of checks, orders, fills.
- Deterministic replay.
- AML anomaly detection.

14. Calibration
- Parameters (Σ window, α, β, η, γ, VaR levels, buffers) versioned, logged, backtested before deployment.

15. Integration with other modules
- Consumes Module 2 outputs (probabilities, forecasts).
- Module 4 simulates execution + risk checks for calibration.