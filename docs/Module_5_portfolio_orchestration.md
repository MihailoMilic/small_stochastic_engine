# Module 5: Portfolio Management & Orchestration

Purpose
Acts as the meta-layer: allocates capital, manages multi-strategy portfolios, and orchestrates workflows across Modules 1–4. Ensures consistency, reporting, and governance.

Core Components

1. Portfolio Construction Engine
- Combines strategies S_1 … S_n into weights w_i.
- Optimization objective:
  max_w (E[R_p] / σ_p)
  subject to w^T1 = 1, w ≥ 0
  where R_p = w^T r, σ_p^2 = w^T Σ w.
- Supports mean-variance, risk parity, Black–Litterman, Kelly sizing.

2. Capital Allocation & Leverage Control
- Allocates capital across accounts/strategies.
- Global leverage cap:
  Σ |Exposure_i| / Capital ≤ L_global
- Dynamic rebalancing based on vol, drawdown, performance.

3. Workflow Orchestration
- Sequence: Module 1 → Module 2 → Module 3 → Module 4.
- DAG/task scheduler for dependency management.
- Parallel strategy runs with containerized isolation.

4. Monitoring & Reporting
- Real-time dashboards: PnL, VaR, margin, leverage, latency, fills, rejects.
- Daily reports with attribution and compliance logs.
- Alerts: drawdown breaches, anomalies, failed jobs.

5. Governance & Compliance
- Version-controlled strategies & parameters.
- Pre-deployment approval workflow.
- Full audit trail with replayability.
- Integration with AML/KYC layers.

6. Feedback Loop
- Aggregates execution outcomes → updates weights, reliability scores, allocations.
- Failed strategies auto-decommissioned, successful ones scaled up.
- Backtest vs. live review for parameter recalibration.

---

Summary
- Module 4 = scientific testbed & simulator (before committing capital).
- Module 5 = portfolio manager/orchestrator, allocating risk-adjusted capital and managing governance across the system.
