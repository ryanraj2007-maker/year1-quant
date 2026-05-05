# Project Timeline

A chronological record of major milestones, design decisions, and features built.
Kept as a living document — updated after every significant change.

---

## V1 — Initial Build
**Date:** Early 2025

First working end-to-end pipeline. Goal was to get something running and understand
the fundamentals before worrying about code quality.

**What was built:**
- `data_loader.py` — pulls OHLCV data from yFinance with chunked downloading to work around API rate limits
- `indicators.py` — EMA, RSI, ATR, Bollinger Bands computed with pandas
- `strategies.py` — signal-based backtesting logic (entry/exit rules on indicator crossovers)
- `backtester.py` — iterates over signals and simulates trades
- `metrics.py` — Sharpe ratio, max drawdown, win rate, profit factor
- `main.py` — ties the pipeline together

**Known limitations at the time:**
- No R-multiple framework — P&L was in raw points/dollars
- No trade-level data structure (just aggregate stats)
- Sharpe ratio had no risk-free rate parameter
- No Monte Carlo simulation
- Hardcoded parameters scattered across files

---

## V2 — Phase 2: Polish & Solidify Core ✅
**Date:** March–April 2026

Clean up V1 mistakes and build a proper foundation before adding new features.

---

### Codebase Review & Bug Fixes
**Date:** March 2026

Full review of all files before starting Phase 2 work. Two bugs found and fixed:

- **Bug 1 — Duplicate function in `monte_carlo.py`:** `plot_monte_carlo_paths()` was defined
  twice in the same file. The second definition silently shadowed the first. Removed the duplicate.

- **Bug 2 — Sharpe ratio missing risk-free rate (V1 `metrics.py`):** The Sharpe calculation
  divided mean return by std deviation with no risk-free subtraction. Added `risk_free_rate=0.0`
  parameter with proper daily scaling: `daily_rf = risk_free_rate / periods_per_year`.

---

### Professional Comments Added to All Files
**Date:** March 2026

Added docstrings and inline comments to every Python file in the project.

---

### Phase 2.0 — Project Infrastructure
**Date:** March 2026

- Created `.gitignore` covering `__pycache__`, `.venv`, `.DS_Store`, `*.csv`, `*.json`, `results/`
- Rewrote root `README.md` with full project structure tree and versioning roadmap table
- Wrote `V1/README.md` from scratch (was empty)
- Updated `V2/README.md` with current structure and roadmap

---

### Phase 2.1 — Trade Log Overhaul (`trade_log.py`)
**Date:** March 2026

Rebuilt the trade data structure to be production-grade.

**Changes made:**
- Added `trade_id` field — UUID4 generated at creation time
- Added `duration` field — computed automatically as `exit_time - entry_time`
- Added full input validation in `create_trade()` — guards against nonsensical trades
- Added `export_trades_csv()` and `export_trades_json()` — one-line exports

---

### Phase 2.2 — Extended Performance Metrics (`performance.py`)
**Date:** March 2026

Added four new metrics that were missing from the original summary stats.

**Changes made:**
- `max_drawdown_pct()` — worst drawdown as % of peak equity
- `sharpe_ratio()` — annualised Sharpe from trade-level R-multiples
- `longest_losing_streak()` — maximum consecutive losing trades
- `recovery_factor()` — `total_r / abs(max_drawdown_r)`

All four added to `summary_stats()` output dictionary.

---

### Phase 2.3 — Monte Carlo Risk Analysis (`monte_carlo.py`)
**Date:** March 2026

**Changes made:**
- `probability_of_ruin(simulations, ruin_threshold)` — fraction of paths that hit the threshold
- `drawdown_distribution(simulations, starting_equity)` — worst drawdown distribution across all paths
- `monte_carlo_summary()` updated to include both new functions

---

### Phase 2.4 — Enhanced Visualisation (`plots.py`)
**Date:** March 2026

**Changes made:**
- `plot_equity_curve_with_mc_bands()` — actual equity curve overlaid on MC percentile bands.
  Supports `style="bands"` (shaded) and `style="paths"` (individual simulation lines)
- `plot_win_loss_streak()` — bar chart of running win/loss streak at each trade
- `save_all_figures()` — saves all open figures to disk as PNGs
- `plot_all()` updated to include MC plot and streak chart

---

### Phase 2.5 — Centralised Configuration (`config.py`)
**Date:** April 2026

Redesigned `config.py` around universal trading primitives only. Strategy-specific
params stay in the strategy file.

**What lives in Config:** data settings, session times, risk primitives, sizing, MC params, plotting.

---

### Phase 2.6 — Data Loader (`data_loader.py`)
**Date:** April 2026

Built a clean data ingestion layer using the Abstract Base Class pattern.

**What was built:**
- `DataProvider` ABC — defines the `fetch()` contract every provider must implement
- `YFinanceProvider` — free equities/ETF data with chunked downloading
- `DatabentoProvider` — institutional futures data with fixed-point price conversion

---

### Phase 2.7 — Test Strategy & Pipeline Verification (`test_strategy.py`, `main.py`)
**Date:** May 2026

Built a simple bullish-candle test strategy and wired up `main.py` to run the full pipeline end-to-end. Verified all Phase 2 modules work together correctly.

---

### Roadmap Restructure — ML Pipeline Promoted, Prop Firm Work Deferred
**Date:** April 2026

**Decision:** Deprioritised prop firm challenge simulator. Promoted ML signal generation
to the centrepiece of the project. Gradient boosting + walk-forward validation is a
significantly stronger story for quant/AI internship recruiting in 2026–2027.

---

## Roadmap

| Version | Phase | Goal | Status |
|---------|-------|------|--------|
| V1 | Phase 1 | Signal-based backtester (baseline) | ✅ Complete |
| V2 | Phase 2 | Polish & solidify core | ✅ Complete — May 2026 |
| V3 | Phase 3 | Walk-forward validation framework ← *current* | 🔨 In progress |
| V4 | Phase 4 | Gradient boosting signal generator | Planned |
| V5 | Phase 5 | Interactive dashboard | Planned |
| V6 | Phase 6 | Second ML layer (NLP / regime detection / RL) | Summer 2026 |

**Deferred:** Prop firm simulator, live trading integration.

---

### Phase 3 Acceptance Criteria (V3)
- Rolling and expanding window walk-forward support
- Strict temporal ordering enforced — no shuffling
- Zero lookahead bias anywhere in the pipeline
- Out-of-sample performance tracked per window
- Rolling Sharpe, rolling drawdown, stability metrics across windows

### Phase 4 Acceptance Criteria (V4)
- Feature engineering pipeline: returns over multiple horizons, rolling volatility,
  technical indicators, volume features — all computed without lookahead
- XGBoost or LightGBM model trained within each walk-forward window
- Signal output consumed by backtester like any other strategy
- Honest benchmarking against buy-and-hold and naive momentum baseline
- Feature importance analysis
- Target Sharpe after costs: 0.5–1.5. Anything above 1.5 = red flag for lookahead bugs

### Phase 6 Options (V6 — pick one)
- **(a) NLP on financial text** — sentiment from news/earnings/FOMC → trading signal
- **(b) Regime detection** — HMMs/clustering/autoencoder to switch strategy params
- **(c) RL for execution** — PPO/DQN agent trained against the backtester
- Default: option (a)

---

## V3 — Phase 3: Walk-Forward Validation Framework
**Date:** May 2026 — in progress

Copied V2 as the base. Adding `walk_forward.py` to enable rigorous out-of-sample
testing. Prerequisite for the gradient boosting work in V4.

---

*Updated after every major change. See git log for granular commit history.*
