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

## V2 — Phase 1: Polish & Solidify Core ✅
**Date:** March–April 2026

Designed a formal build plan. V2 = Phase 1 — clean up V1 mistakes and build a proper
foundation before adding new features.

---

### Codebase Review & Bug Fixes
**Date:** March 2025

Full review of all files before starting Phase 1 work. Two bugs found and fixed:

- **Bug 1 — Duplicate function in `monte_carlo.py`:** `plot_monte_carlo_paths()` was defined
  twice in the same file. The second definition silently shadowed the first. Removed the duplicate.

- **Bug 2 — Sharpe ratio missing risk-free rate (V1 `metrics.py`):** The Sharpe calculation
  divided mean return by std deviation with no risk-free subtraction. Added `risk_free_rate=0.0`
  parameter with proper daily scaling: `daily_rf = risk_free_rate / periods_per_year`.

---

### Professional Comments Added to All Files
**Date:** March 2025

Added docstrings and inline comments to every Python file in the project — V1, V2 strategies,
V2 core modules, and the ML baseline model. Goal: the codebase should be readable by anyone
picking it up cold, and defensible in an interview at line level.

---

### Phase 1.0 — Project Infrastructure
**Date:** March 2025

- Created `.gitignore` covering `__pycache__`, `.venv`, `.DS_Store`, `*.csv`, `*.json`, `results/`
- Rewrote root `README.md` with full project structure tree and versioning roadmap table
- Wrote `V1/README.md` from scratch (was empty)
- Updated `V2/README.md` with current structure, all 3 strategies listed, and 5-phase roadmap

---

### Phase 1.1 — Trade Log Overhaul (`trade_log.py`)
**Date:** March 2025

Rebuilt the trade data structure to be production-grade.

**Changes made:**
- Added `trade_id` field — UUID4 generated at creation time. Every trade is uniquely identifiable,
  which matters for audit trails and linking trades across systems (e.g. connecting a backtest
  trade to a live execution record)
- Added `duration` field — computed automatically as `exit_time - entry_time` (a `pd.Timedelta`)
- Added full input validation in `create_trade()` — guards against nonsensical trades:
  - Long stop must be below entry; short stop must be above entry
  - Target must be on the profit side of entry
  - Prices must be positive
  - Direction must be "long" or "short"
- Added `export_trades_csv(trades, path)` and `export_trades_json(trades, path)` — one-line
  exports for saving results to disk without writing boilerplate every time

**Why this matters:** In prop firm challenges, you need a clean audit trail of every trade.
UUID trade IDs make it easy to match backtest trades to live fills when building the simulator.

---

### Phase 1.2 — Extended Performance Metrics (`performance.py`)
**Date:** March 2025

Added four new metrics that were missing from the original summary stats.

**Changes made:**
- `max_drawdown_pct()` — worst drawdown as % of peak equity, not just raw R units.
  Percentage form is what prop firms and interviewers expect to see.
- `sharpe_ratio()` — annualised Sharpe from trade-level R-multiples. Uses `trades_per_year`
  scaling factor. R-multiples are already excess returns so risk-free rate = 0 is correct here.
- `longest_losing_streak()` — maximum consecutive losing trades. Critical for prop firm
  challenge analysis — a 10-trade losing streak can hit daily loss limits even if the
  strategy has positive expectancy over 200 trades.
- `recovery_factor()` — `total_r / abs(max_drawdown_r)`. Measures how efficiently the
  strategy digs itself out of its worst hole. Above 2.0 is solid.

All four added to `summary_stats()` output dictionary.

---

### Phase 1.3 — Monte Carlo Risk Analysis (`monte_carlo.py`)
**Date:** March 2025

Added two new analytical functions to quantify risk across all simulated paths,
not just the final equity distribution.

**Changes made:**
- `probability_of_ruin(simulations, ruin_threshold)` — fraction of simulated paths
  where equity ever touched or dropped below the threshold. Default threshold = 0.0
  (catastrophic ruin), but accepts any level — e.g. pass `0.9` to model a prop firm
  that cuts you at a 10% drawdown from a 1.0R starting balance.
- `drawdown_distribution(simulations, starting_equity)` — for every simulated path,
  computes the worst peak-to-trough drawdown as a percentage. Returns the full
  distribution (mean, median, p5, p95, absolute worst) so you can see not just one
  backtest's drawdown but the range of drawdowns the strategy might produce in the future.
- `monte_carlo_summary()` updated — now includes both new functions in one call.
  Added `ruin_threshold` parameter so the summary is directly configurable for
  different prop firm account rules.

**Why this matters:** Probability of ruin and drawdown distribution are the two numbers
that determine whether a strategy can survive a prop firm challenge. A strategy might
show positive expectancy but have a 25% probability of hitting the max drawdown limit
before reaching the profit target.

---

### Phase 1.4 — Enhanced Visualisation (`plots.py`)
**Date:** March 2025

Added three new plotting capabilities to complement the existing equity/drawdown/R-distribution charts.

**Changes made:**
- `plot_equity_curve_with_mc_bands(trades, simulations)` — overlays the real backtest equity
  curve on shaded Monte Carlo percentile bands (p5/p95 outer, p25/p75 inner). If the actual
  backtest path hugs the top of the distribution, it likely represents a lucky sequence rather
  than the strategy's true expected performance.
- `plot_win_loss_streak(trades)` — bar chart of running streak length at each trade (green =
  win streak, red = loss streak). Helps spot loss clustering — if losses bunch together rather
  than appearing randomly, it suggests the strategy has regime dependency (breaks down in
  specific market conditions).
- `save_all_figures(output_dir)` — saves every open matplotlib figure to disk as PNG. Call
  before plt.show() to capture results without manual saving. Creates `results/` directory
  automatically.
- `plot_all()` updated to include `plot_win_loss_streak()` automatically.

---

### Phase 1.5 — Centralised Configuration (`config.py`)
**Date:** April 2025

Redesigned `config.py` around a key principle: parameters belong in Config only if
they are **universal trading concepts** shared across all strategies. Strategy-specific
params that no other strategy would use stay as defaults inside the strategy file itself.

**Design rule applied:** if you add a new strategy tomorrow and it can read Config
without modification, the param belongs here. If it only makes sense for one strategy,
it stays in that strategy's file.

**What lives in Config (universal):**
- Data settings: ticker, symbol, dates, interval, data source
- Session times: `SESSION_START`, `SESSION_END`, `FUTURES_SESSION_END`
- Risk primitives: `RR_RATIO`, `OR_MINUTES`, `SLIPPAGE_PER_SHARE`, `ENTRY_SLIP_TICKS`, `EXIT_SLIP_TICKS`
- Sizing: `ACCOUNT_SIZE`, `RISK_PER_TRADE`
- Monte Carlo: `N_SIMULATIONS`, `STARTING_EQUITY`, `RUIN_THRESHOLD`
- Plotting: `N_PATHS_TO_PLOT`, `RESULTS_DIR`

**What stays in strategy files (strategy-specific):**
- Key Level ORB: `sl_scan_low/high/default`, Asia/London session windows,
  zone times, large range threshold, 4R target (vs standard 2R)
- These parameters would be meaningless noise in a universal config

**`__main__` demo blocks updated** in all three strategy files to reference `Config`
instead of inline literals — so changing a date range or risk % in one place updates
every strategy's demo run simultaneously.

---

---

### Phase 1.6 — Data Loader (`data_loader.py`)
**Date:** April 2026

Built a clean data ingestion layer using the Abstract Base Class pattern.

**What was built:**
- `DataProvider` ABC — defines the `fetch()` contract every provider must implement.
  No strategy or backtester needs to know which data source it's talking to.
- `YFinanceProvider` — free equities/ETF data. Chunks requests to stay within yFinance's
  API history limits (e.g. 5m data only available for the last 60 days, downloaded in
  59-day chunks). Clips start dates with a warning rather than silently returning empty data.
- `DatabentoProvider` — institutional futures data. Handles Databento's continuous futures
  symbol format (`ES.c.0`), schema mapping (`5m → ohlcv-5m`), and fixed-point price
  conversion (nanodollars ÷ 1e9 → dollars). Requires a paid API key.

**Key design decisions:**
- Providers are interchangeable — swap `YFinanceProvider` for `DatabentoProvider` in one line.
- Both normalise output to the same format: lowercase OHLCV columns, tz-naive Eastern time index.
- Duplicate rows removed after concat to handle chunk boundary overlap.

---

### Roadmap Restructure — ML Pipeline Promoted, Prop Firm Work Deferred
**Date:** April 2025

**Decision:** Deprioritised prop firm challenge simulator. Promoted ML signal generation
to the centrepiece of the project. Rationale: gradient boosting + walk-forward validation
is a significantly stronger story for quant/AI internship recruiting in 2026–2027 than
prop firm pass rate calculations.

**What changed:**
- Old Phase 2 (prop firm simulator) → parked. Will be reframed as a generic configurable
  risk constraints module inside the dashboard if revived, not a standalone feature.
- Walk-forward validation promoted to Phase 2 — it is a prerequisite for any rigorous ML
  work. Without it, gradient boosting results have lookahead bias and are meaningless.
- Gradient boosting signal generator becomes Phase 3 — the centrepiece of the project.
- Dashboard moved to Phase 4 — it should visualise content that actually matters, and the
  ML work produces the most interesting content to display.
- Live trading deferred until after Phase 3 ships.

---

## Roadmap

| Phase | Version | Goal | Target |
|-------|---------|------|--------|
| Phase 1 | V2 | Polish & solidify core ✅ | Complete — April 2026 |
| Phase 2 | V3 | Walk-forward validation framework ← *current* | 2–3 weeks |
| Phase 3 | V4 | Gradient boosting signal generator | 4–6 weeks |
| Phase 4 | V5 | Interactive dashboard | 2–3 weeks |
| Phase 5 | V6 | Second ML layer (NLP / regime detection / RL) | Summer 2026 |

**Deferred:** Prop firm simulator, live trading integration.

---

### Phase 2 Acceptance Criteria (V3)
- Rolling and expanding window walk-forward support
- Strict train / validate / test splits — temporal ordering enforced, no shuffling
- Zero lookahead bias anywhere in the pipeline
- Out-of-sample performance tracked per window
- Rolling Sharpe, rolling drawdown, stability metrics across windows

### Phase 3 Acceptance Criteria (V4)
- Feature engineering pipeline: returns over multiple horizons, rolling volatility,
  technical indicators, volume features — all computed without lookahead
- XGBoost or LightGBM model trained within each walk-forward window
- Signal output consumed by backtester like any other strategy
- Honest benchmarking against buy-and-hold and naive momentum baseline
- Feature importance analysis
- Target Sharpe after costs: 0.5–1.5. Anything above 1.5 treated as a red flag
  for lookahead bugs until proven otherwise.

### Phase 5 Options (V6 — pick one)
- **(a) NLP on financial text** — sentiment from news/earnings calls/FOMC statements
  feeding into a trading signal. Highest dual-use value, exercises transformer skills.
- **(b) Regime detection** — HMMs, clustering, or autoencoder to switch strategy
  parameters based on detected market regime.
- **(c) RL for execution** — PPO or DQN agent trained against the backtester for
  position sizing or execution.
- Default unless specified: option (a).

---

---

## V3 — Phase 2: Walk-Forward Validation Framework
**Date:** May 2026 — in progress

Copied V2 as the base. Adding `walk_forward.py` to enable rigorous out-of-sample
testing. Prerequisite for the gradient boosting work in V4.

---

*Updated after every major change. See git log for granular commit history.*
